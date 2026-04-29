#!/usr/bin/env python3
"""Multi-target cache builder for Scope-C distillation ablation.

For each (image, question) pair we run a SINGLE LLaVA forward pass on the
prompt and save:
  - vit_hidden  : [576, D_vit]  fp16   --- LLaVA vision_feature_layer output
  - text_emb    : [D_clip]      fp32   --- CLIP text embedding
  - attn_L8     : [576]         fp32   --- L=8 prompt-side attention to images
  - attn_L16    : [576]         fp32   --- L=16 prompt-side attention to images
  - attn_L24    : [576]         fp32   --- L=24 prompt-side attention to images
  - attn_rollout: [576]         fp32   --- attention-rollout aggregated across all
                                          LLM layers (Abnar & Zuidema 2020),
                                          row = last text-token position.

(Earlier revisions also cached an `attn_answer` from a generate() call, but
the extra prefill roughly doubled cache time. Removed in favour of L8 / L16 /
L24 / rollout, which already cover shallow / middle / deep / aggregated
supervision targets.)

All target distributions are renormalised to sum to 1.
Sharded as torch .pt under <out>/shard_XXXX.pt for fast random access.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from papit.core.config import PAPITConfig
from papit.integration.llava import PAPITLlavaRunner

SHARD_SIZE = 100


@torch.no_grad()
def attn_to_images(att_layer: torch.Tensor, img_start: int, img_end: int,
                   query_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sum attention from selected query positions to image keys.

    att_layer: [H, q, k] (head-stacked) or [q, k] (head-averaged).
    Returns: [n_img] fp32.
    """
    if att_layer.dim() == 3:
        att_layer = att_layer.mean(dim=0)
    if query_mask is None:
        query_mask = torch.ones(att_layer.shape[0], dtype=torch.bool, device=att_layer.device)
        query_mask[img_start : img_end + 1] = False
    return att_layer[query_mask][:, img_start : img_end + 1].sum(dim=0).float()


@torch.no_grad()
def attention_rollout(attentions: tuple[torch.Tensor, ...], img_start: int,
                      img_end: int) -> torch.Tensor:
    """Abnar & Zuidema 2020 attention rollout.
    attentions: tuple of [B=1, H, L, L]. We head-average, add residual, normalise,
    then chain-multiply across layers. Take last text-token row, slice image keys.
    """
    out = None
    L_seq = attentions[0].shape[-1]
    eye = torch.eye(L_seq, device=attentions[0].device)
    for att in attentions:
        a = att[0].mean(dim=0)               # [L,L]
        a = 0.5 * a + 0.5 * eye              # residual
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        out = a if out is None else a @ out
    last_text = img_end + 1                  # first non-image position after img block
    if last_text >= out.shape[0]:
        last_text = out.shape[0] - 1
    return out[last_text, img_start : img_end + 1].float()


def normalize(t: torch.Tensor) -> torch.Tensor:
    s = t.sum()
    return t / s if s > 0 else t


@torch.no_grad()
def extract_one(runner: PAPITLlavaRunner, image: Image.Image, question: str):
    text = runner._format_prompt(question)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(runner.device) for k, v in inputs.items()}

    patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
    text_emb = runner._get_text_embedding(question)

    image_token_id = runner.llava.config.image_token_id
    img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    img_start, img_end = int(img_pos[0]), int(img_pos[-1])

    # ---- 1. Single forward for prompt-side attention at L8/16/24 + rollout
    out = runner.llava(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs["attention_mask"],
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    att = out.attentions
    s_l8  = normalize(attn_to_images(att[8 ][0], img_start, img_end))
    s_l16 = normalize(attn_to_images(att[16][0], img_start, img_end))
    s_l24 = normalize(attn_to_images(att[24][0], img_start, img_end))
    s_roll = normalize(attention_rollout(att, img_start, img_end))

    return {
        "vit":         patch_for_proj.half().cpu(),
        "text":        text_emb.float().cpu(),
        "attn_L8":     s_l8.cpu(),
        "attn_L16":    s_l16.cpu(),
        "attn_L24":    s_l24.cpu(),
        "attn_rollout":s_roll.cpu(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=100_000)
    ap.add_argument("--llava-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--skip-first", type=int, default=0)
    args = ap.parse_args()

    df_full = pd.read_csv(args.csv)
    df = df_full.iloc[args.skip_first : args.skip_first + args.max_samples].reset_index(drop=True)
    print(f"Loaded {len(df)} samples from {args.csv} (skip={args.skip_first})")

    runner = PAPITLlavaRunner(
        llava_model_id=args.llava_id,
        config=PAPITConfig(retention_ratio=0.5),
        attn_implementation="eager",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    index_rows = []
    shard_buf: dict[str, list[torch.Tensor]] = {
        "vit": [], "text": [], "attn_L8": [], "attn_L16": [], "attn_L24": [],
        "attn_rollout": [],
    }
    shard_idx = 0
    n_done = 0
    t0 = time.time()

    def flush():
        nonlocal shard_idx, shard_buf
        if not shard_buf["vit"]:
            return
        torch.save(
            {k: torch.stack(v) for k, v in shard_buf.items()},
            args.out / f"shard_{shard_idx:04d}.pt",
        )
        shard_idx += 1
        shard_buf = {k: [] for k in shard_buf}

    for i, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[{i}] image load failed: {e}")
            continue
        try:
            d = extract_one(runner, img, row["question"])
        except Exception as e:
            print(f"[{i}] extract failed: {e}")
            continue
        for k in shard_buf:
            shard_buf[k].append(d[k])
        index_rows.append({
            "shard": shard_idx,
            "idx_in_shard": len(shard_buf["vit"]) - 1,
            "image_path": row["image_path"],
            "question":   row["question"],
            "source":     row.get("source", ""),
        })
        n_done += 1
        if len(shard_buf["vit"]) >= SHARD_SIZE:
            flush()
        if n_done % 100 == 0:
            sec = time.time() - t0
            eta = sec / n_done * (len(df) - n_done) / 60.0
            print(f"  [{n_done}/{len(df)}] {n_done/sec:.2f} samples/s   ETA {eta:.1f} min", flush=True)

    flush()
    pd.DataFrame(index_rows).to_csv(args.out / "index.csv", index=False)
    sec = time.time() - t0
    print(f"\nCached {n_done} samples to {args.out}  ({n_done/sec:.2f} samples/s, {sec/60:.1f} min)")


if __name__ == "__main__":
    main()
