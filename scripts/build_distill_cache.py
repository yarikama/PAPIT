#!/usr/bin/env python3
"""Cache (ViT hidden, text embedding, target attention) tuples for distillation.

For each (image, question) pair we run LLaVA-1.5-7B with output_attentions=True
and save:
  - vit_hidden  : [576, D_vit]  fp16   --- LLaVA vision_feature_layer output
  - text_emb    : [D_clip]      fp32   --- CLIP text embedding (PAPIT scoring)
  - target_attn : [576]         fp32   --- normalised L=target_layer attention
                                          summed over non-image queries,
                                          averaged over heads, on the prompt
                                          forward pass.

Files are sharded as torch .pt dicts under <out>/shard_XXXX.pt to avoid huge
single-file I/O. A summary index.csv tracks (shard, idx_in_shard, image,
question).

Usage:
    PYTHONPATH=. uv run python scripts/build_distill_cache.py \
        --csv /opt/dlami/nvme/data/gqa_train_subset.csv \
        --out /opt/dlami/nvme/distill_cache \
        --max-samples 5000 \
        --target-layer 16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from papit.core.config import PAPITConfig
from papit.integration.llava import PAPITLlavaRunner

SHARD_SIZE = 100  # samples per .pt shard


@torch.no_grad()
def extract_one(
    runner: PAPITLlavaRunner,
    image: Image.Image,
    question: str,
    target_layer: int,
):
    """Return (vit_hidden[576,D], text_emb[D_clip], target_attn[576])."""
    text = runner._format_prompt(question)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(runner.device) for k, v in inputs.items()}

    # ViT hidden states (the same patch_for_proj that PAPIT feeds into the MLP)
    patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])

    # CLIP text embedding (same one PAPIT uses for cosine scoring)
    text_emb = runner._get_text_embedding(question)

    # Layer-`target_layer` attention from non-image queries to image keys
    out = runner.llava(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        attention_mask=inputs["attention_mask"],
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    image_token_id = runner.llava.config.image_token_id
    img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    img_start, img_end = int(img_pos[0]), int(img_pos[-1])

    att = out.attentions[target_layer][0].mean(dim=0)  # [L, L]
    text_mask = torch.ones(att.shape[0], dtype=torch.bool, device=att.device)
    text_mask[img_start : img_end + 1] = False
    target = att[text_mask][:, img_start : img_end + 1].sum(dim=0).float()  # [576]
    target = target / (target.sum() + 1e-12)  # normalize to a probability

    return patch_for_proj.half().cpu(), text_emb.float().cpu(), target.cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=5000)
    ap.add_argument("--target-layer", type=int, default=16)
    ap.add_argument("--llava-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--skip-first", type=int, default=0,
                    help="Skip the first N rows (e.g. to reserve them for eval).")
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

    shard_idx = 0
    shard_buf = {"vit": [], "text": [], "target": []}
    t0 = time.time()
    n_done = 0

    def flush_shard():
        nonlocal shard_idx, shard_buf
        if not shard_buf["vit"]:
            return
        torch.save(
            {
                "vit":    torch.stack(shard_buf["vit"]),     # [n, 576, D] fp16
                "text":   torch.stack(shard_buf["text"]),    # [n, D_clip] fp32
                "target": torch.stack(shard_buf["target"]),  # [n, 576] fp32
            },
            args.out / f"shard_{shard_idx:04d}.pt",
        )
        shard_idx += 1
        shard_buf = {"vit": [], "text": [], "target": []}

    for i, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[{i}] image load failed: {e}")
            continue
        try:
            v, t, y = extract_one(runner, img, row["question"], args.target_layer)
        except Exception as e:
            print(f"[{i}] extract failed: {e}")
            continue
        shard_buf["vit"].append(v)
        shard_buf["text"].append(t)
        shard_buf["target"].append(y)
        index_rows.append({
            "shard": shard_idx,
            "idx_in_shard": len(shard_buf["vit"]) - 1,
            "image_path": row["image_path"],
            "question": row["question"],
        })
        n_done += 1
        if len(shard_buf["vit"]) >= SHARD_SIZE:
            flush_shard()
        if n_done % 50 == 0:
            sec = time.time() - t0
            eta = sec / n_done * (len(df) - n_done) / 60.0
            print(f"  [{n_done}/{len(df)}] {n_done/sec:.2f} samples/s   ETA {eta:.1f} min")

    flush_shard()

    pd.DataFrame(index_rows).to_csv(args.out / "index.csv", index=False)
    sec = time.time() - t0
    print(f"\nCached {n_done} samples → {args.out}  ({n_done/sec:.2f} samples/s, {sec/60:.1f} min)")


if __name__ == "__main__":
    main()
