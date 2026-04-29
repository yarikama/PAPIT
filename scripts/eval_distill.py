#!/usr/bin/env python3
"""Evaluate the distilled predictor as a PAPIT scorer (Scope-A).

Selectors compared, all run through the same PAPIT splice pipeline:
  - Unpruned                : full 576 image tokens
  - Random                  : k uniformly-sampled patches + anchor
  - PAPIT (CLIP GradCAM)    : the original paper's scorer
  - PAPIT-distill           : trained predictor over (ViT hidden, text emb)
  - LLM-attn L=16 oracle    : two-pass upper bound (same as fastv_oracle, L=16)

Usage:
    PYTHONPATH=. uv run python scripts/eval_distill.py \
        --csv /opt/dlami/nvme/data/gqa_val_subset.csv \
        --predictor /opt/dlami/nvme/distill_predictor.pt \
        --max-samples 100 \
        --out outputs/distill_eval_gqa.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from papit.core.config import PAPITConfig
from papit.integration.llava import PAPITLlavaRunner

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_distill import PatchScorePredictor  # noqa: E402


_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    s = _PUNCT_RE.sub("", s.strip().lower()).split()
    return s[-1] if s else ""


def gqa_em(pred: str, gold: str) -> float:
    return 1.0 if _norm(pred) == _norm(gold) else 0.0


@torch.no_grad()
def llm_attn_score(runner, pixel_values, input_ids, attention_mask, layer):
    out = runner.llava(
        input_ids=input_ids, pixel_values=pixel_values,
        attention_mask=attention_mask, output_attentions=True,
        return_dict=True, use_cache=False,
    )
    image_token_id = runner.llava.config.image_token_id
    img_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    s, e = int(img_pos[0]), int(img_pos[-1])
    att = out.attentions[layer][0].mean(dim=0)
    text_mask = torch.ones(att.shape[0], dtype=torch.bool, device=att.device)
    text_mask[s : e + 1] = False
    return att[text_mask][:, s : e + 1].sum(dim=0).float()


@torch.no_grad()
def generate_with_scores(runner, image, prompt, scores, k, seed=None,
                         max_new_tokens=16):
    text = runner._format_prompt(prompt)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {key: val.to(runner.device) for key, val in inputs.items()}
    patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
    N = patch_for_proj.shape[0]; k = min(max(k, 1), N)
    if scores is None:
        gen = torch.Generator(); gen.manual_seed(seed if seed is not None else 0)
        idx = torch.randperm(N, generator=gen).to(runner.device)[:k]
    else:
        _, idx = torch.topk(scores, k, largest=True, sorted=True)
    selected = patch_for_proj[idx]
    if runner.config.anchor_strategy == "global_mean":
        selected = torch.cat([selected, patch_for_proj.mean(0, keepdim=True)], dim=0)
    pruned = runner._project_through_mlp(selected)
    embeds, mask = runner._build_inputs_embeds(
        inputs["input_ids"], pruned, inputs.get("attention_mask"))
    out = runner.llava.generate(
        inputs_embeds=embeds, attention_mask=mask, pixel_values=None,
        max_new_tokens=max_new_tokens, do_sample=False)
    return runner.processor.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--predictor", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--retention", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    ap.add_argument("--oracle-layer", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("outputs/distill_eval.csv"))
    ap.add_argument("--llava-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).head(args.max_samples).reset_index(drop=True)
    print(f"Loaded {len(df)} eval samples from {args.csv}")

    runner = PAPITLlavaRunner(
        llava_model_id=args.llava_id,
        config=PAPITConfig(retention_ratio=0.5),
        attn_implementation="eager",
    )
    device = runner.device

    ckpt = torch.load(args.predictor, weights_only=True, map_location=device)
    predictor = PatchScorePredictor(
        patch_dim=ckpt["patch_dim"], text_dim=ckpt["text_dim"], hidden=ckpt["hidden"],
    ).to(device)
    predictor.load_state_dict(ckpt["model_state"])
    predictor.eval()
    print(f"Loaded distilled predictor: {sum(p.numel() for p in predictor.parameters())/1e6:.2f}M params")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[{i}] image load failed: {e}"); continue
        gold = str(row.get("answer", "")).strip()

        # Compute four score vectors once per sample.
        text = runner._format_prompt(row["question"])
        inputs = runner.processor(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
        text_emb = runner._get_text_embedding(row["question"])

        with torch.no_grad():
            s_distill = predictor(
                patch_for_proj.unsqueeze(0).float(),
                text_emb.unsqueeze(0).float(),
            )[0]
        s_clip = runner._gradcam_scores(inputs["pixel_values"], row["question"], 576)
        s_oracle = llm_attn_score(
            runner, inputs["pixel_values"], inputs["input_ids"],
            inputs["attention_mask"], layer=args.oracle_layer,
        )

        unpruned_ans = runner.generate_unpruned(img, row["question"], max_new_tokens=16)

        for ret in args.retention:
            k = max(1, int(round(576 * ret)))
            try:
                a_papit   = generate_with_scores(runner, img, row["question"], s_clip,    k)
                a_distill = generate_with_scores(runner, img, row["question"], s_distill, k)
                a_oracle  = generate_with_scores(runner, img, row["question"], s_oracle,  k)
                a_random  = generate_with_scores(runner, img, row["question"], None,      k, seed=i)
            except Exception as e:
                print(f"[{i} k={ret}] err: {e}"); continue

            rows.append({
                "idx": i, "question": row["question"], "gold": gold,
                "retention": ret, "k": k,
                "ans_unpruned": unpruned_ans, "ans_random": a_random,
                "ans_papit": a_papit, "ans_distill": a_distill, "ans_oracle": a_oracle,
                "acc_unpruned": gqa_em(unpruned_ans, gold),
                "acc_random":   gqa_em(a_random,    gold),
                "acc_papit":    gqa_em(a_papit,     gold),
                "acc_distill":  gqa_em(a_distill,   gold),
                "acc_oracle":   gqa_em(a_oracle,    gold),
            })

        if (i + 1) % 20 == 0:
            tmp = pd.DataFrame(rows)
            print(f"--- after {i+1} samples ---")
            for ret in args.retention:
                sub = tmp[tmp["retention"] == ret]
                if len(sub) == 0: continue
                print(f"  k={ret:.2f}: unp={sub['acc_unpruned'].mean():.3f}"
                      f" rand={sub['acc_random'].mean():.3f}"
                      f" papit={sub['acc_papit'].mean():.3f}"
                      f" distill={sub['acc_distill'].mean():.3f}"
                      f" oracle={sub['acc_oracle'].mean():.3f}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print("\n=== Distillation eval summary ===")
    print(f"  N samples per retention: {len(out_df) // max(len(args.retention),1)}")
    for ret in args.retention:
        sub = out_df[out_df["retention"] == ret]
        if len(sub) == 0: continue
        print(f"  retention = {ret:.2f}  (k={int(round(576*ret))})")
        print(f"    Unpruned    : {sub['acc_unpruned'].mean():.4f}")
        print(f"    Random      : {sub['acc_random'].mean():.4f}")
        print(f"    PAPIT (CLIP): {sub['acc_papit'].mean():.4f}")
        print(f"    PAPIT-distill: {sub['acc_distill'].mean():.4f}")
        print(f"    Oracle L=16 : {sub['acc_oracle'].mean():.4f}")
    print(f"\nWrote to {args.out}")


if __name__ == "__main__":
    main()
