#!/usr/bin/env python3
"""Discussion experiment #1 — representation misalignment.

For each (image, question), compute two per-patch saliency vectors over
LLaVA's 24x24 patch grid:

  s_clip   : PAPIT GradCAM score (CLIP-text-aligned saliency on LLaVA's ViT)
  s_llava  : Average attention paid by LLaVA's LLM (last layer, all heads,
             over the first generated answer tokens) to each image token.

We then report per-image Spearman rank correlation between s_clip and
s_llava, and the distribution across the dataset. A low correlation
supports the Discussion claim that the CLIP saliency signal is a poor
predictor of what the LLM actually attends to.

Usage:
    python scripts/run_attention_alignment.py \
        --csv data/gqa_val.csv \
        --max-samples 100 \
        --out outputs/alignment_gqa.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

from papit.core.config import PAPITConfig
from papit.integration.llava import PAPITLlavaRunner


@torch.no_grad()
def llava_attention_to_patches(
    runner: PAPITLlavaRunner,
    image: Image.Image,
    prompt: str,
    answer_tokens: int = 8,
) -> np.ndarray:
    """Mean attention from the first `answer_tokens` generated tokens to each
    of the 576 image tokens, averaged over heads of the last LLM layer.

    Returns a length-N (=576) numpy array, normalized to sum to 1.
    """
    text = runner._format_prompt(prompt)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(runner.device) for k, v in inputs.items()}

    image_token_id = runner.llava.config.image_token_id
    img_pos = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=True)[0]
    if img_pos.numel() == 0:
        raise RuntimeError("No image tokens found in prompt")
    img_start, img_end = int(img_pos[0]), int(img_pos[-1])

    out = runner.llava.generate(
        **inputs,
        max_new_tokens=answer_tokens,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True,
    )

    # out.attentions: tuple over generated steps; each is a tuple over layers;
    # each layer = [batch, heads, q_len, k_len]. For step 0, q_len = full prompt;
    # for step t>0, q_len = 1.
    n_image_tokens = img_end - img_start + 1
    accum = torch.zeros(n_image_tokens, device=runner.device, dtype=torch.float32)
    n_used = 0
    for step_atts in out.attentions[:answer_tokens]:
        last = step_atts[-1]                  # [1, H, q, k]
        # Average over heads, take the LAST query position (the just-generated
        # token attending back), then slice the image range from keys.
        a = last.mean(dim=1)[0, -1]           # [k]
        accum += a[img_start : img_end + 1].float()
        n_used += 1

    accum = accum / max(n_used, 1)
    accum = accum / (accum.sum() + 1e-12)
    return accum.cpu().numpy()


@torch.no_grad()
def papit_gradcam_scores(
    runner: PAPITLlavaRunner,
    image: Image.Image,
    prompt: str,
) -> np.ndarray:
    """PAPIT GradCAM scores over the 576 patches. Returns length-N numpy."""
    text = runner._format_prompt(prompt)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(runner.device) for k, v in inputs.items()}
    pixel_values = inputs["pixel_values"]

    # _gradcam_scores requires gradient tracking internally
    n_patches = 576
    scores = runner._gradcam_scores(pixel_values, prompt, n_patches)
    return scores.detach().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Dataset CSV with image_path,question")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--answer-tokens", type=int, default=8,
                    help="How many generated tokens to average attention over")
    ap.add_argument("--out", type=Path, default=Path("outputs/alignment.csv"))
    ap.add_argument("--llava-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).head(args.max_samples).reset_index(drop=True)
    print(f"Loaded {len(df)} samples from {args.csv}")

    # eager attention is required for output_attentions=True from generate()
    runner = PAPITLlavaRunner(
        llava_model_id=args.llava_id,
        config=PAPITConfig(retention_ratio=0.5),
        attn_implementation="eager",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rhos = []
    for i, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            s_clip = papit_gradcam_scores(runner, img, row["question"])
            s_llava = llava_attention_to_patches(
                runner, img, row["question"], answer_tokens=args.answer_tokens
            )
            if s_clip.shape != s_llava.shape:
                print(f"[{i}] shape mismatch {s_clip.shape} vs {s_llava.shape}, skip")
                continue
            rho, _ = spearmanr(s_clip, s_llava)
            top10_clip = set(np.argsort(-s_clip)[:58])   # top 10% of 576
            top10_llava = set(np.argsort(-s_llava)[:58])
            jaccard_top10 = len(top10_clip & top10_llava) / len(top10_clip | top10_llava)
            rows.append({
                "idx": i,
                "image_path": row["image_path"],
                "question": row["question"],
                "spearman": rho,
                "jaccard_top10pct": jaccard_top10,
            })
            rhos.append(rho)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(df)}] running mean ρ = {np.nanmean(rhos):.3f}")
        except Exception as e:
            print(f"[{i}] error: {e}")
            continue

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    rhos_arr = np.array([r for r in rhos if not np.isnan(r)])
    js = out_df["jaccard_top10pct"].dropna().to_numpy()
    print("\n=== Alignment between CLIP-GradCAM and LLaVA LLM attention ===")
    print(f"  N samples       : {len(rhos_arr)}")
    print(f"  Spearman ρ mean : {rhos_arr.mean():.3f}")
    print(f"  Spearman ρ med  : {np.median(rhos_arr):.3f}")
    print(f"  Spearman ρ std  : {rhos_arr.std():.3f}")
    print(f"  Frac ρ > 0.3    : {(rhos_arr > 0.3).mean():.2%}")
    print(f"  Frac ρ < 0.0    : {(rhos_arr < 0.0).mean():.2%}")
    print(f"  Top-10% Jaccard : {js.mean():.3f} (random baseline ≈ 0.053)")
    print(f"\nWrote per-sample results to {args.out}")


if __name__ == "__main__":
    main()
