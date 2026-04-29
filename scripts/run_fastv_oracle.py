#!/usr/bin/env python3
"""FastV-style oracle benchmark — Tier-2 experiment.

For each (image, prompt) we compare four selectors at three retention
ratios on LLaVA-1.5-7B:

  1. Unpruned                         : full 576 image tokens.
  2. Random                           : k uniformly-sampled patches + anchor.
  3. PAPIT (CLIP GradCAM)             : the paper's scorer.
  4. FastV-oracle (LLM-internal attn) : two-pass, uses LLaVA's own layer-2
                                         attention as the patch score.

The oracle is "two-pass" rather than the single-pass FastV variant: we run
LLaVA once on the full 576-token sequence, read attention from layer 2
(averaged over heads, summed over text-side query positions) to the 576
image-token keys, and use that as the per-patch importance score. We
then re-run generation on the spliced (pruned) inputs_embeds. This is
strictly more expensive than FastV but gives the cleanest "could an
LLM-internal signal in principle beat random" answer.

Usage:
    PYTHONPATH=. uv run python scripts/run_fastv_oracle.py \
        --csv /opt/dlami/nvme/data/gqa_val_subset.csv \
        --max-samples 100 \
        --out outputs/fastv_oracle_gqa.csv
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


# ---------------------------------------------------------------------------
# VQA soft-accuracy / GQA exact-match scoring
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\s]")


def _norm(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub("", s)
    return s.split()[-1] if s.split() else ""


def gqa_exact_match(pred: str, gold: str) -> float:
    """1.0 if normalised pred ends in normalised gold (handles 'ASSISTANT: yes')."""
    p = _norm(pred)
    g = _norm(gold)
    return 1.0 if p == g else 0.0


# ---------------------------------------------------------------------------
# FastV-oracle scorer
# ---------------------------------------------------------------------------
@torch.no_grad()
def llm_attention_scores_multi(
    runner: PAPITLlavaRunner,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Run LLaVA once on the full image+prompt and return per-image-token
    attention scores at each requested layer (averaged over heads, summed
    over non-image query positions). Returns {layer_idx: [n_img] tensor}.
    """
    out = runner.llava(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    image_token_id = runner.llava.config.image_token_id
    img_pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    img_start, img_end = int(img_pos[0]), int(img_pos[-1])

    scores: dict[int, torch.Tensor] = {}
    for L in layers:
        att = out.attentions[L][0].mean(dim=0)   # [L, L]
        text_mask = torch.ones(att.shape[0], dtype=torch.bool, device=att.device)
        text_mask[img_start : img_end + 1] = False
        scores[L] = att[text_mask][:, img_start : img_end + 1].sum(dim=0).float()
    return scores


# ---------------------------------------------------------------------------
# Common pruning + generation given a score vector
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_scores(
    runner: PAPITLlavaRunner,
    image: Image.Image,
    prompt: str,
    scores: torch.Tensor | None,
    k: int,
    random_seed: int | None = None,
    max_new_tokens: int = 16,
) -> str:
    """Run LLaVA generation using the existing PAPIT pipeline, but with
    a caller-supplied score vector (or random selection if scores=None).
    """
    text = runner._format_prompt(prompt)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {key: val.to(runner.device) for key, val in inputs.items()}

    patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
    N = patch_for_proj.shape[0]
    k = min(max(k, 1), N)

    if scores is None:
        gen = torch.Generator()
        gen.manual_seed(random_seed if random_seed is not None else 0)
        indices = torch.randperm(N, generator=gen).to(runner.device)[:k]
    else:
        _, indices = torch.topk(scores, k=k, largest=True, sorted=True)

    selected = patch_for_proj[indices]
    if runner.config.anchor_strategy == "global_mean":
        anchor = patch_for_proj.mean(0, keepdim=True)
        selected = torch.cat([selected, anchor], dim=0)

    pruned_llm = runner._project_through_mlp(selected)
    inputs_embeds, attn_mask = runner._build_inputs_embeds(
        inputs["input_ids"], pruned_llm, inputs.get("attention_mask")
    )
    out = runner.llava.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        pixel_values=None,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return runner.processor.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--retention", nargs="+", type=float, default=[0.25, 0.50, 0.75])
    ap.add_argument("--out", type=Path, default=Path("outputs/fastv_oracle.csv"))
    ap.add_argument("--llava-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--layers", nargs="+", type=int, default=[2, 8, 16, 24, 31],
                    help="LLM layers for FastV-oracle attention (LLaVA-1.5-7B has 32 layers).")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).head(args.max_samples).reset_index(drop=True)
    print(f"Loaded {len(df)} samples from {args.csv}")

    # eager attention required for output_attentions=True
    runner = PAPITLlavaRunner(
        llava_model_id=args.llava_id,
        config=PAPITConfig(retention_ratio=0.5),
        attn_implementation="eager",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    n_total = len(df) * len(args.retention)
    n_done = 0
    for i, row in df.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"[{i}] image load failed: {e}")
            continue
        gold = str(row.get("answer", "")).strip()

        # ---- 1. Compute scores once per image: GradCAM (PAPIT) and LLM-attn
        text = runner._format_prompt(row["question"])
        inputs = runner.processor(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(runner.device) for k, v in inputs.items()}
        s_gradcam = runner._gradcam_scores(inputs["pixel_values"], row["question"], 576)
        s_llm_dict = llm_attention_scores_multi(
            runner, inputs["pixel_values"], inputs["input_ids"],
            inputs["attention_mask"], layers=args.layers,
        )

        # ---- 2. Unpruned baseline (once per sample)
        unpruned_ans = runner.generate_unpruned(img, row["question"], max_new_tokens=16)

        for ret in args.retention:
            k_keep = max(1, int(round(576 * ret)))
            try:
                ans_papit  = generate_with_scores(runner, img, row["question"], s_gradcam, k_keep)
                ans_random = generate_with_scores(runner, img, row["question"], None, k_keep, random_seed=i)
                ans_llm = {
                    L: generate_with_scores(runner, img, row["question"], s, k_keep)
                    for L, s in s_llm_dict.items()
                }
            except Exception as e:
                print(f"[{i} k={ret}] generate err: {e}")
                continue

            rec = {
                "idx": i,
                "question": row["question"],
                "gold": gold,
                "retention": ret,
                "k": k_keep,
                "ans_unpruned": unpruned_ans,
                "ans_papit":    ans_papit,
                "ans_random":   ans_random,
                "acc_unpruned": gqa_exact_match(unpruned_ans, gold),
                "acc_papit":    gqa_exact_match(ans_papit,    gold),
                "acc_random":   gqa_exact_match(ans_random,   gold),
            }
            for L, ans in ans_llm.items():
                rec[f"ans_llm_L{L}"] = ans
                rec[f"acc_llm_L{L}"] = gqa_exact_match(ans, gold)
            rows.append(rec)
            n_done += 1

        if (i + 1) % 20 == 0:
            tmp = pd.DataFrame(rows)
            print(f"--- after {i+1} samples ---")
            for ret in args.retention:
                sub = tmp[tmp["retention"] == ret]
                if len(sub) == 0:
                    continue
                line = (f"  k={ret:.2f}: unp={sub['acc_unpruned'].mean():.3f}"
                        f" rand={sub['acc_random'].mean():.3f}"
                        f" papit={sub['acc_papit'].mean():.3f}")
                for L in args.layers:
                    col = f"acc_llm_L{L}"
                    if col in sub.columns:
                        line += f" L{L}={sub[col].mean():.3f}"
                print(line)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print("\n=== FastV-oracle benchmark summary ===")
    print(f"  N samples per retention : {len(out_df) // max(len(args.retention),1)}")
    for ret in args.retention:
        sub = out_df[out_df["retention"] == ret]
        if len(sub) == 0:
            continue
        print(f"  retention = {ret:.2f}  (k={int(round(576*ret))})")
        print(f"    Unpruned       : {sub['acc_unpruned'].mean():.4f}")
        print(f"    Random         : {sub['acc_random'].mean():.4f}")
        print(f"    PAPIT (CLIP)   : {sub['acc_papit'].mean():.4f}")
        for L in args.layers:
            col = f"acc_llm_L{L}"
            if col in sub.columns:
                print(f"    LLM-attn  L={L:2d}: {sub[col].mean():.4f}")
    print(f"\nWrote per-sample results to {args.out}")


if __name__ == "__main__":
    main()
