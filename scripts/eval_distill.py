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
from train_distill_arch import ARCHS  # noqa: E402


def build_predictor_from_ckpt(ckpt: dict, device: str):
    """Construct a predictor object from a checkpoint, supporting both
    Scope-A `train_distill` checkpoints (no `arch` key, MLP2 baseline)
    and Scope-C `train_distill_arch` checkpoints (with `arch` key).
    """
    arch = ckpt.get("arch", "mlp2")
    patch_dim = ckpt.get("patch_dim", 1024)
    text_dim  = ckpt.get("text_dim", 768)
    if arch == "mlp2" and "hidden" in ckpt:
        # Scope-A baseline shape with explicit hidden
        m = PatchScorePredictor(patch_dim=patch_dim, text_dim=text_dim,
                                hidden=ckpt["hidden"])
    elif arch in ARCHS:
        m = ARCHS[arch](patch_dim=patch_dim, text_dim=text_dim)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    m.load_state_dict(ckpt["model_state"])
    m.to(device).eval()
    return m, arch


_PUNCT_RE = re.compile(r"[^\w\s]")
_GOLD_LIST_RE = re.compile(r"'([^']*)'")


def _norm(s: str) -> str:
    """Normalize a string: lowercase, strip 'ASSISTANT:' prefix and
    surrounding artifacts, drop punctuation, collapse whitespace.
    Preserves multi-word answers ('dakota digital', 'red shirt', etc.)."""
    s = s.strip().lower()
    if "assistant:" in s:
        s = s.rsplit("assistant:", 1)[-1]
    s = _PUNCT_RE.sub("", s)
    return " ".join(s.split())


def _parse_gold(gold: str) -> list[str]:
    """Returns list of normalized answers. Handles both GQA's single-word
    answer and VQA-style numpy-array string `"['a' 'b' 'c' ...]"`."""
    g = gold.strip()
    if g.startswith("[") and "'" in g:
        items = _GOLD_LIST_RE.findall(g)
        return [_norm(x) for x in items if x.strip()]
    return [_norm(g)]


def gqa_em(pred: str, gold: str) -> float:
    """Backward-compatible: name kept for old code paths but now returns
    GQA exact match OR VQA soft accuracy depending on gold format."""
    p = _norm(pred)
    if not p:
        return 0.0
    answers = _parse_gold(gold)
    if len(answers) <= 1:
        return 1.0 if p == answers[0] else 0.0
    # VQA soft: min(count_matching / 3, 1)
    count = sum(1 for a in answers if a == p)
    return min(count / 3.0, 1.0)


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
    ap.add_argument("--skip-oracle", action="store_true",
                    help="Skip the two-pass oracle (saves ~30%% time on large evals).")
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
    predictor, arch = build_predictor_from_ckpt(ckpt, device)
    print(f"Loaded distilled predictor (arch={arch}): "
          f"{sum(p.numel() for p in predictor.parameters())/1e6:.2f}M params")

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
        if args.skip_oracle:
            s_oracle = None
        else:
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
                a_random  = generate_with_scores(runner, img, row["question"], None,      k, seed=i)
                a_oracle  = (None if args.skip_oracle
                             else generate_with_scores(runner, img, row["question"], s_oracle, k))
            except Exception as e:
                print(f"[{i} k={ret}] err: {e}"); continue

            rec = {
                "idx": i, "question": row["question"], "gold": gold,
                "retention": ret, "k": k,
                "ans_unpruned": unpruned_ans, "ans_random": a_random,
                "ans_papit": a_papit, "ans_distill": a_distill,
                "acc_unpruned": gqa_em(unpruned_ans, gold),
                "acc_random":   gqa_em(a_random,    gold),
                "acc_papit":    gqa_em(a_papit,     gold),
                "acc_distill":  gqa_em(a_distill,   gold),
            }
            if not args.skip_oracle:
                rec["ans_oracle"] = a_oracle
                rec["acc_oracle"] = gqa_em(a_oracle, gold)
            rows.append(rec)

        if (i + 1) % 20 == 0:
            tmp = pd.DataFrame(rows)
            print(f"--- after {i+1} samples ---")
            for ret in args.retention:
                sub = tmp[tmp["retention"] == ret]
                if len(sub) == 0: continue
                line = (f"  k={ret:.2f}: unp={sub['acc_unpruned'].mean():.3f}"
                        f" rand={sub['acc_random'].mean():.3f}"
                        f" papit={sub['acc_papit'].mean():.3f}"
                        f" distill={sub['acc_distill'].mean():.3f}")
                if "acc_oracle" in sub.columns:
                    line += f" oracle={sub['acc_oracle'].mean():.3f}"
                print(line)

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
        if "acc_oracle" in sub.columns:
            print(f"    Oracle L=16 : {sub['acc_oracle'].mean():.4f}")
    print(f"\nWrote to {args.out}")


if __name__ == "__main__":
    main()
