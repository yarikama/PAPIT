# PAPIT: Prompt-Aware Pruning for Image Tokens in Efficient Multimodal Inference

> Chao Hsuan Ho (ch218) · Heng Jui Hsu (hh83) · Kerstin Sun (ks256) · Rice University · COMP 646

LLaVA-1.5-7B feeds 576 image patch tokens through every self-attention
layer, making inference cost quadratic in visual sequence length.
**PAPIT-Distill** is a 1.3M-parameter MLP that distils LLaVA's mid-layer
attention into a single-pass image-token pruner: scoring a 576-patch
image takes 0.43 ms and at *k*=25% retention recovers near-unpruned
TextVQA accuracy.

📄 Paper: [`report/final_report.pdf`](report/final_report.pdf)

---

## Headline numbers

LLaVA-1.5-7B downstream accuracy at 100 samples / cell
(`lmms-lab` test-dev / val splits, σ ≈ 0.05):

| Dataset | *k*  | Random | PAPIT-CLIP | **PAPIT-Distill** | Unpruned |
| ------- | ---- | ------ | ---------- | ----------------- | -------- |
| TextVQA | 25%  | 23.7   | 20.0       | **36.0** (+12.3)  | 36.7     |
| GQA     | 50%  | 52.0   | 52.0       | **56.0**          | 57.0     |
| VQA v2  | 75%  | 82.0   | 82.7       | **85.0**          | 83.3     |

Scoring latency on g5.2xlarge (A10G, retention 0.5):

| Method                 | ms / image |
| ---------------------- | ---------- |
| Random                 | 0.07       |
| **PAPIT-Distill**      | **0.43**   |
| PAPIT-CLIP (GradCAM)   | 65.23      |
| Two-pass Oracle (L=16) | 216.17     |

Distill is **~150× faster** than PAPIT-CLIP and **~500× faster** than the
oracle it learns from, while matching the oracle's downstream accuracy.

---

## What's actually here

The headline result above is the *third* of three findings — the first
two are why distillation was necessary at all:

1. **PAPIT-CLIP (training-free)** — GradCAM on LLaVA's own CLIP ViT.
   Inserts a top-*k* between the ViT and the MLP projector. *Result:*
   only matches random pruning, sometimes worse (VQAv2 *k*=25%, −9.4 pt).
2. **Alignment diagnostic** — Spearman ρ(PAPIT-CLIP, LLM attention)
   ≈ 0 on every benchmark; top-10% Jaccard *below* uniform-random.
   Rules out the entire family of training-free CLIP-aligned scorers.
3. **PAPIT-Distill** — 1.3M-param MLP₄ trained with KL against LLaVA's
   layer-8 attention. Single-pass at deploy. Selected over L∈{16, 24,
   rollout} by downstream-accuracy ablation: distillability beats
   oracle strength.

---

## Pipeline

```text
Image  →  LLaVA ViT (frozen)  →  Patch features  H ∈ ℝ^(576×1024)
Prompt →  CLIP text encoder    →  Text embedding  T ∈ ℝ^(768)
                                       ↓
                  scorer:  s ∈ ℝ^576  (PAPIT-CLIP or PAPIT-Distill)
                                       ↓
                          Top-k + 1 anchor (mean) token
                                       ↓
                      LLaVA MLP projector  →  LLaVA LLM  →  Answer
```

LLM weights are unmodified; pruning happens between ViT and projector.

---

## Setup

```bash
uv sync
uv sync --extra llava   # accelerate for LLaVA
uv sync --extra ocr     # EasyOCR (optional)
```

Or with pip: `pip install -e ".[llava,ocr]"`.

Apple Silicon (MPS) is auto-detected; otherwise CUDA, then CPU.

---

## Reproduce the paper numbers

The deployed predictor (5.1 MB) is checked in and runs on a single
g5.2xlarge A10G:

```bash
# 1) Single-image inference
papit path/to/image.jpg "What color is the car?" \
    --retention 0.5 --device cuda

# 2) Reproduce Table 1 (downstream accuracy, N=100)
python scripts/run_eval.py --max-samples 100 --retention 0.25 0.5 0.75

# 3) Reproduce hero figure (Fig 1, paper)
python scripts/make_hero_figure.py \
    --predictor outputs/mlp4_attn_L8_20k.pt \
    --out report/fig_hero.pdf
```

Pre-trained artifact:

- `outputs/mlp4_attn_L8_20k.pt` — MLP₄ predictor (target = layer-8 attn,
  20K mixed-domain training pairs, val_kl 0.094, top-10% Jaccard 0.60).

---

## Code map

```text
papit/
  core/        config + stateless pruning logic
  integration/ LLaVA hook (extracts pre-projector features, splices
               pruned tokens back into inputs_embeds)
  benchmark/   accuracy + efficiency runners (LLaVA + BLIP)
  data/        GQA / VQA v2 / TextVQA loaders → unified DataFrame
  ocr/         EasyOCR-backed text-region forced retention
  risk/        safety/jailbreak keyword overrides

scripts/
  train_distill.py         train MLP predictor on cached features
  build_distill_cache_*.py cache LLaVA forward features for training
  run_eval.py              full LLaVA accuracy benchmark
  make_hero_figure.py      paper Fig 1
  gen_qualitative_fig.py   paper Fig 3 (GradCAM diagnostic)
  make_pareto.py           paper Fig 2 (accuracy-FLOPs Pareto)

report/
  final_report.tex / .pdf  4-page paper
  references.bib
  fig_*.pdf                vector figures
```

---

## Datasets

| Dataset               | Use                | Why                                                                       |
| --------------------- | ------------------ | ------------------------------------------------------------------------- |
| GQA test-dev balanced | Accuracy + oracle  | Multi-step spatial / semantic reasoning                                   |
| VQA v2 val-balanced   | Accuracy           | Standard VQA baseline                                                     |
| TextVQA val           | Accuracy           | Stress test for query-aware pruning — text patches are small but critical |

All three loaded from `lmms-lab/{GQA,VQAv2,textvqa}` on Hugging Face.

---

## Acknowledgements

Built on top of LLaVA-1.5 ([Liu et al. 2024](https://arxiv.org/abs/2310.03744)),
CLIP ([Radford et al. 2021](https://arxiv.org/abs/2103.00020)), and
Grad-CAM ([Selvaraju et al. 2017](https://arxiv.org/abs/1610.02391)).
Code assistance: GitHub Copilot and Claude (Anthropic). All experimental
design and analysis are the authors'.
