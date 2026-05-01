# PAPIT Scoring Method Experiment Log

Tracks the evolution of patch scoring in `papit/integration/llava.py`.
All runs: LLaVA-1.5-7B, retention ratios k=25% / 50% / 75%.

---

## Method 1 — ViT Last-Layer Cosine + GradCAM Hybrid

**Change:** Use LLaVA's own ViT `last_hidden_state` (instead of separate CLIP vision encoder)
projected via `visual_projection` → cosine with CLIP text embedding.
GradCAM used when `retention_ratio ≤ 0.375`; cosine used for k=50% and k=75%.

**Runs:** `aws_results_0413` (700 samples), `aws_results_300_0415_1703` (300 samples)

| Dataset | Method       | k=25% | k=50% | k=75% |
|---------|--------------|-------|-------|-------|
| | Rel. FLOPs   | 0.10  | 0.29  | 0.59  |
| GQA     | Unpruned     | 72.0  | 72.0  | 72.0  |
|         | Random       | 65.9  | 69.1  | 70.1  |
|         | PAPIT        | 62.9  | 67.4  | 68.6  |
| VQA-v2  | Unpruned     | 76.5  | 76.5  | 76.5  |
|         | Random       | 71.9  | 74.6  | 75.8  |
|         | PAPIT        | 69.4  | 72.1  | 73.6  |
| TextVQA | Unpruned     | 49.0  | 49.0  | 49.0  |
|         | Random       | 32.3  | 39.8  | 42.0  |
|         | PAPIT        | 31.7  | 31.1  | 37.2  |

**Problem:** Cosine scoring at k=50%/75% collapses well below random, especially TextVQA (31.1 vs 39.8).

---

## Method 2 — GradCAM Only (threshold=1.0) ✅ Best

**Change:** `_GRADCAM_THRESHOLD = 1.0` → GradCAM used at all retention ratios.
Backprops through CLS→text cosine similarity; hooks `layers[-2]` activations and gradients.

**Runs:** `aws_results_300_0415_2207` (300), `aws_results_300_0416_0228` (300), `aws_results_700_0415_2355` (700 — primary)

| Dataset | Method       | k=25% | k=50% | k=75% |
|---------|--------------|-------|-------|-------|
| | Rel. FLOPs   | 0.10  | 0.29  | 0.59  |
| GQA     | Unpruned     | 72.0  | 72.0  | 72.0  |
|         | Random       | 65.9  | 69.1  | 70.1  |
|         | PAPIT        | 65.1  | 68.7  | 69.7  |
| VQA-v2  | Unpruned     | 76.5  | 76.5  | 76.5  |
|         | Random       | 71.9  | 74.6  | 75.8  |
|         | PAPIT        | 70.9  | 74.5  | 74.8  |
| TextVQA | Unpruned     | 49.0  | 49.0  | 49.0  |
|         | Random       | 32.3  | 39.8  | 42.0  |
|         | PAPIT        | 31.6  | 38.4  | 42.9  |

**Result:** No collapse. PAPIT ≈ random within ~1% across all datasets and retention ratios.
TextVQA k=75%: PAPIT slightly beats random (42.9 vs 42.0).

**Note:** GradCAM requires backward pass → at k=75%, latency slightly exceeds unpruned.

---

## Method 3 — ViT Second-to-Last Layer Value Features (threshold=0.0) ❌ Worst

**Change:** Hook `layers[-2].self_attn.v_proj` (V features before attention aggregation).
Inspired by MaskCLIP (Zhou et al., 2022). `_GRADCAM_THRESHOLD = 0.0` → cosine only, no GradCAM.

**Runs:** `aws_results_300_0416_0343` (300 samples)

| Dataset | Method       | k=25% | k=50% | k=75% |
|---------|--------------|-------|-------|-------|
| | Rel. FLOPs   | 0.10  | 0.29  | 0.59  |
| GQA     | Unpruned     | 71.3  | 71.3  | 71.3  |
|         | Random       | 64.3  | 68.7  | 69.3  |
|         | PAPIT        | 58.3  | 64.3  | 68.3  |
| VQA-v2  | Unpruned     | 76.6  | 76.6  | 76.6  |
|         | Random       | 71.0  | 72.8  | 73.3  |
|         | PAPIT        | 65.0  | 70.8  | 72.2  |
| TextVQA | Unpruned     | 49.6  | 49.6  | 49.6  |
|         | Random       | 33.8  | 38.4  | 43.1  |
|         | PAPIT        | 18.0  | 24.9  | 33.3  |

**Problem:** Worst of all methods. TextVQA catastrophically low (18.0 vs random 33.8 at k=25%).
LLaVA's ViT `v_proj` features are not aligned with CLIP text space — unlike MaskCLIP,
which uses CLIP's own ViT trained jointly with the text encoder.

---

## Summary (GQA)

| Method              | Samples | k=25% | k=50% | k=75% |
|---------------------|---------|-------|-------|-------|
| Random              | 700     | 65.9  | 69.1  | 70.1  |
| ViT cosine hybrid   | 700     | 62.9  | 67.4  | 68.6  |
| **GradCAM only**    | 700     | **65.1** | **68.7** | **69.7** |
| Value features      | 300     | 58.3  | 64.3  | 68.3  |

---

## Final Choice: GradCAM Only (`threshold=1.0`)

GradCAM at all retention ratios is the best method, consistently within ~1% of random.

**Why not value features:** LLaVA's ViT `v_proj` and CLIP's text encoder were never trained
together, so cosine similarity is unreliable as a scoring signal.

**Why GradCAM works best:** Gradient-weighted saliency via CLS→text backprop gives at least
partial cross-modal relevance signal, even without perfect feature alignment.

**Known limitation:** All training-free methods ≈ random pruning. Significant gains over random
require fine-tuning or LLM-internal attention (FastV-style). PAPIT's contribution is the
system design and the demonstration that token pruning in LLaVA incurs minimal accuracy loss.

---

## Run Index

| Folder | Method | Samples | Notes |
|--------|--------|---------|-------|
| `aws_results_0413` | ViT cosine hybrid (threshold=0.375) | 700 | First ViT-based run |
| `aws_results_300_0415_1703` | ViT cosine hybrid (threshold=0.375) | 300 | Small-scale confirmation |
| `aws_results_300_0415_2207` | GradCAM only (threshold=1.0) | 300 | First GradCAM-only run |
| `aws_results_300_0416_0228` | GradCAM only (threshold=1.0) | 300 | Re-run after git cleanup |
| `aws_results_700_0415_2355` | GradCAM only (threshold=1.0) | 700 | **Primary reference result** |
| `aws_results_300_0416_0343` | Value features (threshold=0.0) | 300 | v_proj second-to-last layer — worst |
