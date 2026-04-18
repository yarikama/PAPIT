# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PAPIT** (Prompt-Aware Pruning for Image Tokens) reduces computational cost in Multimodal LLMs (specifically LLaVA) by using CLIP's cross-modal embeddings to score image patches against a user's query and retain only the most relevant ones. Pruning occurs between the ViT output and MLP projector (true token-level), not post-hoc.

**Package name:** `papimg` (install name) / `papit` (import name and CLI command)

## Setup & Installation

```bash
# Install with uv (preferred)
uv sync
uv sync --extra ocr    # Include EasyOCR support
uv sync --extra llava  # Include accelerate for LLaVA

# Or pip
pip install -e .
pip install -e ".[ocr,llava]"
```

## Common Commands

```bash
# CLI: prune a single image
papit /path/to/image.jpg "What color is the car?" --retention 0.5 --anchor global_mean --device cuda

# Download datasets
bash scripts/download_data.sh           # all datasets
bash scripts/download_data.sh gqa       # GQA only
bash scripts/download_data.sh textvqa   # TextVQA only

# Prepare datasets (convert raw → CSV)
python scripts/prepare_datasets.py gqa \
    --questions data/raw/gqa/val_balanced_questions.json \
    --images data/raw/gqa/images \
    --output data/gqa_val.csv

# Run full evaluation benchmark
python scripts/run_eval.py                              # all datasets
python scripts/run_eval.py --dataset gqa                # GQA only
python scripts/run_eval.py --max-samples 100 --retention 0.25 0.5 0.75

# Launch notebooks
jupyter lab notebooks/demo.ipynb    # interactive walkthrough
jupyter lab notebooks/eval.ipynb    # full evaluation pipeline
```

There is no test suite, Makefile, or linter configured. Validation is done through the evaluation notebooks and scripts.

## Architecture

### Three-Layer Design

**1. Core Pruning (`papit/core/`)**
- `config.py`: `PAPITConfig` (retention_ratio, anchor_strategy, device, model_id) and `PAPITOutput` dataclass
- `pruner.py`: `PromptAwarePruner` — stateless CLIP-based scoring pipeline: extract text embedding → cosine similarity scores → top-k selection → spatial anchor appended → position remapping

**2. LLaVA Integration (`papit/integration/llava.py`)**
- `PAPITLlavaRunner`: hooks into LLaVA between ViT hidden states and MLP projector. Extracts `patch_for_scoring` (pre-projector ViT features) separately from `patch_for_projector`, scores with CLIP, selects top-k, projects with LLaVA's MLP, then splices pruned token embeddings into `inputs_embeds`.
- Also exposes `generate_unpruned()` and `generate_random()` baselines for comparison.

**3. Benchmarking (`papit/benchmark/`)**
- `runner.py`: fast batch eval using BLIP-VQA (PAPIT vs OCR-guided vs Random)
- `efficiency.py`: latency/memory profiling across retention ratios
- `llava_runner.py`: full accuracy eval with LLaVA; writes `detailed.csv` and `summary.csv` to `outputs/<dataset>_eval/`

### Supporting Modules
- `papit/data/`: Loaders for GQA, VQA v2, TextVQA → unified pandas DataFrames
- `papit/ocr/retention.py`: EasyOCR integration; `ocr_forced_indices()` maps text bounding boxes to patch grid; `merge_topk_with_forced()` ensures text patches are always retained
- `papit/risk/awareness.py`: Safety keyword detection (`SAFETY_KEYWORDS`, `INSTRUCTION_KEYWORDS`); `risk_aware_topk()` forces safety-critical patches and blocks adversarial content patches
- `papit/utils/metrics.py`: `vqa_soft_accuracy()` (multi-annotator), `token_f1()`, `patch_recall()` (fraction of OCR text patches retained)
- `papit/utils/visualization.py`: `build_pruned_image()` (blacks out non-retained patches), `draw_patch_rects()` for matplotlib overlays

### Key Design Decisions
- CLIP model default: `openai/clip-vit-base-patch32` (512D shared embedding space)
- Default anchor strategy: `"global_mean"` — appends mean of all patches as a single anchor token to preserve global context
- LLaVA runs in `float16` on GPU; CLIP scoring runs in `float32`; explicit dtype conversions in `llava.py`
- Position IDs are remapped to a dense sequence after pruning (no gaps)
- OCR forced indices are merged *after* top-k selection, not as a pre-filter

### Data Flow
```
Image + Prompt
  → ViT (LLaVA vision tower) → patch hidden states
  → CLIP text encoder → text embedding
  → cosine similarity scoring
  → top-k selection (+ OCR forced / risk-aware override)
  → spatial anchor token appended
  → LLaVA MLP projector → pruned visual tokens
  → LLaVA LLM with spliced inputs_embeds → answer
```

### Outputs
Benchmark results land in `outputs/{gqa,vqa_v2,textvqa}_eval/` as `detailed.csv` (per-sample) and `summary.csv` (aggregate statistics).
