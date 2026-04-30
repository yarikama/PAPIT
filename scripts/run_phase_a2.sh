#!/usr/bin/env bash
# Phase A2 chain: re-cache 20K, retrain mlp4 @ L=8, render hero Fig 1,
# and stage all artefacts in ~/PAPIT/outputs for harvest. Idempotent
# (skips steps whose outputs already exist).
set -eo pipefail   # NOTE: no -u; ~/.bashrc references PS1 which is unset
exec > /tmp/phase_a2.log 2>&1

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/opt/dlami/nvme/hf_cache
export UV_CACHE_DIR=/opt/dlami/nvme/uv_cache
# Pull HF_TOKEN out of ~/.bashrc directly (one line, exact match) so we
# don't have to source the whole file.
HF_TOKEN_LINE=$(grep -E '^export HF_TOKEN=' "$HOME/.bashrc" 2>/dev/null | head -1 || true)
if [ -n "$HF_TOKEN_LINE" ]; then eval "$HF_TOKEN_LINE"; export HF_TOKEN; fi
cd "$HOME/PAPIT"

mkdir -p /opt/dlami/nvme/{hf_cache,uv_cache,data,distill_cache_20k,predictors_a2}

CSV=/opt/dlami/nvme/data/distill_mix_100k.csv
CACHE=/opt/dlami/nvme/distill_cache_20k
PRED=/opt/dlami/nvme/predictors_a2/mlp4_attn_L8.pt
HERO_OUT=$HOME/PAPIT/report/fig_hero.pdf

echo "[$(date)] === Phase 0: env check ==="
nvidia-smi --query-gpu=memory.used --format=csv | tail -1
df -h /opt/dlami/nvme | tail -1

echo "[$(date)] === Phase 1a: prep dataset CSV ==="
if [ -f "$CSV" ]; then
    echo "  CSV exists ($(wc -l < $CSV) rows), skip"
else
    PYTHONPATH=. uv run python scripts/prep_distill_dataset.py \
        --out "$CSV" --gqa 50000 --vqav2 30000 --textvqa 20000
fi

echo "[$(date)] === Phase 1b: build 20K multi-target cache ==="
if [ -f "$CACHE/index.csv" ]; then
    echo "  Cache exists ($(wc -l < $CACHE/index.csv) rows), skip"
else
    PYTHONPATH=. uv run python scripts/build_distill_cache_multi.py \
        --csv "$CSV" --out "$CACHE" --max-samples 20000
fi

echo "[$(date)] === Phase 2: train mlp4 on attn_L8 ==="
if [ -f "$PRED" ]; then
    echo "  Predictor exists, skip"
else
    PYTHONPATH=. uv run python scripts/train_distill_arch.py \
        --cache "$CACHE" --target attn_L8 --archs mlp4 \
        --epochs 8 --batch 64 --in-memory-threshold 25000 \
        --save-best "$PRED"
fi

echo "[$(date)] === Phase 3: render hero figure ==="
PYTHONPATH=. uv run python scripts/make_hero_figure.py \
    --predictor "$PRED" --out "$HERO_OUT"

echo "[$(date)] === Phase 4: stage harvest artefacts ==="
mkdir -p "$HOME/PAPIT/outputs/aws_artifacts"
cp "$PRED" "$HOME/PAPIT/outputs/aws_artifacts/mlp4_attn_L8_20k.pt"
cp "$CACHE/index.csv" "$HOME/PAPIT/outputs/aws_artifacts/distill_cache_20k_index.csv"
echo "Cache size: $(du -sh $CACHE | cut -f1)"  # for log
echo
echo "PHASE_A2_COMPLETE"
ls -la "$HOME/PAPIT/outputs/aws_artifacts/"
ls -la "$HERO_OUT"
