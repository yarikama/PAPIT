#!/usr/bin/env bash
# Scope-C Phase 3 + Phase 4 continuation script.
# Run on the AWS instance once the 100K multi-target cache has finished.
#
# Phase 3: train winning architecture (mlp4) on each of the 4 supervision
#          targets (L=8, L=16, L=24, attention rollout) using the 100K cache.
# Phase 4: evaluate every predictor + Random + PAPIT-CLIP + Oracle L=16
#          on a held-out subset of each of the three benchmarks.
#
# All output goes to /tmp/scope_c_p34.log so you can tail -f it.
set -euo pipefail
exec > /tmp/scope_c_p34.log 2>&1

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME=/opt/dlami/nvme/hf_cache
# HF_TOKEN, if needed for fresh downloads, should be set in the
# environment that launches this script (e.g. ~/.bashrc on AWS).
cd "$HOME/PAPIT"

CACHE=/opt/dlami/nvme/distill_cache_100k
OUT=/opt/dlami/nvme/predictors_100k
mkdir -p "$OUT"

if [ ! -f "$CACHE/index.csv" ]; then
    echo "[$(date)] ERROR: cache not finished (no index.csv at $CACHE)"
    exit 1
fi
echo "[$(date)] Cache OK: $(wc -l < $CACHE/index.csv) samples"

# --------------------------------------------------------------------------
# Phase 3: train mlp4 on each target
# --------------------------------------------------------------------------
for tgt in attn_L8 attn_L16 attn_L24 attn_rollout; do
    if [ -f "$OUT/mlp4_${tgt}.pt" ]; then
        echo "[$(date)] Phase 3: skipping $tgt (predictor already exists)"
        continue
    fi
    echo "=================================================================="
    echo "[$(date)] Phase 3: training mlp4 on target=$tgt (20K subset, 8 epoch)"
    PYTHONPATH=. uv run python scripts/train_distill_arch.py \
        --cache "$CACHE" --target "$tgt" --archs mlp4 \
        --epochs 8 --batch 64 --subset 20000 \
        --save-best "$OUT/mlp4_${tgt}.pt"
done
echo "[$(date)] PHASE_3_DONE — all 4 predictors saved"
echo "[$(date)] Phase 3 DONE. Predictors at $OUT/"
ls -la "$OUT/"

# --------------------------------------------------------------------------
# Phase 4: eval each predictor on each dataset (100 samples per dataset
# for first pass; can scale to 700 once we know which target wins).
# --------------------------------------------------------------------------
mkdir -p outputs
for tgt in attn_L8 attn_L16 attn_L24 attn_rollout; do
  for ds in gqa textvqa vqa_v2; do
    case "$ds" in
      gqa)     CSV=/opt/dlami/nvme/data/gqa_val_subset.csv ;;
      textvqa) CSV=/opt/dlami/nvme/data/textvqa_val_subset.csv ;;
      vqa_v2)  CSV=/opt/dlami/nvme/data/vqa_v2_val_subset.csv ;;
    esac
    OUTCSV="outputs/distill100k_${tgt}_${ds}.csv"
    if [ -f "$OUTCSV" ]; then
        echo "[$(date)] Phase 4: skipping $tgt/$ds (CSV exists)"
        continue
    fi
    echo "=================================================================="
    echo "[$(date)] Phase 4: eval mlp4-$tgt on $ds (100 samples, skip-oracle)"
    PYTHONPATH=. uv run python scripts/eval_distill.py \
        --csv "$CSV" \
        --predictor "$OUT/mlp4_${tgt}.pt" \
        --max-samples 100 --skip-oracle \
        --out "$OUTCSV"
  done
done
echo "[$(date)] Phase 4 DONE. CSVs in outputs/distill100k_*.csv"

# --------------------------------------------------------------------------
# Quick summary table for Phase 4
# --------------------------------------------------------------------------
PYTHONPATH=. uv run python - <<'PY'
import pandas as pd
from pathlib import Path
rows = []
for csv in sorted(Path("outputs").glob("distill100k_*.csv")):
    parts = csv.stem.split("_")
    dataset = parts[-1] if parts[-1] != "v2" else parts[-2] + "_" + parts[-1]
    target = "_".join(parts[1:-1]) if dataset != "vqa_v2" else "_".join(parts[1:-2])
    df = pd.read_csv(csv)
    for ret, sub in df.groupby("retention"):
        row = {
            "target": target, "dataset": dataset, "retention": ret,
            "unpruned":   sub["acc_unpruned"].mean(),
            "random":     sub["acc_random"].mean(),
            "papit_clip": sub["acc_papit"].mean(),
            "distill":    sub["acc_distill"].mean(),
        }
        if "acc_oracle" in sub.columns:
            row["oracle_L16"] = sub["acc_oracle"].mean()
        rows.append(row)
print("\n========== Phase 4 summary ==========")
print(pd.DataFrame(rows).round(3).to_string(index=False))
PY

echo "[$(date)] Phase 3+4 done. Stopping here per user instruction —"
echo "[$(date)] Phase 5+6 will be launched separately after winner-target review."
echo "[$(date)] PHASE_3_4_COMPLETE"
