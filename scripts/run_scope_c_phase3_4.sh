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
    echo "=================================================================="
    echo "[$(date)] Phase 3: training mlp4 on target=$tgt"
    PYTHONPATH=. uv run python scripts/train_distill_arch.py \
        --cache "$CACHE" --target "$tgt" --archs mlp4 \
        --epochs 12 --batch 64 \
        --save-best "$OUT/mlp4_${tgt}.pt"
done
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
    echo "=================================================================="
    echo "[$(date)] Phase 4: eval mlp4-$tgt on $ds (100 samples)"
    PYTHONPATH=. uv run python scripts/eval_distill.py \
        --csv "$CSV" \
        --predictor "$OUT/mlp4_${tgt}.pt" \
        --max-samples 100 \
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
        rows.append({
            "target": target, "dataset": dataset, "retention": ret,
            "unpruned":   sub["acc_unpruned"].mean(),
            "random":     sub["acc_random"].mean(),
            "papit_clip": sub["acc_papit"].mean(),
            "distill":    sub["acc_distill"].mean(),
            "oracle_L16": sub["acc_oracle"].mean(),
        })
print("\n========== Phase 4 summary ==========")
print(pd.DataFrame(rows).round(3).to_string(index=False))
PY

# --------------------------------------------------------------------------
# Phase 5: prep 700-sample CSVs and run headline eval with the L=16
#          predictor against {Unpruned, Random, PAPIT-CLIP, PAPIT-distill}.
#          Oracle is skipped at 700-sample (already covered at 100-sample).
# --------------------------------------------------------------------------
echo "=================================================================="
echo "[$(date)] Phase 5: preparing 700-sample eval CSVs"
PYTHONPATH=. uv run python scripts/prep_700_eval.py

WINNER_PRED=$OUT/mlp4_attn_L16.pt
if [ ! -f "$WINNER_PRED" ]; then
    echo "[$(date)] ERROR: winner predictor $WINNER_PRED missing; aborting Phase 5"
    exit 1
fi

for ds in gqa textvqa vqa_v2; do
    CSV=/opt/dlami/nvme/data/${ds}_700.csv
    OUTCSV="outputs/distill700_${ds}.csv"
    echo "=================================================================="
    echo "[$(date)] Phase 5: 700-sample eval on $ds with L=16 predictor"
    PYTHONPATH=. uv run python scripts/eval_distill.py \
        --csv "$CSV" --predictor "$WINNER_PRED" \
        --max-samples 700 --skip-oracle --out "$OUTCSV"
done

# --------------------------------------------------------------------------
# Phase 6: efficiency benchmark — wall-clock latency for the four scorers.
# --------------------------------------------------------------------------
echo "=================================================================="
echo "[$(date)] Phase 6: efficiency benchmark on 20 GQA samples"
PYTHONPATH=. uv run python scripts/run_efficiency_distill.py \
    --csv /opt/dlami/nvme/data/gqa_700.csv \
    --predictor "$WINNER_PRED" \
    --n 20 \
    --out outputs/efficiency_distill.csv

# --------------------------------------------------------------------------
# Final headline summary
# --------------------------------------------------------------------------
PYTHONPATH=. uv run python - <<'PY'
import pandas as pd
from pathlib import Path
print("\n========== Phase 5 headline (700-sample) ==========")
all_rows = []
for ds in ["gqa", "textvqa", "vqa_v2"]:
    p = Path(f"outputs/distill700_{ds}.csv")
    if not p.exists(): continue
    df = pd.read_csv(p)
    for ret, sub in df.groupby("retention"):
        all_rows.append({
            "dataset": ds, "k": ret,
            "unpruned": round(sub["acc_unpruned"].mean(), 3),
            "random":   round(sub["acc_random"].mean(),   3),
            "papit_clip": round(sub["acc_papit"].mean(),  3),
            "papit_distill": round(sub["acc_distill"].mean(), 3),
        })
print(pd.DataFrame(all_rows).to_string(index=False))

eff = Path("outputs/efficiency_distill.csv")
if eff.exists():
    print("\n========== Phase 6 latency (ms, mean over samples) ==========")
    print(pd.read_csv(eff).groupby("method")["mean_ms"].agg(["mean","std"]).round(2))
PY

echo "[$(date)] All Scope-C phases done."
