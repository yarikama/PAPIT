#!/usr/bin/env bash
# Harvest all Scope-C artefacts from AWS, print a summary, and report
# whether the full pipeline (cache → Phase 3 → Phase 4 → Phase 5 →
# Phase 6) finished cleanly.
#
# Usage (from repo root, locally):
#     bash scripts/harvest_scope_c.sh
set -euo pipefail
LOCAL_OUT=outputs

echo "[harvest] checking AWS pipeline state..."
ssh aws "tail -3 /tmp/scope_c_p34.log 2>/dev/null | head -3
ls /opt/dlami/nvme/predictors_100k/ 2>/dev/null
ls outputs/distill100k_*.csv 2>/dev/null | wc -l
ls outputs/distill700_*.csv 2>/dev/null | wc -l
ls outputs/efficiency_distill.csv 2>/dev/null && echo HAS_EFF || echo NO_EFF" || echo "ssh failed"

echo "[harvest] pulling artefacts..."
mkdir -p $LOCAL_OUT
scp 'aws:~/PAPIT/outputs/distill100k_*.csv' $LOCAL_OUT/ 2>/dev/null || echo "no Phase-4 CSVs yet"
scp 'aws:~/PAPIT/outputs/distill700_*.csv'  $LOCAL_OUT/ 2>/dev/null || echo "no Phase-5 CSVs yet"
scp 'aws:~/PAPIT/outputs/efficiency_distill.csv' $LOCAL_OUT/ 2>/dev/null || echo "no Phase-6 CSV yet"
scp aws:/opt/dlami/nvme/predictors_100k/mlp4_attn_L16.pt $LOCAL_OUT/predictor_100k_L16.pt 2>/dev/null || echo "no L16 predictor"
scp aws:/opt/dlami/nvme/predictors_100k/mlp4_attn_L8.pt $LOCAL_OUT/predictor_100k_L8.pt 2>/dev/null || true
scp aws:/opt/dlami/nvme/predictors_100k/mlp4_attn_L24.pt $LOCAL_OUT/predictor_100k_L24.pt 2>/dev/null || true
scp aws:/opt/dlami/nvme/predictors_100k/mlp4_attn_rollout.pt $LOCAL_OUT/predictor_100k_rollout.pt 2>/dev/null || true

echo "[harvest] local files:"
ls -la $LOCAL_OUT/distill100k_*.csv $LOCAL_OUT/distill700_*.csv $LOCAL_OUT/efficiency_distill.csv $LOCAL_OUT/predictor_100k_*.pt 2>/dev/null

echo "[harvest] summary:"
python3 - <<'PY'
import pandas as pd
from pathlib import Path
out = Path("outputs")
print("\n--- Phase 4 (100-sample × 3 datasets × 4 targets) ---")
for csv in sorted(out.glob("distill100k_*.csv")):
    try:
        df = pd.read_csv(csv)
        s = df.groupby("retention").mean(numeric_only=True)
        print(csv.name)
        print(s[[c for c in ["acc_unpruned","acc_random","acc_papit","acc_distill","acc_oracle"] if c in s.columns]].round(3))
    except Exception as e:
        print(csv.name, "err:", e)
print("\n--- Phase 5 (700-sample headline) ---")
for ds in ["gqa","textvqa","vqa_v2"]:
    p = out/f"distill700_{ds}.csv"
    if p.exists():
        df = pd.read_csv(p)
        s = df.groupby("retention").mean(numeric_only=True)[[c for c in ["acc_unpruned","acc_random","acc_papit","acc_distill"]]].round(3)
        print(ds); print(s)
print("\n--- Phase 6 (latency, ms) ---")
p = out/"efficiency_distill.csv"
if p.exists():
    df = pd.read_csv(p)
    print(df.groupby("method")["mean_ms"].agg(["mean","std"]).round(2))
PY
