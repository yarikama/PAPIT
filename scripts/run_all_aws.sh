#!/usr/bin/env bash
# run_all_aws.sh — Launch all PAPIT experiments on AWS via tmux sessions.
#
# Usage:
#   bash run_all_aws.sh              # default: 700 samples
#   bash run_all_aws.sh --samples 300
#   bash run_all_aws.sh --samples 500 --skip-ocr --skip-anchor
#
# Each experiment runs in its own tmux session.
# SSH disconnection will NOT interrupt any job.
# Monitor: tmux ls  /  tmux attach -t <session>  (Ctrl+B, D to detach)

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
SAMPLES=700
SKIP_OCR=false
SKIP_ANCHOR=false
RETENTION="0.25 0.5 0.75"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --samples)     SAMPLES="$2";      shift 2 ;;
        --skip-ocr)    SKIP_OCR=true;     shift ;;
        --skip-anchor) SKIP_ANCHOR=true;  shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$REPO_DIR"
mkdir -p logs outputs

# Activate the uv venv
VENV_ACTIVATE="source $REPO_DIR/.venv/bin/activate"
PYTHON_ENV="$VENV_ACTIVATE && export PYTHONPATH=$REPO_DIR"

info() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── helper: launch a tmux session ─────────────────────────────────────────────
launch() {
    local name="$1"
    local cmd="$2"

    if tmux has-session -t "$name" 2>/dev/null; then
        info "Session '$name' already exists — skipping. Kill with: tmux kill-session -t $name"
        return
    fi

    tmux new-session -d -s "$name" \
        "cd $REPO_DIR && $PYTHON_ENV && $cmd; echo '[DONE] $name exit \$?'; sleep 30"
    info "Launched: $name"
}

# ── Step 4: Prepare CSVs ──────────────────────────────────────────────────────
info "=== Step 4: Preparing CSVs ==="

NEED_PREP=false

if [ ! -f "data/gqa_val.csv" ]; then
    launch "prep_gqa" \
        "python scripts/prepare_datasets.py gqa \
        --questions data/raw/gqa/val_balanced_questions.json \
        --images data/raw/gqa/images \
        --output data/gqa_val.csv 2>&1 | tee logs/prep_gqa.log"
    NEED_PREP=true
else
    info "data/gqa_val.csv already exists — skipping"
fi

if [ ! -f "data/vqa_v2_val.csv" ]; then
    launch "prep_vqa" \
        "python scripts/prepare_datasets.py vqa_v2 \
        --questions data/raw/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json \
        --annotations data/raw/vqa_v2/v2_mscoco_val2014_annotations.json \
        --images data/raw/vqa_v2/val2014 \
        --output data/vqa_v2_val.csv 2>&1 | tee logs/prep_vqa.log"
    NEED_PREP=true
else
    info "data/vqa_v2_val.csv already exists — skipping"
fi

if [ ! -f "data/textvqa_val.csv" ]; then
    launch "prep_textvqa" \
        "python scripts/prepare_datasets.py textvqa \
        --annotations data/raw/textvqa/TextVQA_0.5.1_val.json \
        --images data/raw/textvqa/train_val_images \
        --max-samples $SAMPLES \
        --output data/textvqa_val.csv 2>&1 | tee logs/prep_textvqa.log"
    NEED_PREP=true
else
    info "data/textvqa_val.csv already exists — skipping"
fi

if [ "$NEED_PREP" = true ]; then
    info "CSV prep is running. Wait for it to finish, then re-run this script."
    info "Monitor: tmux ls  /  tmux attach -t prep_textvqa"
    exit 0
fi

# ── Step 5: Main experiments ──────────────────────────────────────────────────
info "=== Step 5: Main experiments (${SAMPLES} samples) ==="
OUT_DIR="outputs/hybrid_${SAMPLES}"

launch "gqa" \
    "python scripts/run_eval.py --dataset gqa --max-samples $SAMPLES \
    --retention $RETENTION --output-dir $OUT_DIR 2>&1 | tee logs/gqa.log"

launch "vqa" \
    "python scripts/run_eval.py --dataset vqa_v2 --max-samples $SAMPLES \
    --retention $RETENTION --output-dir $OUT_DIR 2>&1 | tee logs/vqa.log"

launch "textvqa" \
    "python scripts/run_eval.py --dataset textvqa --max-samples $SAMPLES \
    --retention $RETENTION --output-dir $OUT_DIR 2>&1 | tee logs/textvqa.log"

# ── Step 6: OCR-forced ────────────────────────────────────────────────────────
if [ "$SKIP_OCR" = false ]; then
    info "=== Step 6: OCR-forced TextVQA ==="
    launch "ocr_forced" \
        "python scripts/run_eval.py --dataset textvqa --force-ocr --max-samples $SAMPLES \
        --retention $RETENTION --output-dir outputs/ocr_forced_${SAMPLES} 2>&1 | tee logs/ocr_forced.log"
fi

# ── Step 7: Anchor ablation ───────────────────────────────────────────────────
if [ "$SKIP_ANCHOR" = false ]; then
    info "=== Step 7: Anchor ablation ==="
    launch "anchor_dropped" \
        "python scripts/run_eval.py --dataset gqa --anchor dropped_mean --max-samples $SAMPLES \
        --retention $RETENTION --output-dir outputs/anchor_dropped_mean 2>&1 | tee logs/anchor_dropped.log"

    launch "anchor_none" \
        "python scripts/run_eval.py --dataset gqa --anchor none --max-samples $SAMPLES \
        --retention $RETENTION --output-dir outputs/anchor_none 2>&1 | tee logs/anchor_none.log"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
info "=== All sessions launched ==="
tmux ls
echo ""
echo "Monitor:   tmux attach -t <session>   (Ctrl+B, D to detach)"
echo "Kill all:  tmux kill-server"
echo "Results:   ls outputs/"
