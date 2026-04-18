#!/usr/bin/env bash
# download_data.sh  — Download & extract VQA datasets for PAPIT evaluation
#
# Usage:
#   bash scripts/download_data.sh              # download all datasets
#   bash scripts/download_data.sh gqa          # only GQA
#   bash scripts/download_data.sh vqa_v2       # only VQA v2
#   bash scripts/download_data.sh textvqa      # only TextVQA
#
# Zips are deleted after extraction to save disk space.
# Approximate disk needed (after extraction, zips removed):
#   GQA:     ~20 GB   VQA v2: ~6 GB   TextVQA: ~7 GB

set -euo pipefail

DATASET="${1:-all}"
DATA_DIR="$(dirname "$0")/../data/raw"

# ── helpers ──────────────────────────────────────────────────────────────────
info()  { echo "[INFO]  $*"; }
die()   { echo "[ERROR] $*" >&2; exit 1; }

# ── GQA ──────────────────────────────────────────────────────────────────────
download_gqa() {
    local dir="$DATA_DIR/gqa"
    mkdir -p "$dir"
    info "=== GQA ==="

    # Images (~20 GB)
    if [ ! -d "$dir/images" ]; then
        info "Downloading GQA images..."
        wget -nc -q --show-progress \
            https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip \
            -O "$dir/images.zip"
        info "Extracting GQA images..."
        unzip -q "$dir/images.zip" -d "$dir"   # zip contains images/ internally
        rm "$dir/images.zip"
        info "GQA images done."
    else
        info "GQA images already exist, skipping."
    fi

    # Questions (~50 MB) — only keep val_balanced_questions.json
    if [ ! -f "$dir/val_balanced_questions.json" ]; then
        info "Downloading GQA questions..."
        wget -nc -q --show-progress \
            https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip \
            -O "$dir/questions1.2.zip"
        info "Extracting val_balanced_questions.json..."
        unzip -q "$dir/questions1.2.zip" val_balanced_questions.json -d "$dir"
        rm "$dir/questions1.2.zip"
        info "GQA questions done."
    else
        info "GQA questions already exist, skipping."
    fi
}

# ── VQA v2 ───────────────────────────────────────────────────────────────────
download_vqa_v2() {
    local dir="$DATA_DIR/vqa_v2"
    mkdir -p "$dir"
    info "=== VQA v2 ==="

    # Images (~6 GB)
    if [ ! -d "$dir/val2014" ]; then
        info "Downloading VQA v2 images..."
        wget -nc -q --show-progress \
            http://images.cocodataset.org/zips/val2014.zip \
            -O "$dir/val2014.zip"
        info "Extracting VQA v2 images..."
        unzip -q "$dir/val2014.zip" -d "$dir"   # zip contains val2014/ internally
        rm "$dir/val2014.zip"
        info "VQA v2 images done."
    else
        info "VQA v2 images already exist, skipping."
    fi

    # Questions
    if [ ! -f "$dir/v2_OpenEnded_mscoco_val2014_questions.json" ]; then
        info "Downloading VQA v2 questions..."
        wget -nc -q --show-progress \
            https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip \
            -O "$dir/v2_Questions_Val_mscoco.zip"
        unzip -q "$dir/v2_Questions_Val_mscoco.zip" -d "$dir"
        rm "$dir/v2_Questions_Val_mscoco.zip"
        info "VQA v2 questions done."
    else
        info "VQA v2 questions already exist, skipping."
    fi

    # Annotations
    if [ ! -f "$dir/v2_mscoco_val2014_annotations.json" ]; then
        info "Downloading VQA v2 annotations..."
        wget -nc -q --show-progress \
            https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip \
            -O "$dir/v2_Annotations_Val_mscoco.zip"
        unzip -q "$dir/v2_Annotations_Val_mscoco.zip" -d "$dir"
        rm "$dir/v2_Annotations_Val_mscoco.zip"
        info "VQA v2 annotations done."
    else
        info "VQA v2 annotations already exist, skipping."
    fi
}

# ── TextVQA ──────────────────────────────────────────────────────────────────
download_textvqa() {
    local dir="$DATA_DIR/textvqa"
    mkdir -p "$dir"
    info "=== TextVQA ==="

    # Images (~7 GB)
    if [ ! -d "$dir/train_val_images" ]; then
        info "Downloading TextVQA images..."
        wget -nc -q --show-progress \
            https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip \
            -O "$dir/train_val_images.zip"
        info "Extracting TextVQA images..."
        unzip -q "$dir/train_val_images.zip" -d "$dir"
        rm "$dir/train_val_images.zip"
        # zip extracts to train_images/ — rename to match expected path
        [ -d "$dir/train_images" ] && mv "$dir/train_images" "$dir/train_val_images"
        info "TextVQA images done."
    else
        info "TextVQA images already exist, skipping."
    fi

    # Annotations (~15 MB, direct JSON)
    if [ ! -f "$dir/TextVQA_0.5.1_val.json" ]; then
        info "Downloading TextVQA annotations..."
        wget -nc -q --show-progress \
            https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json \
            -O "$dir/TextVQA_0.5.1_val.json"
        info "TextVQA annotations done."
    else
        info "TextVQA annotations already exist, skipping."
    fi
}

# ── main ─────────────────────────────────────────────────────────────────────
case "$DATASET" in
    gqa)     download_gqa ;;
    vqa_v2)  download_vqa_v2 ;;
    textvqa) download_textvqa ;;
    all)
        download_gqa
        download_vqa_v2
        download_textvqa
        ;;
    *) die "Unknown dataset '$DATASET'. Choose: gqa | vqa_v2 | textvqa | all" ;;
esac

info "All done. Data in: $DATA_DIR"
