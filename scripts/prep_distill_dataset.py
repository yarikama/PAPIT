#!/usr/bin/env python3
"""Prepare a mixed (GQA + VQAv2 + TextVQA) train CSV for distillation cache.

Default: 50K GQA + 30K VQAv2 + 20K TextVQA = 100K samples.
Pulls from each dataset's HuggingFace mirror, materialises images to JPG, and
writes the unified CSV (image_path, question, source).

Usage:
    HF_HOME=/opt/dlami/nvme/hf_cache uv run python \
        scripts/prep_distill_dataset.py \
        --out /opt/dlami/nvme/data/distill_mix_100k.csv \
        --gqa 50000 --vqav2 30000 --textvqa 20000
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image


def _save_image(out_dir: str, name: str, img_bytes: bytes) -> str | None:
    path = os.path.join(out_dir, f"{name}.jpg")
    if os.path.exists(path):
        return path
    try:
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(path, "JPEG", quality=92)
    except Exception:
        return None
    return path


def _bytes_of(field) -> bytes | None:
    if field is None:
        return None
    if isinstance(field, dict):
        return field.get("bytes")
    return field


def collect_gqa(target_n: int, img_root: str) -> list[dict]:
    """Pull (image, question) pairs from GQA train_balanced parquets."""
    files = list_repo_files("lmms-lab/GQA", repo_type="dataset")
    inst_files  = sorted([f for f in files if "train_balanced_instructions" in f])
    image_files = sorted([f for f in files if "train_balanced_images" in f])
    if not inst_files:
        # train balanced may be packed under train_balanced_instructions only
        inst_files = sorted([f for f in files if f.startswith("train_balanced") and "instruction" in f.lower()])
    if not image_files:
        image_files = sorted([f for f in files if f.startswith("train_balanced") and "image" in f.lower()])
    print(f"[gqa] {len(inst_files)} instruction shards, {len(image_files)} image shards")

    out_img_dir = os.path.join(img_root, "gqa_train")
    os.makedirs(out_img_dir, exist_ok=True)

    # Lazy-load image shards: build a dict from id -> bytes by scanning shards
    # only as we need images. We pre-load all questions first then look up.
    inst_df = pd.concat(
        [pd.read_parquet(hf_hub_download("lmms-lab/GQA", f, repo_type="dataset"))
         for f in inst_files], ignore_index=True)
    print(f"[gqa] loaded {len(inst_df)} questions; sampling {target_n}")
    inst_df = inst_df.sample(n=min(target_n, len(inst_df)), random_state=0).reset_index(drop=True)

    needed_ids = set(inst_df["imageId"].astype(str).tolist())
    img_lut: dict[str, bytes] = {}
    for shard in image_files:
        if not needed_ids:
            break
        path = hf_hub_download("lmms-lab/GQA", shard, repo_type="dataset")
        sdf = pd.read_parquet(path)
        sdf["id"] = sdf["id"].astype(str)
        hit = sdf[sdf["id"].isin(needed_ids)]
        for _, r in hit.iterrows():
            img_lut[r["id"]] = _bytes_of(r["image"])
        needed_ids -= set(hit["id"].tolist())
        print(f"[gqa] shard {shard}: +{len(hit)} images, {len(needed_ids)} still needed")

    rows: list[dict] = []
    for _, r in inst_df.iterrows():
        iid = str(r["imageId"])
        b = img_lut.get(iid)
        if b is None:
            continue
        path = _save_image(out_img_dir, iid, b)
        if path is None:
            continue
        rows.append({"image_path": path, "question": r["question"], "source": "gqa"})
    print(f"[gqa] kept {len(rows)} samples")
    return rows


def collect_simple(repo: str, split_glob: str, target_n: int, img_root: str,
                   subdir: str, label: str, q_field: str = "question") -> list[dict]:
    """Generic puller for VQAv2 / TextVQA where image is embedded in the same parquet."""
    files = list_repo_files(repo, repo_type="dataset")
    matching = sorted([f for f in files if split_glob in f])
    print(f"[{label}] {len(matching)} shards matching '{split_glob}'")
    out_img_dir = os.path.join(img_root, subdir)
    os.makedirs(out_img_dir, exist_ok=True)

    rows: list[dict] = []
    for shard in matching:
        if len(rows) >= target_n:
            break
        path = hf_hub_download(repo, shard, repo_type="dataset")
        sdf = pd.read_parquet(path)
        for i, r in sdf.iterrows():
            if len(rows) >= target_n:
                break
            b = _bytes_of(r.get("image"))
            if b is None:
                continue
            name = f"{label}_{shard.split('/')[-1].replace('.parquet','')}_{i}"
            ipath = _save_image(out_img_dir, name, b)
            if ipath is None:
                continue
            rows.append({"image_path": ipath, "question": r.get(q_field, ""), "source": label})
        print(f"[{label}] {shard}: total {len(rows)}/{target_n}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--img-root", type=str, default="/opt/dlami/nvme/data/raw_distill")
    ap.add_argument("--gqa", type=int, default=50_000)
    ap.add_argument("--vqav2", type=int, default=30_000)
    ap.add_argument("--textvqa", type=int, default=20_000)
    args = ap.parse_args()

    rows: list[dict] = []
    if args.gqa > 0:
        rows.extend(collect_gqa(args.gqa, args.img_root))
    if args.vqav2 > 0:
        rows.extend(collect_simple(
            "lmms-lab/VQAv2", "data/train-",
            args.vqav2, args.img_root, "vqav2_train", "vqav2"))
    if args.textvqa > 0:
        rows.extend(collect_simple(
            "lmms-lab/textvqa", "data/train-",
            args.textvqa, args.img_root, "textvqa_train", "textvqa"))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle mix
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")
    print(df["source"].value_counts())


if __name__ == "__main__":
    main()
