#!/usr/bin/env python3
"""Prepare 700-sample eval CSVs from each of {GQA, TextVQA, VQAv2} for
the Phase 5 headline evaluation. Reuses cached HF parquets so this is
cheap to re-run."""
from __future__ import annotations
import io, os
from pathlib import Path
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
OUT_DIR = Path("/opt/dlami/nvme/data")
N = 700


def save(out_dir, name, b):
    p = os.path.join(out_dir, f"{name}.jpg")
    if not os.path.exists(p):
        Image.open(io.BytesIO(b)).convert("RGB").save(p, "JPEG", quality=92)
    return p


def field(x):
    return x["bytes"] if isinstance(x, dict) else x


def gqa():
    inst = hf_hub_download("lmms-lab/GQA",
        "testdev_balanced_instructions/testdev-00000-of-00001.parquet",
        repo_type="dataset")
    imgs = hf_hub_download("lmms-lab/GQA",
        "testdev_balanced_images/testdev-00000-of-00001.parquet",
        repo_type="dataset")
    df_q = pd.read_parquet(inst)
    df_i = pd.read_parquet(imgs).drop_duplicates("id").set_index("id")
    img_dir = "/opt/dlami/nvme/data/raw/gqa/images"
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for _, r in df_q.head(N + 100).iterrows():  # buffer for missing imgs
        iid = str(r["imageId"])
        if iid not in df_i.index: continue
        b = field(df_i.loc[iid, "image"])
        if b is None: continue
        try: p = save(img_dir, iid, b)
        except Exception: continue
        rows.append({"image_path": p, "question": r["question"], "answer": r["answer"]})
        if len(rows) >= N: break
    pd.DataFrame(rows).to_csv(OUT_DIR / "gqa_700.csv", index=False)
    print(f"gqa: {len(rows)} → {OUT_DIR/'gqa_700.csv'}")


def vqav2():
    pq = hf_hub_download("lmms-lab/VQAv2",
        "data/validation-00000-of-00068.parquet", repo_type="dataset")
    df = pd.read_parquet(pq)
    img_dir = "/opt/dlami/nvme/data/raw/vqav2_val"
    os.makedirs(img_dir, exist_ok=True)
    rows = []
    for i, r in df.iterrows():
        b = field(r.get("image"))
        if b is None: continue
        try: p = save(img_dir, f"{i:06d}", b)
        except Exception: continue
        ans_field = r.get("multiple_choice_answer", r.get("answers", ""))
        rows.append({"image_path": p, "question": r["question"], "answer": str(ans_field)})
        if len(rows) >= N: break
    pd.DataFrame(rows).to_csv(OUT_DIR / "vqa_v2_700.csv", index=False)
    print(f"vqa_v2: {len(rows)} → {OUT_DIR/'vqa_v2_700.csv'}")


def textvqa():
    rows = []
    img_dir = "/opt/dlami/nvme/data/raw/textvqa_val"
    os.makedirs(img_dir, exist_ok=True)
    for shard in [0, 1]:  # 700 may need 2 shards
        if len(rows) >= N: break
        pq = hf_hub_download("lmms-lab/textvqa",
            f"data/validation-0000{shard}-of-00003.parquet", repo_type="dataset")
        df = pd.read_parquet(pq)
        for i, r in df.iterrows():
            b = field(r.get("image"))
            if b is None: continue
            try: p = save(img_dir, f"s{shard}_{i:06d}", b)
            except Exception: continue
            rows.append({"image_path": p, "question": r["question"], "answer": str(r.get("answers", ""))})
            if len(rows) >= N: break
    pd.DataFrame(rows).to_csv(OUT_DIR / "textvqa_700.csv", index=False)
    print(f"textvqa: {len(rows)} → {OUT_DIR/'textvqa_700.csv'}")


if __name__ == "__main__":
    gqa()
    vqav2()
    textvqa()
