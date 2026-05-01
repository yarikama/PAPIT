#!/usr/bin/env python3
"""Pull a single TextVQA / GQA / VQAv2 image from the HF parquet shard
by question-substring match, and save it as a JPEG.

Used by the docs/demo.md recipe so the live demo doesn't depend on a
local raw-image cache.
"""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

from PIL import Image


def fetch(dataset: str, question_substr: str) -> tuple[Image.Image, str]:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    if dataset == "textvqa":
        pq = hf_hub_download("lmms-lab/textvqa",
            "data/validation-00000-of-00003.parquet", repo_type="dataset")
        df = pd.read_parquet(pq)
        m = df[df["question"].str.contains(question_substr, case=False, regex=False)]
        if len(m) == 0:
            raise SystemExit(f"no textvqa match for {question_substr!r}")
        field, q = m.iloc[0]["image"], m.iloc[0]["question"]
    elif dataset == "gqa":
        inst = hf_hub_download("lmms-lab/GQA",
            "testdev_balanced_instructions/testdev-00000-of-00001.parquet",
            repo_type="dataset")
        imgs = hf_hub_download("lmms-lab/GQA",
            "testdev_balanced_images/testdev-00000-of-00001.parquet",
            repo_type="dataset")
        dq = pd.read_parquet(inst)
        di = pd.read_parquet(imgs).drop_duplicates("id").set_index("id")
        m = dq[dq["question"].str.contains(question_substr, case=False, regex=False)]
        if len(m) == 0:
            raise SystemExit(f"no gqa match for {question_substr!r}")
        field, q = di.loc[str(m.iloc[0]["imageId"]), "image"], m.iloc[0]["question"]
    elif dataset == "vqa_v2":
        pq = hf_hub_download("lmms-lab/VQAv2",
            "data/validation-00000-of-00068.parquet", repo_type="dataset")
        df = pd.read_parquet(pq)
        m = df[df["question"].str.contains(question_substr, case=False, regex=False)]
        if len(m) == 0:
            raise SystemExit(f"no vqa_v2 match for {question_substr!r}")
        field, q = m.iloc[0]["image"], m.iloc[0]["question"]
    else:
        raise SystemExit(f"unknown dataset: {dataset}")
    img_bytes = field["bytes"] if isinstance(field, dict) else field
    return Image.open(io.BytesIO(img_bytes)).convert("RGB"), q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["textvqa", "gqa", "vqa_v2"], default="textvqa")
    ap.add_argument("--question", required=True,
                    help="substring to match in the question column")
    ap.add_argument("--out", type=Path, default=Path("/tmp/demo.jpg"))
    args = ap.parse_args()

    img, q = fetch(args.dataset, args.question)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"saved {args.out}  (matched: {q!r})")


if __name__ == "__main__":
    main()
