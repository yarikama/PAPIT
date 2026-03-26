"""TextVQA validation loader.

Download
--------
1. Annotations (val split):
   https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json

2. Images (train + val share the same folder):
   https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
   → extract to <images_dir>/   (files named <image_id>.jpg)

Notes
-----
TextVQA includes ``ocr_info`` with normalised bounding boxes (0-1 range).
These are stored as a JSON column (``ocr_boxes``) and used by
``papit.utils.metrics.patch_recall`` to compute the TextVQA-specific
patch-recall metric.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def load_textvqa(
    annotations_json: str,
    images_dir: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load TextVQA validation split into a unified benchmark CSV.

    Parameters
    ----------
    annotations_json:
        Path to ``TextVQA_0.5.1_val.json``.
    images_dir:
        Directory containing ``<image_id>.jpg`` files.
    max_samples:
        Randomly subsample to this many rows.
    seed:
        Random seed used when subsampling.

    Returns
    -------
    DataFrame with columns:
        question_id, image_path, question, answer, answer_list, ocr_boxes
    """
    with open(annotations_json, encoding="utf-8") as f:
        data: list[dict] = json.load(f)["data"]

    img_root = Path(images_dir)
    rows = []
    for item in data:
        answers = [str(a) for a in item.get("answers", [])]
        canonical = Counter(answers).most_common(1)[0][0] if answers else ""

        rows.append(
            {
                "question_id": item["question_id"],
                "image_path": str(img_root / f"{item['image_id']}.jpg"),
                "question": str(item["question"]),
                "answer": canonical,
                "answer_list": json.dumps(answers),
                # ocr_info entries: {"word": ..., "bounding_box": {top_left_x, top_left_y, width, height}}
                # Coordinates are normalised 0-1 relative to image dimensions.
                "ocr_boxes": json.dumps(item.get("ocr_info", [])),
            }
        )

    df = pd.DataFrame(rows)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df
