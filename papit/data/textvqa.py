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
TextVQA_0.5.1_val.json does not include OCR bounding boxes.
This loader uses EasyOCR (if installed) to detect text regions at CSV-build
time and stores them in an ``ocr_boxes`` column for patch-recall evaluation.
If EasyOCR is not installed the column is omitted and patch recall is skipped.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# EasyOCR helper (optional dependency)
# ---------------------------------------------------------------------------

def _init_ocr_reader():
    """Return an EasyOCR Reader or None if the package is not installed."""
    try:
        import easyocr
        import torch
        gpu = torch.cuda.is_available()
        return easyocr.Reader(["en"], gpu=gpu, verbose=False)
    except Exception:
        return None


def _ocr_boxes_for_image(
    reader,
    image_path: str,
    img_w: int,
    img_h: int,
    min_conf: float = 0.3,
) -> list[dict]:
    """Run EasyOCR on one image; return boxes in patch_recall format.

    Coordinates are normalised 0-1 relative to image dimensions.
    """
    try:
        results = reader.readtext(image_path)
    except Exception:
        return []

    boxes = []
    for pts, _text, conf in results:
        if conf < min_conf:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = max(0.0, min(xs) / img_w)
        y0 = max(0.0, min(ys) / img_h)
        x1 = min(1.0, max(xs) / img_w)
        y1 = min(1.0, max(ys) / img_h)
        if x1 > x0 and y1 > y0:
            boxes.append({
                "bounding_box": {
                    "top_left_x": x0,
                    "top_left_y": y0,
                    "width":      x1 - x0,
                    "height":     y1 - y0,
                }
            })
    return boxes


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

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
        question_id, image_path, question, answer, answer_list
        ocr_boxes (only when EasyOCR is available)
    """
    with open(annotations_json, encoding="utf-8") as f:
        data: list[dict] = json.load(f)["data"]

    img_root = Path(images_dir)

    # Subsample first so EasyOCR only runs on the retained rows.
    import random as _rnd
    rng = _rnd.Random(seed)
    if max_samples is not None and max_samples < len(data):
        data = rng.sample(data, max_samples)

    # Initialise EasyOCR once (slow first call downloads models).
    reader = _init_ocr_reader()
    use_ocr = reader is not None
    if use_ocr:
        print("[TextVQA] EasyOCR initialised — extracting text bounding boxes.")
    else:
        print("[TextVQA] EasyOCR not available — ocr_boxes column will be omitted.")

    rows = []
    for item in data:
        answers = [str(a) for a in item.get("answers", [])]
        canonical = Counter(answers).most_common(1)[0][0] if answers else ""
        img_path = str(img_root / f"{item['image_id']}.jpg")

        row: dict = {
            "question_id": item["question_id"],
            "image_path":  img_path,
            "question":    str(item["question"]),
            "answer":      canonical,
            "answer_list": json.dumps(answers),
        }

        if use_ocr:
            img_w = int(item.get("image_width",  1))
            img_h = int(item.get("image_height", 1))
            row["ocr_boxes"] = json.dumps(
                _ocr_boxes_for_image(reader, img_path, img_w, img_h)
            )

        rows.append(row)

    return pd.DataFrame(rows)
