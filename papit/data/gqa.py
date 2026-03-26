"""GQA balanced-validation loader.

Download
--------
1. Images (all split):
   https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
   → extract to <images_dir>/

2. Balanced val questions:
   https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
   → unzip, use val_balanced_questions.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_gqa(
    questions_json: str,
    images_dir: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load GQA balanced-val questions into a unified benchmark CSV.

    Parameters
    ----------
    questions_json:
        Path to ``val_balanced_questions.json``.
    images_dir:
        Directory containing ``<imageId>.jpg`` files.
    max_samples:
        Randomly subsample to this many rows (reproducible via *seed*).
    seed:
        Random seed used when subsampling.

    Returns
    -------
    DataFrame with columns:
        question_id, image_path, question, answer, answer_list
    """
    with open(questions_json, encoding="utf-8") as f:
        data: dict = json.load(f)

    img_root = Path(images_dir)
    rows = []
    for qid, item in data.items():
        answer = str(item["answer"])
        rows.append(
            {
                "question_id": str(qid),
                "image_path": str(img_root / f"{item['imageId']}.jpg"),
                "question": str(item["question"]),
                "answer": answer,
                # Single canonical answer stored as length-1 list;
                # vqa_soft_accuracy falls back to exact-match for len==1.
                "answer_list": json.dumps([answer]),
            }
        )

    df = pd.DataFrame(rows)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df
