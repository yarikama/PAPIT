"""VQA v2 validation loader.

Download
--------
1. Val images (COCO 2014):
   http://images.cocodataset.org/zips/val2014.zip
   → extract to <images_dir>/   (files named COCO_val2014_000000######.jpg)

2. Questions:
   https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
   → v2_OpenEnded_mscoco_val2014_questions.json

3. Annotations:
   https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
   → v2_mscoco_val2014_annotations.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


def load_vqa_v2(
    questions_json: str,
    annotations_json: str,
    images_dir: str,
    max_samples: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load VQA v2 validation split into a unified benchmark CSV.

    Parameters
    ----------
    questions_json:
        Path to ``v2_OpenEnded_mscoco_val2014_questions.json``.
    annotations_json:
        Path to ``v2_mscoco_val2014_annotations.json``.
    images_dir:
        Directory containing ``COCO_val2014_*.jpg`` files.
    max_samples:
        Randomly subsample to this many rows.
    seed:
        Random seed used when subsampling.

    Returns
    -------
    DataFrame with columns:
        question_id, image_path, question, answer, answer_list
    """
    with open(questions_json, encoding="utf-8") as f:
        q_data = json.load(f)
    with open(annotations_json, encoding="utf-8") as f:
        a_data = json.load(f)

    # Index questions by question_id for O(1) lookup.
    questions: dict[int, dict] = {q["question_id"]: q for q in q_data["questions"]}

    img_root = Path(images_dir)
    rows = []
    for ann in a_data["annotations"]:
        qid = int(ann["question_id"])
        q = questions[qid]
        image_id = int(q["image_id"])
        # All 10 human answers for soft scoring.
        answers = [str(a["answer"]) for a in ann["answers"]]
        canonical = Counter(answers).most_common(1)[0][0]

        rows.append(
            {
                "question_id": qid,
                "image_path": str(img_root / f"COCO_val2014_{image_id:012d}.jpg"),
                "question": str(q["question"]),
                "answer": canonical,
                "answer_list": json.dumps(answers),
            }
        )

    df = pd.DataFrame(rows)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df
