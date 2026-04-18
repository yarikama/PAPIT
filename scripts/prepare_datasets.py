"""Convert raw GQA / VQA-v2 / TextVQA files to benchmark CSVs.

Usage
-----
# GQA
python scripts/prepare_datasets.py gqa \
    --questions  data/gqa/val_balanced_questions.json \
    --images     data/gqa/images \
    --output     data/gqa_val.csv \
    --max-samples 1000

# VQA v2
python scripts/prepare_datasets.py vqa_v2 \
    --questions    data/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations  data/vqa_v2/v2_mscoco_val2014_annotations.json \
    --images       data/vqa_v2/val2014 \
    --output       data/vqa_v2_val.csv \
    --max-samples  1000

# TextVQA
python scripts/prepare_datasets.py textvqa \
    --annotations  data/textvqa/TextVQA_0.5.1_val.json \
    --images       data/textvqa/train_val_images \
    --output       data/textvqa_val.csv \
    --max-samples  500

Download links
--------------
GQA
  Images:    https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
  Questions: https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip
             (use val_balanced_questions.json)

VQA v2
  Images:      http://images.cocodataset.org/zips/val2014.zip
  Questions:   https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
  Annotations: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

TextVQA
  Annotations: https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json
  Images:      https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _cmd_gqa(args: argparse.Namespace) -> None:
    from papit.data.gqa import load_gqa

    df = load_gqa(
        questions_json=args.questions,
        images_dir=args.images,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[GQA] {len(df)} samples → {args.output}")
    print(df.head(3).to_string(index=False))


def _cmd_vqa_v2(args: argparse.Namespace) -> None:
    from papit.data.vqa_v2 import load_vqa_v2

    df = load_vqa_v2(
        questions_json=args.questions,
        annotations_json=args.annotations,
        images_dir=args.images,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[VQA v2] {len(df)} samples → {args.output}")
    print(df.head(3).to_string(index=False))


def _cmd_textvqa(args: argparse.Namespace) -> None:
    from papit.data.textvqa import load_textvqa

    df = load_textvqa(
        annotations_json=args.annotations,
        images_dir=args.images,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[TextVQA] {len(df)} samples → {args.output}")
    print(df.head(3).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw dataset files to PAPIT benchmark CSVs."
    )
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="dataset", required=True)

    # GQA
    p_gqa = sub.add_parser("gqa")
    p_gqa.add_argument("--questions", required=True)
    p_gqa.add_argument("--images", required=True)
    p_gqa.add_argument("--output", required=True)
    p_gqa.add_argument("--max-samples", type=int, default=None, dest="max_samples")

    # VQA v2
    p_vqa = sub.add_parser("vqa_v2")
    p_vqa.add_argument("--questions", required=True)
    p_vqa.add_argument("--annotations", required=True)
    p_vqa.add_argument("--images", required=True)
    p_vqa.add_argument("--output", required=True)
    p_vqa.add_argument("--max-samples", type=int, default=None, dest="max_samples")

    # TextVQA
    p_tex = sub.add_parser("textvqa")
    p_tex.add_argument("--annotations", required=True)
    p_tex.add_argument("--images", required=True)
    p_tex.add_argument("--output", required=True)
    p_tex.add_argument("--max-samples", type=int, default=None, dest="max_samples")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    {"gqa": _cmd_gqa, "vqa_v2": _cmd_vqa_v2, "textvqa": _cmd_textvqa}[args.dataset](args)
