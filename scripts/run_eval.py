#!/usr/bin/env python3
"""
run_eval.py — PAPIT LLaVA accuracy benchmark (Section 1-B + Section 2 of eval.ipynb)

Usage:
    python scripts/run_eval.py                         # run all datasets
    python scripts/run_eval.py --dataset gqa           # only GQA
    python scripts/run_eval.py --dataset vqa_v2 textvqa  # multiple datasets
    python scripts/run_eval.py --max-samples 200       # quick test run
    python scripts/run_eval.py --retention 0.25 0.5    # custom retention ratios
"""

import argparse
import logging
import sys
from pathlib import Path

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("eval.log"),
    ],
)
log = logging.getLogger(__name__)


# ── args ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="PAPIT LLaVA evaluation benchmark")
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["gqa", "vqa_v2", "textvqa"],
        default=["gqa", "vqa_v2", "textvqa"],
        help="Which dataset(s) to evaluate (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Root data directory (default: ../data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs",
        help="Output directory for results (default: ../outputs)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (default: None = use all). Use 200 for a quick test.",
    )
    parser.add_argument(
        "--retention",
        nargs="+",
        type=float,
        default=[0.25, 0.50, 0.75],
        help="Patch retention ratios to evaluate (default: 0.25 0.50 0.75)",
    )
    parser.add_argument(
        "--llava-model",
        default="llava-hf/llava-1.5-7b-hf",
        help="LLaVA model ID from HuggingFace",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14",
        help="CLIP model ID (must match LLaVA vision tower)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cuda | mps | cpu (default: auto-detect)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens for LLaVA generation (default: 32)",
    )
    return parser.parse_args()


# ── device detection ─────────────────────────────────────────────────────────
def detect_device():
    import torch
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    log.info(f"Device: {dev}  |  torch {torch.__version__}")
    if dev == "cuda":
        import torch
        props = torch.cuda.get_device_properties(0)
        log.info(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
    return dev


# ── Section 1-B: dataset → CSV (delegates to prepare_datasets.py) ────────────
_PREPARE = Path(__file__).parent / "prepare_datasets.py"

_CSV_CONFIGS = {
    "gqa": {
        "args": lambda d: [
            "gqa",
            "--questions", str(d / "raw/gqa/val_balanced_questions.json"),
            "--images",    str(d / "raw/gqa/images"),
            "--output",    str(d / "gqa_val.csv"),
        ],
        "out": lambda d: d / "gqa_val.csv",
    },
    "vqa_v2": {
        "args": lambda d: [
            "vqa_v2",
            "--questions",    str(d / "raw/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json"),
            "--annotations",  str(d / "raw/vqa_v2/v2_mscoco_val2014_annotations.json"),
            "--images",       str(d / "raw/vqa_v2/val2014"),
            "--output",       str(d / "vqa_v2_val.csv"),
        ],
        "out": lambda d: d / "vqa_v2_val.csv",
    },
    "textvqa": {
        "args": lambda d: [
            "textvqa",
            "--annotations", str(d / "raw/textvqa/TextVQA_0.5.1_val.json"),
            "--images",      str(d / "raw/textvqa/train_val_images"),
            "--output",      str(d / "textvqa_val.csv"),
        ],
        "out": lambda d: d / "textvqa_val.csv",
    },
}


def build_csv(dataset: str, data_dir: Path, max_samples: int | None) -> Path:
    import subprocess

    cfg = _CSV_CONFIGS[dataset]
    out_csv: Path = cfg["out"](data_dir)

    if out_csv.exists():
        log.info(f"{dataset} CSV already exists: {out_csv}")
        return out_csv

    cmd = [sys.executable, str(_PREPARE)] + cfg["args"](data_dir)
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]

    log.info(f"Building {dataset} CSV → {out_csv}")
    subprocess.run(cmd, check=True)
    return out_csv


# ── Section 2: LLaVA benchmark ───────────────────────────────────────────────
def run_benchmark(dataset: str, csv_path: Path, output_dir: Path, args, device: str):
    from papit.benchmark.llava_runner import run_llava_benchmark

    out = output_dir / f"{dataset}_eval"
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"=== Running benchmark: {dataset.upper()} ===")
    log.info(f"  CSV:       {csv_path}")
    log.info(f"  Output:    {out}")
    log.info(f"  Retention: {args.retention}")
    log.info(f"  Samples:   {args.max_samples or 'all'}")

    results = run_llava_benchmark(
        dataset_csv=str(csv_path),
        output_dir=str(out),
        retention_list=args.retention,
        llava_model_id=args.llava_model,
        clip_model_id=args.clip_model,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    # Print summary
    cols = ["vqa_acc_papit", "vqa_acc_random", "vqa_acc_unpruned"]
    if dataset == "textvqa":
        cols.append("papit_patch_recall")
    summary = results.groupby("retention_ratio")[cols].mean()
    log.info(f"\n{dataset.upper()} results:\n{summary.to_string()}\n")

    return results


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = args.device or detect_device()

    data_dir   = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Datasets to run: {args.dataset}")
    log.info(f"Data dir:        {data_dir}")
    log.info(f"Output dir:      {output_dir}")

    for dataset in args.dataset:
        try:
            csv_path = build_csv(dataset, data_dir, args.max_samples)
            run_benchmark(dataset, csv_path, output_dir, args, device)
        except Exception as e:
            log.error(f"Failed on {dataset}: {e}", exc_info=True)
            sys.exit(1)

    log.info("All evaluations complete.")


if __name__ == "__main__":
    main()
