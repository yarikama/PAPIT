#!/usr/bin/env python3
"""Run the PAPIT efficiency benchmark and save results to --output-dir.

Usage:
    python scripts/run_efficiency_benchmark.py --output-dir outputs/aws_results_XXX/hybrid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _find_sample_image() -> Path | None:
    for pattern in [
        "data/raw/gqa/images/*.jpg",
        "data/raw/textvqa/train_val_images/train_images/*.jpg",
        "data/raw/vqa_v2/val2014/*.jpg",
    ]:
        candidates = sorted((ROOT / pattern.split("*")[0]).glob("*.jpg"))
        if candidates:
            return candidates[0]
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write efficiency_benchmark.csv")
    p.add_argument("--image", default=None, help="Image path to profile (default: auto-pick)")
    p.add_argument("--prompt", default="What is in the image?")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--runs", type=int, default=3)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_img = Path(args.image) if args.image else _find_sample_image()
    if sample_img is None:
        print("[efficiency] No sample image found — skipping")
        return
    print(f"[efficiency] Image  : {sample_img}")
    print(f"[efficiency] Device : {device}")

    from transformers import BlipForQuestionAnswering, BlipProcessor
    print("[efficiency] Loading BLIP-VQA-base …")
    proc  = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    model.eval()

    @torch.no_grad()
    def qa_fn(image: Image.Image, question: str) -> str:
        inputs = proc(image, question, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=10)
        return proc.decode(out[0], skip_special_tokens=True)

    # Global warmup before the benchmark loop so k=100% is not measured cold
    print("[efficiency] Warming up BLIP …")
    qa_fn(Image.new("RGB", (224, 224), color=(128, 128, 128)), "what is this?")
    print("[efficiency] BLIP ready.")

    from papit.benchmark.efficiency import run_efficiency_benchmark
    csv_path = args.output_dir / "efficiency_benchmark.csv"

    # Start from low retention so model stays warm when k=1.0 runs
    run_efficiency_benchmark(
        image_path=str(sample_img),
        question=args.prompt,
        retention_grid=[0.25, 0.5, 0.75, 1.0],
        runs_per_setting=args.runs,
        device=device,
        qa_fn=qa_fn,
        output_path=str(csv_path),
    )
    print(f"[efficiency] Saved → {csv_path}")


if __name__ == "__main__":
    main()
