#!/usr/bin/env python3
"""
generate_figures.py — Generate all report figures and re-run the efficiency benchmark.

Purpose
-------
This script is the single entry-point for producing every artifact that
the progress report (report/progress_report.tex) cannot derive from code
alone.  Running it does three things:

  1. Re-runs the efficiency benchmark (papit.benchmark.efficiency) with a
     BLIP-VQA proxy model as the QA function.  A global warmup call is
     issued *before* the benchmark loop so the k=100% row is no longer
     measured cold-start (fixing the TBD in Table 1 of the report).
     → outputs/efficiency_benchmark.csv

  2. Generates fig_efficiency.pdf — a grouped bar chart of prune latency
     vs. E2E latency across retention ratios (force_ocr=0 rows only).
     → outputs/fig_efficiency.pdf

  3. Generates fig_qualitative.pdf — a two-panel figure showing:
       Left:  CLIP patch-saliency heatmap (green = high score)
       Right: top-50% patches selected by PAPIT (red boxes) side-by-side
              with the same number of randomly selected patches (blue boxes)
     This replicates what demo.ipynb Section 2/4 produces manually.
     → outputs/fig_qualitative.pdf

  4. Generates fig_pareto.pdf — accuracy vs relative FLOPs Pareto curves
     for all three datasets (GQA, VQA v2, TextVQA), drawn from the
     *existing* llava_benchmark_summary.csv files without re-running LLaVA.
     → outputs/fig_pareto.pdf

Usage
-----
    python scripts/generate_figures.py                   # all figures
    python scripts/generate_figures.py --skip-efficiency # skip CSV re-run
    python scripts/generate_figures.py \\
        --image data/raw/textvqa/train_val_images/train_images/a50551c2199738ce.jpg \\
        --prompt "What does the sign say?"
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate PAPIT report figures")
    p.add_argument(
        "--image",
        default=None,
        help="Image path for fig_qualitative (default: auto-pick from TextVQA)",
    )
    p.add_argument(
        "--prompt",
        default="What does the sign say?",
        help="Prompt for fig_qualitative (default: 'What does the sign say?')",
    )
    p.add_argument(
        "--skip-efficiency",
        action="store_true",
        help="Skip re-running the efficiency benchmark; use existing CSV",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device override: cuda | cpu (default: auto)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS,
        help="Directory to write all outputs (default: outputs/)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _device(override: str | None) -> str:
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 1. Efficiency benchmark (re-run with correct warmup)
# ---------------------------------------------------------------------------

def _make_blip_qa_fn(device: str):
    """Return a callable (PIL image, question) → str using BLIP-VQA-base."""
    from transformers import BlipForQuestionAnswering, BlipProcessor

    print("  Loading BLIP-VQA-base …")
    proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    model.eval()

    @torch.no_grad()
    def qa_fn(image: Image.Image, question: str) -> str:
        inputs = proc(image, question, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=10)
        return proc.decode(out[0], skip_special_tokens=True)

    # Global warmup — ensures the model is fully initialised before the
    # benchmark loop, so the k=100% row is not measured cold-start.
    print("  Warming up BLIP …")
    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    qa_fn(dummy, "what is this?")
    print("  BLIP ready.")
    return qa_fn


def run_efficiency(args, device: str) -> Path:
    from papit.benchmark.efficiency import run_efficiency_benchmark

    csv_path = args.output_dir / "efficiency_benchmark.csv"

    if args.skip_efficiency and csv_path.exists():
        print(f"[efficiency] Skipping re-run; using {csv_path}")
        return csv_path

    # Find a sample image to profile against
    sample_img = _find_sample_image()
    if sample_img is None:
        print("[efficiency] WARNING: no sample image found — skipping benchmark re-run")
        return csv_path

    print(f"[efficiency] Benchmark image: {sample_img}")
    qa_fn = _make_blip_qa_fn(device)

    # retention_grid: start small so QA model is warm when k=1.0 runs
    df = run_efficiency_benchmark(
        image_path=str(sample_img),
        question=args.prompt,
        retention_grid=[0.25, 0.5, 0.75, 1.0],
        runs_per_setting=3,
        device=device,
        qa_fn=qa_fn,
        output_path=str(csv_path),
    )
    print(f"[efficiency] Saved → {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# 2. fig_efficiency.pdf
# ---------------------------------------------------------------------------

def fig_efficiency(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        print(f"[fig_efficiency] {csv_path} not found — skipping")
        return

    df = pd.read_csv(csv_path)
    # Keep only the plain PAPIT rows (no OCR, no risk-aware)
    df = df[(df["force_ocr"] == 0) & (df["risk_aware"] == 0)].copy()
    df = df.sort_values("retention_ratio").reset_index(drop=True)

    labels = [f"{int(r * 100)}%" for r in df["retention_ratio"]]
    prune_ms  = df["avg_prune_latency_sec"] * 1000
    e2e_ms    = df["avg_end_to_end_sec"] * 1000

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width / 2, prune_ms, width, label="Prune latency", color="#5b9bd5")
    ax.bar(x + width / 2, e2e_ms,   width, label="E2E latency",   color="#ed7d31")

    ax.set_xlabel("Retention ratio $k$")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Efficiency: prune vs E2E latency (BLIP proxy)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_efficiency] Saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. fig_qualitative.pdf
# ---------------------------------------------------------------------------

def _find_sample_image() -> Path | None:
    candidates = sorted(
        (ROOT / "data/raw/textvqa/train_val_images/train_images").glob("*.jpg")
    )
    return candidates[0] if candidates else None


def fig_qualitative(image_path: str | None, prompt: str, out_path: Path, device: str) -> None:
    if image_path is None:
        found = _find_sample_image()
        if found is None:
            print("[fig_qualitative] No image found — skipping")
            return
        image_path = str(found)

    from papit.core.config import PAPITConfig
    from papit.core.pruner import PromptAwarePruner
    from papit.utils.visualization import draw_patch_rects

    print(f"[fig_qualitative] Image : {image_path}")
    print(f"[fig_qualitative] Prompt: {prompt}")

    cfg = PAPITConfig(retention_ratio=0.5, device=device)
    pruner = PromptAwarePruner(cfg)

    out = pruner.run(image_path, prompt)
    grid = pruner.grid_size
    scores_np = out.scores.cpu().float().numpy()

    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    # ── Heatmap (left panel) ──────────────────────────────────────────────
    heat = scores_np.reshape(grid, grid)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)

    # ── PAPIT selected patches (right panel, red boxes) ───────────────────
    papit_coords = out.coords  # list of (row, col)

    # ── Random selection with same k (blue boxes) ─────────────────────────
    k = len(papit_coords)
    all_indices = list(range(grid * grid))
    rng = random.Random(42)
    rand_idx = rng.sample(all_indices, k)
    rand_coords = [(i // grid, i % grid) for i in rand_idx]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 0: original image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Panel 1: saliency heatmap overlay
    axes[1].imshow(image)
    axes[1].imshow(
        np.repeat(np.repeat(heat[:, :, np.newaxis], 1, axis=2), 1, axis=2),
        extent=[0, W, H, 0],
        cmap="RdYlGn",
        alpha=0.55,
        vmin=0, vmax=1,
    )
    axes[1].set_title("CLIP saliency heatmap")
    axes[1].axis("off")

    # Panel 2: PAPIT (red) vs Random (blue) patch boxes
    axes[2].imshow(image)
    draw_patch_rects(axes[2], papit_coords,  (W, H), grid, edgecolor="red",  linewidth=1.2)
    draw_patch_rects(axes[2], rand_coords,   (W, H), grid, edgecolor="blue", linewidth=1.2)
    axes[2].set_title(f"PAPIT (red) vs Random (blue) — top {k}/{grid*grid} patches")
    axes[2].axis("off")

    # Legend
    handles = [
        mpatches.Patch(edgecolor="red",  facecolor="none", label="PAPIT"),
        mpatches.Patch(edgecolor="blue", facecolor="none", label="Random"),
    ]
    axes[2].legend(handles=handles, loc="upper right", fontsize=8)

    fig.suptitle(f'Prompt: "{prompt}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_qualitative] Saved → {out_path}")


# ---------------------------------------------------------------------------
# 4. fig_pareto.pdf
# ---------------------------------------------------------------------------

_DATASET_LABELS = {
    "gqa":     "GQA",
    "vqa_v2":  "VQA v2",
    "textvqa": "TextVQA",
}

_COLORS = {
    "papit":    "#e63946",
    "random":   "#457b9d",
    "unpruned": "#2d6a4f",
}


def fig_pareto(output_dir: Path, out_path: Path) -> None:
    datasets = ["gqa", "vqa_v2", "textvqa"]
    available = []
    for ds in datasets:
        p = output_dir / f"{ds}_eval" / "llava_benchmark_summary.csv"
        if p.exists():
            available.append((ds, p))

    if not available:
        print("[fig_pareto] No summary CSVs found — skipping")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4), sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, (ds, csv_path) in zip(axes, available):
        df = pd.read_csv(csv_path).sort_values("avg_relative_flops")

        # Unpruned: constant horizontal line (same value for all k)
        unpruned_acc = df["avg_vqa_acc_unpruned"].iloc[0]
        ax.axhline(
            unpruned_acc * 100,
            color=_COLORS["unpruned"],
            linestyle="--",
            linewidth=1.5,
            label=f"Unpruned ({unpruned_acc*100:.1f}%)",
        )

        # PAPIT Pareto
        ax.plot(
            df["avg_relative_flops"],
            df["avg_vqa_acc_papit"] * 100,
            "o-",
            color=_COLORS["papit"],
            linewidth=2,
            markersize=6,
            label="PAPIT",
        )

        # Random Pareto
        ax.plot(
            df["avg_relative_flops"],
            df["avg_vqa_acc_random"] * 100,
            "s--",
            color=_COLORS["random"],
            linewidth=2,
            markersize=6,
            label="Random",
        )

        # Annotate retention ratios
        for _, row in df.iterrows():
            k_pct = int(round(row["avg_token_keep_ratio"] * 100))
            ax.annotate(
                f"{k_pct}%",
                xy=(row["avg_relative_flops"], row["avg_vqa_acc_papit"] * 100),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=_COLORS["papit"],
            )

        ax.set_xlabel("Relative attention FLOPs")
        ax.set_ylabel("VQA accuracy (%)")
        ax.set_title(_DATASET_LABELS.get(ds, ds))
        ax.legend(fontsize=8)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)

    fig.suptitle("Accuracy–Efficiency Pareto (LLaVA-1.5-7B, 300 samples/dataset)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_pareto] Saved → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    dev = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {dev}")
    print(f"Output dir: {args.output_dir}")

    # 1. Efficiency CSV
    eff_csv = run_efficiency(args, dev)

    # 2. fig_efficiency.pdf
    fig_efficiency(eff_csv, args.output_dir / "fig_efficiency.pdf")

    # 3. fig_qualitative.pdf
    fig_qualitative(args.image, args.prompt, args.output_dir / "fig_qualitative.pdf", dev)

    # 4. fig_pareto.pdf
    fig_pareto(args.output_dir, args.output_dir / "fig_pareto.pdf")

    print("\nDone. Files written to:", args.output_dir)


if __name__ == "__main__":
    main()
