"""Generate all report figures and save directly to report/.

Produces three PDFs:
  report/fig_qualitative.pdf  — GradCAM heatmap + PAPIT vs Random (2x2 per example)
  report/fig_efficiency.pdf   — Prune vs E2E latency bar chart (BLIP proxy CSV)
  report/fig_pareto.pdf       — Accuracy vs Rel. FLOPs Pareto (LLaVA 700-sample)

Usage (on AWS with GPU):
    python scripts/gen_qualitative_fig.py                  # all figures
    python scripts/gen_qualitative_fig.py --skip-qualitative  # skip LLaVA loading
    python scripts/gen_qualitative_fig.py --skip-efficiency
    python scripts/gen_qualitative_fig.py --skip-pareto
"""
import argparse
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT   = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
sys.path.insert(0, str(ROOT))

FONT = 11

# ── Primary results directory (700-sample GradCAM run) ────────────────────────
RESULTS_DIR = ROOT / "outputs" / "aws_results_700_0415_2355" / "hybrid"

# ── Efficiency CSV (BLIP proxy) ───────────────────────────────────────────────
EFFICIENCY_CSV = ROOT / "outputs" / "outputs" / "efficiency_benchmark.csv"

# ── Qualitative examples ──────────────────────────────────────────────────────
# Edit to change which images appear.
# TextVQA: data/raw/textvqa/train_val_images/<id>.jpg
# GQA:     data/raw/gqa/images/<id>.jpg
EXAMPLES = [
    (
        str(ROOT / "data/raw/gqa/images/2349976.jpg"),
        "What color is the jersey the boy is wearing?",
        "black",
    ),
    (
        str(ROOT / "data/raw/gqa/images/2371845.jpg"),
        "What vegetable is to the right of the tomato?",
        "lettuce",
    ),
]

RETENTION = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def scores_to_heatmap(scores: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    s = scores.cpu().float().numpy()
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    return s.reshape(grid_h, grid_w)


def make_pruned_image(image: Image.Image, kept_indices: list[int],
                      grid_size: int, alpha: float = 0.15) -> np.ndarray:
    arr = np.array(image.convert("RGB")).copy().astype(float)
    h, w = arr.shape[:2]
    cell_h, cell_w = h / grid_size, w / grid_size
    kept = set(kept_indices)
    for idx in range(grid_size * grid_size):
        if idx in kept:
            continue
        r, c = divmod(idx, grid_size)
        y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
        x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
        arr[y0:y1, x0:x1] *= alpha
    return arr.clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: fig_qualitative.pdf
# ─────────────────────────────────────────────────────────────────────────────

def gen_qualitative(device: str) -> None:
    from papit.core.config import PAPITConfig
    from papit.integration.llava import PAPITLlavaRunner

    out_path = REPORT / "fig_qualitative.pdf"
    print("\n[qualitative] Loading LLaVA …")
    config = PAPITConfig(retention_ratio=RETENTION, anchor_strategy="global_mean", device=device)
    runner = PAPITLlavaRunner(config=config, device=device)

    n = len(EXAMPLES)
    plt.rcParams.update({"font.size": FONT})
    fig, axes = plt.subplots(n * 2, 2, figsize=(7, 4.5 * n))

    for ex_idx, (img_path, prompt, answer) in enumerate(EXAMPLES):
        image = Image.open(img_path).convert("RGB")

        print(f"  [{ex_idx+1}/{n}] {prompt!r}")
        out       = runner.generate(image, prompt)
        info      = out.pruning_info
        N         = info.total_patches
        k         = info.selected_patches
        grid_size = int(math.isqrt(N))
        print(f"    Answer: {out.answer!r}  (expected: {answer!r})")

        llava_dtype  = next(runner.llava.parameters()).dtype
        pixel_values = runner.processor(
            images=image,
            text=runner._format_prompt(prompt),
            return_tensors="pt",
        )["pixel_values"].to(device)
        scores_gc = runner._gradcam_scores(pixel_values.to(llava_dtype), prompt, N)

        random.seed(42)
        rand_indices = random.sample(range(N), k)

        row0, row1 = ex_idx * 2, ex_idx * 2 + 1

        axes[row0, 0].imshow(image)
        axes[row0, 0].set_title("Original image", fontsize=FONT, fontweight="bold")
        axes[row0, 0].axis("off")

        hmap = scores_to_heatmap(scores_gc, grid_size, grid_size)
        axes[row0, 1].imshow(image)
        im = axes[row0, 1].imshow(
            hmap, cmap="jet", alpha=0.55,
            extent=[0, image.width, image.height, 0], aspect="auto",
        )
        axes[row0, 1].set_title("GradCAM saliency", fontsize=FONT, fontweight="bold")
        axes[row0, 1].axis("off")
        plt.colorbar(im, ax=axes[row0, 1], fraction=0.046, pad=0.02)

        axes[row1, 0].imshow(make_pruned_image(image, info.selected_indices, grid_size))
        axes[row1, 0].set_title(
            f"PAPIT  (k={RETENTION:.0%}, {k}/{N} patches)",
            fontsize=FONT, fontweight="bold",
        )
        axes[row1, 0].axis("off")

        axes[row1, 1].imshow(make_pruned_image(image, rand_indices, grid_size))
        axes[row1, 1].set_title(
            f"Random (k={RETENTION:.0%}, {k}/{N} patches)",
            fontsize=FONT, fontweight="bold",
        )
        axes[row1, 1].axis("off")

        fig.text(
            0.5, 1.0 - ex_idx / n - 0.01,
            f'Prompt: "{prompt}"  →  PAPIT: {out.answer!r}',
            ha="center", va="top", fontsize=FONT, style="italic", color="#333333",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: fig_efficiency.pdf
# ─────────────────────────────────────────────────────────────────────────────

def gen_efficiency() -> None:
    out_path = REPORT / "fig_efficiency.pdf"
    if not EFFICIENCY_CSV.exists():
        print(f"[efficiency] {EFFICIENCY_CSV} not found — skipping")
        return

    df = pd.read_csv(EFFICIENCY_CSV)
    df = df[(df["force_ocr"] == 0) & (df["risk_aware"] == 0)].copy()
    df = df.sort_values("retention_ratio").reset_index(drop=True)

    labels   = [f"{int(r*100)}%" for r in df["retention_ratio"]]
    prune_ms = df["avg_prune_latency_sec"] * 1000
    e2e_ms   = df["avg_end_to_end_sec"]   * 1000

    x, width = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(x - width/2, prune_ms, width, label="Prune latency", color="#5b9bd5")
    ax.bar(x + width/2, e2e_ms,   width, label="E2E latency",   color="#ed7d31")
    ax.set_xlabel("Retention ratio $k$", fontsize=FONT)
    ax.set_ylabel("Latency (ms)",        fontsize=FONT)
    ax.set_title("Efficiency: prune vs E2E latency\n(BLIP proxy, single A6000 GPU)", fontsize=FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONT)
    ax.legend(fontsize=FONT)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[efficiency] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: fig_pareto.pdf
# ─────────────────────────────────────────────────────────────────────────────

_DATASETS = {"gqa": "GQA", "vqa_v2": "VQA v2", "textvqa": "TextVQA"}
_COLORS   = {"papit": "#e63946", "random": "#457b9d", "unpruned": "#2d6a4f"}


def gen_pareto() -> None:
    out_path  = REPORT / "fig_pareto.pdf"
    available = []
    for ds, label in _DATASETS.items():
        csv = RESULTS_DIR / f"{ds}_eval" / "llava_benchmark_summary.csv"
        if csv.exists():
            available.append((label, csv))
        else:
            print(f"[pareto] Missing {csv} — skipping {label}")

    if not available:
        print("[pareto] No CSVs found — skipping")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, (label, csv) in zip(axes, available):
        df       = pd.read_csv(csv).sort_values("avg_relative_flops")
        unpruned = df["avg_vqa_acc_unpruned"].iloc[0]

        ax.axhline(unpruned * 100, color=_COLORS["unpruned"], linestyle="--",
                   linewidth=1.5, label=f"Unpruned ({unpruned*100:.1f}%)")
        ax.plot(df["avg_relative_flops"], df["avg_vqa_acc_papit"]  * 100,
                "o-",  color=_COLORS["papit"],  linewidth=2, markersize=6, label="PAPIT")
        ax.plot(df["avg_relative_flops"], df["avg_vqa_acc_random"] * 100,
                "s--", color=_COLORS["random"], linewidth=2, markersize=6, label="Random")

        for _, row in df.iterrows():
            k_pct = int(round(row["avg_token_keep_ratio"] * 100))
            ax.annotate(f"{k_pct}%",
                        xy=(row["avg_relative_flops"], row["avg_vqa_acc_papit"] * 100),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=9, color=_COLORS["papit"])

        ax.set_xlabel("Relative attention FLOPs", fontsize=FONT)
        ax.set_ylabel("VQA accuracy (%)",          fontsize=FONT)
        ax.set_title(label,                        fontsize=FONT)
        ax.legend(fontsize=FONT - 1)
        ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(left=0)
        ax.tick_params(labelsize=FONT)

    fig.suptitle("Accuracy–Efficiency Pareto (LLaVA-1.5-7B, 700 samples/dataset)",
                 fontsize=FONT)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[pareto] Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-qualitative", action="store_true",
                   help="Skip fig_qualitative (avoids loading LLaVA)")
    p.add_argument("--skip-efficiency",  action="store_true")
    p.add_argument("--skip-pareto",      action="store_true")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    REPORT.mkdir(parents=True, exist_ok=True)
    print(f"Device : {device}")
    print(f"Output : {REPORT}/")

    if not args.skip_efficiency:
        gen_efficiency()

    if not args.skip_pareto:
        gen_pareto()

    if not args.skip_qualitative:
        gen_qualitative(device)

    print("\nAll done.")


if __name__ == "__main__":
    main()
