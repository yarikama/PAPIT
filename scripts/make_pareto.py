#!/usr/bin/env python3
"""Generate Pareto plot (relative-FLOPs vs accuracy) on the three benchmarks
using the Phase-4 N=100 results, with PAPIT-Distill as the new line."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Charter", "Bitstream Charter", "Times"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False

OUT = Path("outputs")
DATASETS = ("gqa", "textvqa", "vqa_v2")
LABELS = {"gqa": "GQA", "textvqa": "TextVQA", "vqa_v2": "VQA v2"}
RETENTIONS = (0.25, 0.50, 0.75)


def rel_flops(k_frac: float, n: int = 576, lt: int = 50) -> float:
    """Relative attention FLOPs (Eq. 2)."""
    k = round(n * k_frac)
    return ((k + lt) ** 2) / ((n + lt) ** 2)


def collect():
    rows = []
    for ds in DATASETS:
        df = pd.read_csv(OUT / f"distill100k_attn_L8_{ds}.csv")
        for ret, sub in df.groupby("retention"):
            rows.append({
                "ds": ds, "ret": ret, "rel_flops": rel_flops(ret),
                "unpruned":      sub.acc_unpruned.mean() * 100,
                "random":        sub.acc_random.mean() * 100,
                "papit_clip":    sub.acc_papit.mean() * 100,
                "papit_distill": sub.acc_distill.mean() * 100,
            })
    return pd.DataFrame(rows)


def main():
    df = collect()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 1.9), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        sub = df[df.ds == ds].sort_values("rel_flops")
        unpruned = sub.unpruned.iloc[0]
        x = sub.rel_flops.values

        ax.axhline(unpruned, color="#444444", ls="--", lw=1.2, alpha=0.85,
                   label="Unpruned" if ds == DATASETS[0] else None)
        ax.plot(x, sub.random,        marker="o", lw=1.5, color="#4C72B0",
                label="Random"             if ds == DATASETS[0] else None)
        ax.plot(x, sub.papit_clip,    marker="s", lw=1.5, color="#DD8452",
                label="PAPIT-CLIP"         if ds == DATASETS[0] else None)
        ax.plot(x, sub.papit_distill, marker="D", lw=1.7, color="#C44E52",
                label="PAPIT-Distill (L=8)" if ds == DATASETS[0] else None)

        # mark retention pct on each x point
        for r, xi in zip(RETENTIONS, x):
            ax.annotate(f"{int(r*100)}%", (xi, sub.random.min() - 4),
                        ha="center", fontsize=7, color="#888")

        ax.set_title(LABELS[ds], fontsize=11)
        ax.set_xlabel("Relative attention FLOPs", fontsize=9)
        if ds == DATASETS[0]:
            ax.set_ylabel("VQA accuracy (%)", fontsize=9)
        ax.set_xlim(0.05, 1.0)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)

    fig.legend(loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_pdf = Path("report/fig_pareto.pdf")
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
