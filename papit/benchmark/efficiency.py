"""Efficiency benchmark: latency, token reduction, and peak GPU memory."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from papit.core.config import PAPITConfig
from papit.core.pruner import PromptAwarePruner
from papit.ocr.retention import merge_topk_with_forced, ocr_forced_indices
from papit.risk.awareness import (
    INSTRUCTION_KEYWORDS,
    SAFETY_KEYWORDS,
    classify_risk_indices,
    mask_indices_on_image,
    risk_aware_topk,
    text_to_patch_indices,
)
from papit.utils.visualization import build_pruned_image


def measure_variant(
    image_path: str,
    question: str,
    retention_ratio: float = 0.5,
    force_ocr: bool = False,
    risk_aware: bool = False,
    runs: int = 3,
    device: str | None = None,
    qa_fn=None,
) -> dict[str, Any]:
    """Measure latency and memory for a single configuration.

    Parameters
    ----------
    qa_fn:
        Optional callable ``(image_pil, question) -> str``. When provided, QA
        latency is also measured.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PAPITConfig(retention_ratio=retention_ratio, device=device)
    pruner = PromptAwarePruner(cfg)

    # Warmup – both pruner and qa_fn must be warm so all k values are comparable
    _wu_out = pruner.run(image_path, question)
    if qa_fn is not None:
        _wu_idx = [int(x) for x in _wu_out.topk_indices.detach().cpu().tolist()]
        qa_fn(build_pruned_image(image_path, _wu_idx, pruner.grid_size), question)

    prune_times: list[float] = []
    qa_times: list[float] = []
    mem_peaks: list[float] = []
    keep_counts: list[int] = []
    total_counts: list[int] = []

    for _ in range(runs):
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = pruner.run(image_path, question)
        grid = pruner.grid_size

        base_idx = [int(x) for x in out.topk_indices.detach().cpu().tolist()]
        final_idx = base_idx

        if force_ocr or risk_aware:
            forced, ocr_res = ocr_forced_indices(image_path, grid)
            final_idx = merge_topk_with_forced(
                out.scores, out.topk_indices, k=len(base_idx), forced_indices=forced
            )
            if risk_aware:
                img_np = np.array(Image.open(image_path).convert("RGB"))
                per_text = text_to_patch_indices(ocr_res, img_np.shape[:2], grid)
                safety, instr = classify_risk_indices(per_text, SAFETY_KEYWORDS, INSTRUCTION_KEYWORDS)
                final_idx = risk_aware_topk(
                    scores=out.scores,
                    k=len(base_idx),
                    base_topk=final_idx,
                    safety_force_keep=safety,
                    instruction_blocklist=instr,
                )

        t1 = time.perf_counter()
        prune_times.append(t1 - t0)

        if qa_fn is not None:
            pruned_img = build_pruned_image(image_path, final_idx, grid)
            tq0 = time.perf_counter()
            qa_fn(pruned_img, question)
            qa_times.append(time.perf_counter() - tq0)

        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.synchronize()
            mem_peaks.append(torch.cuda.max_memory_allocated() / (1024**2))
        else:
            mem_peaks.append(float("nan"))

        keep_counts.append(len(final_idx))
        total_counts.append(int(out.patch_tokens.shape[0]))

    avg_keep = float(np.mean(keep_counts))
    avg_total = float(np.mean(total_counts))
    avg_prune = float(np.mean(prune_times))
    avg_qa = float(np.mean(qa_times)) if qa_times else float("nan")

    result: dict[str, Any] = {
        "retention_ratio": retention_ratio,
        "force_ocr": int(force_ocr),
        "risk_aware": int(risk_aware),
        "avg_tokens_kept": avg_keep,
        "avg_total_tokens": avg_total,
        "token_keep_ratio": avg_keep / max(avg_total, 1.0),
        "avg_prune_latency_sec": avg_prune,
        "avg_qa_latency_sec": avg_qa,
        "avg_end_to_end_sec": avg_prune + (avg_qa if not np.isnan(avg_qa) else 0.0),
        "avg_peak_gpu_mem_mb": float(np.nanmean(mem_peaks)),
    }
    return result


def run_efficiency_benchmark(
    image_path: str,
    question: str,
    retention_grid: list[float] = (1.0, 0.75, 0.5, 0.25),
    runs_per_setting: int = 3,
    device: str | None = None,
    qa_fn=None,
    output_path: str = "outputs/efficiency_benchmark.csv",
) -> pd.DataFrame:
    """Run all three pruning variants across *retention_grid* and save results."""
    rows: list[dict[str, Any]] = []
    for ratio in retention_grid:
        rows.append(measure_variant(image_path, question, ratio, False, False, runs_per_setting, device, qa_fn))
        rows.append(measure_variant(image_path, question, ratio, True, False, runs_per_setting, device, qa_fn))
        rows.append(measure_variant(image_path, question, ratio, True, True, runs_per_setting, device, qa_fn))

    df = (
        pd.DataFrame(rows)
        .sort_values(["retention_ratio", "force_ocr", "risk_aware"], ascending=[False, True, True])
        .reset_index(drop=True)
    )

    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
