"""Batch benchmark runner: evaluates Base / OCR-guided / Random / Risk-aware variants."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from papit.core.config import PAPITConfig
from papit.core.pruner import PromptAwarePruner
from papit.ocr.retention import merge_topk_with_forced, ocr_forced_indices
from papit.utils.metrics import exact_match, normalize_text, pct, random_topk_indices, token_f1
from papit.utils.visualization import build_pruned_image


# ---------------------------------------------------------------------------
# QA model helpers (lazy-loaded to avoid VRAM cost when not needed)
# ---------------------------------------------------------------------------

_qa_processor = None
_qa_model = None


def _get_qa_model(device: str):
    global _qa_processor, _qa_model
    if _qa_model is None:
        from transformers import BlipForQuestionAnswering, BlipProcessor

        model_id = "Salesforce/blip-vqa-base"
        _qa_processor = BlipProcessor.from_pretrained(model_id)
        _qa_model = BlipForQuestionAnswering.from_pretrained(model_id).to(device)
        _qa_model.eval()
    return _qa_processor, _qa_model


def answer_with_blip(
    image_pil: Image.Image,
    question: str,
    device: str = "cpu",
    max_new_tokens: int = 20,
) -> str:
    processor, model = _get_qa_model(device)
    inputs = processor(images=image_pil, text=question, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Batch benchmark
# ---------------------------------------------------------------------------


def run_batch_benchmark(
    dataset_csv: str,
    retention_list: list[float] = (0.25, 0.5, 0.75),
    anchor_strategy: str = "global_mean",
    max_samples: int | None = None,
    seed: int = 42,
    device: str | None = None,
    output_dir: str = "outputs",
) -> pd.DataFrame:
    """Run Base / OCR-guided / Random comparisons across multiple retention ratios.

    Parameters
    ----------
    dataset_csv:
        Path to CSV with columns ``image_path``, ``question``, and optionally ``answer``.
    output_dir:
        Directory where ``benchmark_detailed.csv`` and ``benchmark_summary.csv`` are written.

    Returns
    -------
    Detailed per-sample DataFrame.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = pd.read_csv(dataset_csv)
    required = {"image_path", "question"}
    if not required.issubset(set(samples.columns)):
        raise ValueError(f"CSV must contain columns: {required}")

    if max_samples is not None:
        samples = samples.head(max_samples).copy()

    rows: list[dict[str, Any]] = []

    for ratio in retention_list:
        cfg = PAPITConfig(
            retention_ratio=ratio,
            anchor_strategy=anchor_strategy,  # type: ignore[arg-type]
            device=device,
        )
        pruner = PromptAwarePruner(cfg)

        for i, sample in samples.iterrows():
            img_path = str(sample["image_path"])
            question = str(sample["question"])
            gold = str(sample.get("answer", ""))

            t0 = time.perf_counter()
            out = pruner.run(img_path, question)
            prune_time = time.perf_counter() - t0

            total_tokens = int(out.patch_tokens.shape[0])
            k = int(out.topk_indices.shape[0])
            grid = pruner.grid_size

            forced, _ = ocr_forced_indices(img_path, grid)
            forced_set = set(forced)

            base_idx = out.topk_indices.detach().cpu().tolist()
            ocr_idx = merge_topk_with_forced(out.scores, out.topk_indices, k=k, forced_indices=forced)
            rnd_idx = random_topk_indices(total_tokens, k=k, seed=seed + i + int(ratio * 1000))  # type: ignore[operator]

            orig_img = Image.open(img_path).convert("RGB")
            base_img = build_pruned_image(img_path, base_idx, grid)
            ocr_img = build_pruned_image(img_path, ocr_idx, grid)
            rnd_img = build_pruned_image(img_path, rnd_idx, grid)

            ans_orig = answer_with_blip(orig_img, question, device)
            ans_base = answer_with_blip(base_img, question, device)
            ans_ocr = answer_with_blip(ocr_img, question, device)
            ans_rnd = answer_with_blip(rnd_img, question, device)

            row: dict[str, Any] = {
                "sample_index": int(i),  # type: ignore[arg-type]
                "image_path": img_path,
                "question": question,
                "retention_ratio": ratio,
                "k": k,
                "total_tokens": total_tokens,
                "forced_patch_count": len(forced_set),
                "prune_time_sec": prune_time,
                "answer_orig": ans_orig,
                "answer_base": ans_base,
                "answer_ocr": ans_ocr,
                "answer_random": ans_rnd,
                "base_cov_pct": pct(len(set(base_idx) & forced_set), len(forced_set)),
                "ocr_cov_pct": pct(len(set(ocr_idx) & forced_set), len(forced_set)),
                "random_cov_pct": pct(len(set(rnd_idx) & forced_set), len(forced_set)),
                "base_same_as_orig": int(normalize_text(ans_base) == normalize_text(ans_orig)),
                "ocr_same_as_orig": int(normalize_text(ans_ocr) == normalize_text(ans_orig)),
                "random_same_as_orig": int(normalize_text(ans_rnd) == normalize_text(ans_orig)),
            }

            if gold.strip():
                row["gold_answer"] = gold
                for tag, ans in [("orig", ans_orig), ("base", ans_base), ("ocr", ans_ocr), ("random", ans_rnd)]:
                    row[f"{tag}_em"] = exact_match(ans, gold)
                    row[f"{tag}_f1"] = token_f1(ans, gold)

            rows.append(row)

    results_df = pd.DataFrame(rows)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "benchmark_detailed.csv", index=False)

    summary = _aggregate(results_df)
    summary.to_csv(out_dir / "benchmark_summary.csv", index=False)

    return results_df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ratio, g in df.groupby("retention_ratio"):
        row: dict[str, Any] = {
            "retention_ratio": ratio,
            "n_samples": len(g),
            "avg_prune_time_sec": g["prune_time_sec"].mean(),
            "avg_base_cov_pct": g["base_cov_pct"].mean(),
            "avg_ocr_cov_pct": g["ocr_cov_pct"].mean(),
            "avg_random_cov_pct": g["random_cov_pct"].mean(),
            "avg_base_same_as_orig": g["base_same_as_orig"].mean(),
            "avg_ocr_same_as_orig": g["ocr_same_as_orig"].mean(),
            "avg_random_same_as_orig": g["random_same_as_orig"].mean(),
        }
        for metric in ("em", "f1"):
            for tag in ("base", "ocr", "random"):
                col = f"{tag}_{metric}"
                if col in g.columns:
                    row[f"avg_{col}"] = g[col].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("retention_ratio").reset_index(drop=True)
