"""LLaVA-based evaluation: PAPIT vs random vs unpruned baselines.

Metrics computed per sample
---------------------------
- VQA soft accuracy  (PAPIT / random / unpruned)
- Pruning latency and generation latency
- Token keep ratio and relative attention FLOPs
- Patch recall on text regions (TextVQA only, when ``ocr_boxes`` column present)

Usage
-----
    from papit.benchmark.llava_runner import run_llava_benchmark
    df = run_llava_benchmark(
        dataset_csv="data/textvqa_val.csv",
        retention_list=[0.25, 0.5, 0.75],
        max_samples=500,
    )
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

import pandas as pd
import torch
from PIL import Image

from papit.core.config import PAPITConfig
from papit.integration.llava import PAPITLlavaRunner
from papit.utils.metrics import patch_recall, vqa_soft_accuracy

# ---------------------------------------------------------------------------
# Extended runner: adds random-selection baseline
# ---------------------------------------------------------------------------

_ANCHOR_STRATEGIES = {"global_mean", "dropped_mean"}


class _ExtendedRunner(PAPITLlavaRunner):
    """Adds ``generate_random`` to PAPITLlavaRunner without modifying it."""

    @torch.no_grad()
    def generate_random(
        self,
        image: Image.Image,
        prompt: str,
        k: int,
        seed: int = 0,
        max_new_tokens: int = 32,
        **generate_kwargs: Any,
    ) -> tuple[str, list[int]]:
        """Generate an answer using *k* randomly selected patches.

        Returns
        -------
        (decoded_answer, selected_patch_indices)
        """
        text = self._format_prompt(prompt)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        patch_for_proj, _, _v = self._extract_vit_features(inputs["pixel_values"])
        N = patch_for_proj.shape[0]
        k = min(max(k, 1), N)

        # Reproducible random selection (CPU generator → device-agnostic)
        gen = torch.Generator()
        gen.manual_seed(seed)
        indices = torch.randperm(N, generator=gen).to(self.device)[:k]

        selected = patch_for_proj[indices]

        # Apply anchor in ViT space (mirrors PAPITLlavaRunner._score_and_prune)
        strategy = self.config.anchor_strategy
        if strategy == "global_mean":
            anchor = patch_for_proj.mean(0, keepdim=True)
            selected = torch.cat([selected, anchor], dim=0)
        elif strategy == "dropped_mean":
            mask = torch.ones(N, dtype=torch.bool, device=self.device)
            mask[indices] = False
            anchor = (
                patch_for_proj[mask].mean(0, keepdim=True)
                if mask.any()
                else patch_for_proj.mean(0, keepdim=True)
            )
            selected = torch.cat([selected, anchor], dim=0)

        pruned_llm = self._project_through_mlp(selected)
        inputs_embeds, attention_mask = self._build_inputs_embeds(
            inputs["input_ids"], pruned_llm, inputs.get("attention_mask")
        )

        output_ids = self.llava.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        input_len = int(attention_mask.shape[1])
        answer = self._decode_generated_tokens(output_ids, input_len)
        return answer, indices.cpu().tolist()

    @torch.no_grad()
    def generate_ocr_forced(
        self,
        image: Image.Image,
        image_path: str,
        prompt: str,
        k: int,
        grid_size: int,
        max_new_tokens: int = 32,
        **generate_kwargs: Any,
    ) -> tuple[str, list[int]]:
        """Generate with PAPIT scoring + OCR-forced patch retention merged.

        Runs the same hybrid CLIP scoring as ``generate()``, then forces any
        OCR-detected text patches into the selection budget before projecting
        and generating.  This is the OCR-guided variant described in the report.

        Returns
        -------
        (decoded_answer, final_patch_indices)
        """
        from papit.ocr.retention import merge_topk_with_forced, ocr_forced_indices

        text = self._format_prompt(prompt)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        patch_for_proj, patch_for_scoring, v_feats = self._extract_vit_features(inputs["pixel_values"])
        N = patch_for_proj.shape[0]
        k = min(max(k, 1), N)

        # Hybrid scoring (mirrors _score_and_prune)
        if self.config.retention_ratio <= self._GRADCAM_THRESHOLD:
            scores = self._gradcam_scores(inputs["pixel_values"], prompt, N)
        else:
            scores = self._cosine_scores(patch_for_scoring, prompt)

        _, topk_indices = torch.topk(scores, k=k, largest=True, sorted=True)

        # OCR forced indices — fail gracefully if easyocr is not installed
        try:
            forced, _ = ocr_forced_indices(image_path, grid_size)
        except Exception:
            forced = []

        # Merge: OCR-detected text patches are guaranteed to be retained
        final_indices = merge_topk_with_forced(
            scores, topk_indices, k=k, forced_indices=forced
        )
        final_tensor = torch.tensor(final_indices, device=self.device, dtype=torch.long)
        selected = patch_for_proj[final_tensor]

        # Anchor (same strategy as _score_and_prune)
        strategy = self.config.anchor_strategy
        if strategy == "global_mean":
            anchor = patch_for_proj.mean(0, keepdim=True)
            selected = torch.cat([selected, anchor], dim=0)
        elif strategy == "dropped_mean":
            mask = torch.ones(N, dtype=torch.bool, device=self.device)
            mask[final_tensor] = False
            anchor = (
                patch_for_proj[mask].mean(0, keepdim=True)
                if mask.any()
                else patch_for_proj.mean(0, keepdim=True)
            )
            selected = torch.cat([selected, anchor], dim=0)

        pruned_llm = self._project_through_mlp(selected)
        inputs_embeds, attention_mask = self._build_inputs_embeds(
            inputs["input_ids"], pruned_llm, inputs.get("attention_mask")
        )

        output_ids = self.llava.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        input_len = int(attention_mask.shape[1])
        answer = self._decode_generated_tokens(output_ids, input_len)
        return answer, final_indices


# ---------------------------------------------------------------------------
# FLOPs helper
# ---------------------------------------------------------------------------

# Approximate text-token count for LLaVA chat template + question.
# (system prompt ~30 tokens + question ~20 tokens = ~50)
_N_TEXT_APPROX = 50


def _relative_flops(k: int, n_total: int, n_text: int = _N_TEXT_APPROX) -> float:
    """Attention FLOPs ratio: pruned / unpruned = (k + L_text)^2 / (N + L_text)^2."""
    return ((k + n_text) ** 2) / max((n_total + n_text) ** 2, 1)


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------


def run_llava_benchmark(
    dataset_csv: str,
    output_dir: str = "outputs/llava_eval",
    retention_list: list[float] = (0.25, 0.5, 0.75),
    llava_model_id: str = "llava-hf/llava-1.5-7b-hf",
    clip_model_id: str = "openai/clip-vit-large-patch14",
    max_samples: int | None = None,
    seed: int = 42,
    device: str | None = None,
    max_new_tokens: int = 32,
    anchor_strategy: str = "global_mean",
    force_ocr: bool = False,
) -> pd.DataFrame:
    """Evaluate PAPIT, random, and unpruned LLaVA across retention ratios.

    Parameters
    ----------
    dataset_csv:
        CSV produced by ``scripts/prepare_datasets.py``.
        Required columns: ``image_path``, ``question``, ``answer_list`` (JSON list).
        Optional column: ``ocr_boxes`` (JSON list, TextVQA only) for patch recall.
    output_dir:
        Where ``llava_benchmark_detailed.csv`` and ``llava_benchmark_summary.csv``
        are written.
    retention_list:
        Retention ratios to sweep (e.g. ``[0.25, 0.5, 0.75]``).
    llava_model_id:
        HuggingFace model ID for LLaVA-1.5.
    clip_model_id:
        CLIP model used for cross-modal scoring (must match LLaVA's vision tower).
    max_samples:
        Subsample the dataset (useful for quick sanity checks).
    seed:
        Controls subsampling and per-sample random-baseline seeds.
    max_new_tokens:
        Token generation cap. 32 is sufficient for most VQA answers.
    anchor_strategy:
        Anchor token strategy passed to PAPITConfig: ``"global_mean"``,
        ``"dropped_mean"``, or ``"none"``.
    force_ocr:
        When True, also run the OCR-forced variant (requires easyocr).
        Adds columns ``answer_ocr``, ``vqa_acc_ocr``, ``latency_ocr_sec``,
        and ``ocr_patch_recall`` (when OCR boxes are present).

    Returns
    -------
    Detailed per-sample DataFrame (also written to *output_dir*).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = pd.read_csv(dataset_csv)
    for col in ("image_path", "question", "answer_list"):
        if col not in samples.columns:
            raise ValueError(f"CSV must contain column: {col!r}")

    if max_samples is not None and max_samples < len(samples):
        samples = samples.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    # Load LLaVA once and reuse across all samples / ratios.
    cfg = PAPITConfig(
        retention_ratio=retention_list[0],
        device=device,
        anchor_strategy=anchor_strategy,
    )
    runner = _ExtendedRunner(
        llava_model_id=llava_model_id,
        clip_model_id=clip_model_id,
        config=cfg,
        device=device,
    )

    has_ocr_col = "ocr_boxes" in samples.columns
    # Grid size from LLaVA vision config (e.g. 336/14 = 24 for LLaVA-1.5)
    vision_cfg = runner.llava.config.vision_config
    grid_size: int = int(vision_cfg.image_size) // int(vision_cfg.patch_size)

    rows: list[dict[str, Any]] = []

    for i, sample in tqdm(samples.iterrows(), total=len(samples), desc="Samples", unit="sample"):
        img_path = str(sample["image_path"])
        question = str(sample["question"])
        answer_list: list[str] = json.loads(str(sample["answer_list"]))

        image = Image.open(img_path).convert("RGB")

        # ── Unpruned baseline (once per sample) ───────────────────────────
        t0 = time.perf_counter()
        unpruned_ans = runner.generate_unpruned(
            image, question, max_new_tokens=max_new_tokens
        )
        latency_unpruned = time.perf_counter() - t0
        acc_unpruned = vqa_soft_accuracy(unpruned_ans, answer_list)

        # ── PAPIT + Random at each retention ratio ─────────────────────────
        for ratio in retention_list:
            runner.config.retention_ratio = ratio

            # PAPIT
            t0 = time.perf_counter()
            papit_out = runner.generate(image, question, max_new_tokens=max_new_tokens)
            latency_papit = time.perf_counter() - t0

            info = papit_out.pruning_info
            k: int = info.selected_patches
            N: int = info.total_patches

            # Random baseline (same k; unique seed per sample to avoid bias)
            t0 = time.perf_counter()
            random_ans, random_indices = runner.generate_random(
                image,
                question,
                k=k,
                seed=seed + int(i),  # type: ignore[arg-type]
                max_new_tokens=max_new_tokens,
            )
            latency_random = time.perf_counter() - t0

            acc_papit = vqa_soft_accuracy(papit_out.answer, answer_list)
            acc_random = vqa_soft_accuracy(random_ans, answer_list)

            # OCR-forced variant (optional)
            ocr_ans: str | None = None
            ocr_indices: list[int] = []
            latency_ocr = float("nan")
            if force_ocr:
                t0 = time.perf_counter()
                ocr_ans, ocr_indices = runner.generate_ocr_forced(
                    image,
                    img_path,
                    question,
                    k=k,
                    grid_size=grid_size,
                    max_new_tokens=max_new_tokens,
                )
                latency_ocr = time.perf_counter() - t0

            row: dict[str, Any] = {
                "sample_index": int(i),  # type: ignore[arg-type]
                "image_path": img_path,
                "question": question,
                "retention_ratio": ratio,
                "k": k,
                "total_tokens": N,
                "token_keep_ratio": k / max(N, 1),
                # Relative attention FLOPs: (k + L_text)^2 / (N + L_text)^2
                "relative_flops": _relative_flops(k, N),
                # Answers
                "answer_unpruned": unpruned_ans,
                "answer_papit": papit_out.answer,
                "answer_random": random_ans,
                # VQA soft accuracy
                "vqa_acc_unpruned": acc_unpruned,
                "vqa_acc_papit": acc_papit,
                "vqa_acc_random": acc_random,
                # Latency (seconds)
                "latency_unpruned_sec": latency_unpruned,
                "latency_papit_sec": latency_papit,
                "latency_random_sec": latency_random,
                # OCR-forced variant
                "answer_ocr": ocr_ans,
                "vqa_acc_ocr": vqa_soft_accuracy(ocr_ans, answer_list) if ocr_ans is not None else float("nan"),
                "latency_ocr_sec": latency_ocr,
            }

            # TextVQA patch recall (only when ocr_boxes column is present)
            if has_ocr_col and pd.notna(sample.get("ocr_boxes")):
                ocr_boxes: list[dict] = json.loads(str(sample["ocr_boxes"]))
                row["papit_patch_recall"] = patch_recall(
                    info.selected_indices, ocr_boxes, grid_size
                )
                row["random_patch_recall"] = patch_recall(
                    random_indices, ocr_boxes, grid_size
                )
                if force_ocr and ocr_indices:
                    row["ocr_patch_recall"] = patch_recall(
                        ocr_indices, ocr_boxes, grid_size
                    )

            rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "llava_benchmark_detailed.csv", index=False)

    summary = _aggregate(df)
    summary.to_csv(out_dir / "llava_benchmark_summary.csv", index=False)

    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ratio, g in df.groupby("retention_ratio"):
        row: dict[str, Any] = {
            "retention_ratio": ratio,
            "n_samples": len(g),
            "avg_token_keep_ratio": g["token_keep_ratio"].mean(),
            "avg_relative_flops": g["relative_flops"].mean(),
            "avg_vqa_acc_unpruned": g["vqa_acc_unpruned"].mean(),
            "avg_vqa_acc_papit": g["vqa_acc_papit"].mean(),
            "avg_vqa_acc_random": g["vqa_acc_random"].mean(),
            "avg_latency_unpruned_sec": g["latency_unpruned_sec"].mean(),
            "avg_latency_papit_sec": g["latency_papit_sec"].mean(),
            "avg_latency_random_sec": g["latency_random_sec"].mean(),
        }
        for col in ("papit_patch_recall", "random_patch_recall", "vqa_acc_ocr", "ocr_patch_recall"):
            if col in g.columns:
                row[f"avg_{col}"] = g[col].mean()
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values("retention_ratio")
        .reset_index(drop=True)
    )
