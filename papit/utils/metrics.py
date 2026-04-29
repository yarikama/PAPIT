"""Shared text-normalisation and scoring utilities."""
from __future__ import annotations

import random


def normalize_text(s: object) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    for ch in [",", ".", "!", "?", ":", ";", '"', "'", "(", ")"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p = normalize_text(pred).split()
    g = normalize_text(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    p_count: dict[str, int] = {}
    g_count: dict[str, int] = {}
    for t in p:
        p_count[t] = p_count.get(t, 0) + 1
    for t in g:
        g_count[t] = g_count.get(t, 0) + 1
    overlap = sum(min(c, g_count[t]) for t, c in p_count.items() if t in g_count)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_text(pred) == normalize_text(gold))


def pct(num: int | float, den: int | float) -> float:
    return 0.0 if den == 0 else 100.0 * num / den


def random_topk_indices(total_tokens: int, k: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    idx = list(range(total_tokens))
    rng.shuffle(idx)
    return sorted(idx[:k])


def vqa_soft_accuracy(pred: str, answer_list: list[str]) -> float:
    """VQA soft accuracy.

    For multi-annotator datasets (VQA v2, TextVQA): min(count_matching / 3, 1.0).
    For single-answer datasets (GQA): lenient match — counts as correct if the
    expected answer word(s) appear as a token sequence inside the prediction.
    This handles LLMs that answer in full sentences ("The car is blue.") rather
    than single words ("blue"), which is common with 4-bit quantised models.
    """
    pred_norm = normalize_text(pred)
    if len(answer_list) == 1:
        gold_norm = normalize_text(answer_list[0])
        # Exact match first (fast path)
        if pred_norm == gold_norm:
            return 1.0
        # Lenient: gold tokens appear as a contiguous substring in prediction
        gold_tokens = gold_norm.split()
        pred_tokens = pred_norm.split()
        n = len(gold_tokens)
        for i in range(len(pred_tokens) - n + 1):
            if pred_tokens[i:i + n] == gold_tokens:
                return 1.0
        return 0.0
    count = sum(1 for a in answer_list if normalize_text(a) == pred_norm)
    return min(count / 3.0, 1.0)


def patch_recall(
    selected_indices: list[int],
    ocr_boxes: list[dict],
    grid_size: int,
) -> float:
    """Fraction of text-region patches retained by pruning.

    ocr_boxes entries use TextVQA format:
    ``{"bounding_box": {"top_left_x": 0.1, "top_left_y": 0.2, "width": 0.3, "height": 0.05}}``
    where coordinates are normalised 0-1 relative to image dimensions.
    """
    text_patches: set[int] = set()
    for item in ocr_boxes:
        bb = item.get("bounding_box", {})
        x0 = float(bb.get("top_left_x", 0.0))
        y0 = float(bb.get("top_left_y", 0.0))
        w = float(bb.get("width", 0.0))
        h = float(bb.get("height", 0.0))
        x1 = min(x0 + w, 1.0)
        y1 = min(y0 + h, 1.0)

        c0 = max(0, min(int(x0 * grid_size), grid_size - 1))
        c1 = max(0, min(int(x1 * grid_size), grid_size - 1))
        r0 = max(0, min(int(y0 * grid_size), grid_size - 1))
        r1 = max(0, min(int(y1 * grid_size), grid_size - 1))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                text_patches.add(r * grid_size + c)

    if not text_patches:
        return 1.0  # no text in image → recall trivially 1.0
    kept = set(selected_indices) & text_patches
    return len(kept) / len(text_patches)
