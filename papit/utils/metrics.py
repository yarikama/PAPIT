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
