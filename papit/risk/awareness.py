"""Risk-aware patch selection: safety-keep and instruction-neutralisation."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from papit.utils.metrics import normalize_text

SAFETY_KEYWORDS: frozenset[str] = frozenset(
    {
        "stop", "yield", "warning", "danger", "hazard", "pedestrian", "school",
        "crosswalk", "fire", "exit", "caution", "biohazard", "poison", "emergency",
    }
)

INSTRUCTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "ignore", "system prompt", "developer message", "jailbreak", "bypass",
        "override", "do not follow", "reveal", "secret", "password", "prompt injection",
    }
)


def text_to_patch_indices(
    ocr_results: list,
    image_shape_hw: tuple[int, int],
    grid_size: int,
) -> list[tuple[str, set[int]]]:
    """Map each OCR result to the set of patch indices it overlaps.

    Returns a list of ``(text, patch_index_set)`` pairs.
    """
    h, w = image_shape_hw
    cell_w = w / grid_size
    cell_h = h / grid_size
    per_text: list[tuple[str, set[int]]] = []

    for item in ocr_results:
        box = item[0]
        text = str(item[1]).lower() if len(item) > 1 else ""
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = max(min(xs), 0), min(max(xs), w - 1)
        y0, y1 = max(min(ys), 0), min(max(ys), h - 1)

        c0 = max(0, min(int(x0 // cell_w), grid_size - 1))
        c1 = max(0, min(int(x1 // cell_w), grid_size - 1))
        r0 = max(0, min(int(y0 // cell_h), grid_size - 1))
        r1 = max(0, min(int(y1 // cell_h), grid_size - 1))

        idxs: set[int] = set()
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                idxs.add(r * grid_size + c)
        per_text.append((text, idxs))

    return per_text


def classify_risk_indices(
    per_text_indices: list[tuple[str, set[int]]],
    safety_keywords: frozenset[str] = SAFETY_KEYWORDS,
    instruction_keywords: frozenset[str] = INSTRUCTION_KEYWORDS,
) -> tuple[set[int], set[int]]:
    """Return ``(safety_indices, instruction_indices)`` from OCR text/patch pairs."""
    safety_idxs: set[int] = set()
    instruction_idxs: set[int] = set()

    for text, idxs in per_text_indices:
        text_norm = normalize_text(text)
        tokens = set(text_norm.split())

        if any(k in text_norm for k in safety_keywords) or tokens & safety_keywords:
            safety_idxs |= idxs
        if any(k in text_norm for k in instruction_keywords) or tokens & instruction_keywords:
            instruction_idxs |= idxs

    return safety_idxs, instruction_idxs


def risk_aware_topk(
    scores: torch.Tensor,
    k: int,
    base_topk: list[int],
    safety_force_keep: set[int],
    instruction_blocklist: set[int],
) -> list[int]:
    """Select *k* patches, forcing safety patches in and blocking instruction patches.

    Priority:
    1. Safety-critical patches (not blocked).
    2. Base top-k selection (not blocked).
    3. Score-ranked fill (excluding blocked).
    """
    ranked = torch.argsort(scores, descending=True).detach().cpu().tolist()
    final: list[int] = []

    for idx in sorted(int(x) for x in safety_force_keep):
        if idx not in instruction_blocklist and idx not in final:
            final.append(idx)

    for idx in (int(x) for x in base_topk):
        if idx not in instruction_blocklist and idx not in final:
            final.append(idx)

    for idx in (int(x) for x in ranked):
        if idx in instruction_blocklist:
            continue
        if idx not in final:
            final.append(idx)
        if len(final) >= k:
            break

    return final[:k]


def mask_indices_on_image(
    image_pil: Image.Image,
    indices_to_mask: set[int] | list[int],
    grid_size: int,
    fill_value: int = 0,
) -> Image.Image:
    """Black out (or fill) specific patch regions in *image_pil*."""
    arr = np.array(image_pil).copy()
    h, w = arr.shape[:2]
    cell_w = w / grid_size
    cell_h = h / grid_size

    for idx in sorted(int(i) for i in indices_to_mask):
        r, c = divmod(idx, grid_size)
        x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
        y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
        arr[y0:y1, x0:x1] = fill_value
    return Image.fromarray(arr)
