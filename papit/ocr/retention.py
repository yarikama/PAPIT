"""OCR-guided patch retention: force-keep patches that overlap with detected text regions."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def ocr_forced_indices(
    image_path: str,
    grid_size: int,
    lang_list: list[str] | None = None,
) -> tuple[list[int], list]:
    """Run EasyOCR and return (forced_patch_indices, raw_ocr_results).

    Forced indices are all patch grid cells that overlap with any detected text box.
    """
    import easyocr

    if lang_list is None:
        lang_list = ["en"]

    reader = easyocr.Reader(lang_list, gpu=_cuda_available())
    img_np = np.array(Image.open(image_path).convert("RGB"))
    ocr_results = reader.readtext(img_np)

    h, w = img_np.shape[:2]
    cell_w = w / grid_size
    cell_h = h / grid_size

    forced: set[int] = set()
    for item in ocr_results:
        box = item[0]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x0, x1 = max(min(xs), 0), min(max(xs), w - 1)
        y0, y1 = max(min(ys), 0), min(max(ys), h - 1)

        c0 = max(0, min(int(x0 // cell_w), grid_size - 1))
        c1 = max(0, min(int(x1 // cell_w), grid_size - 1))
        r0 = max(0, min(int(y0 // cell_h), grid_size - 1))
        r1 = max(0, min(int(y1 // cell_h), grid_size - 1))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                forced.add(r * grid_size + c)

    return sorted(forced), ocr_results


def merge_topk_with_forced(
    scores: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
    forced_indices: list[int],
) -> list[int]:
    """Merge OCR-forced indices with top-k score ranking within the same budget *k*.

    Priority order:
    1. OCR-forced patches that are already in top-k.
    2. OCR-forced patches not in top-k.
    3. Score-ranked patches until budget is filled.
    """
    forced_set = set(int(i) for i in forced_indices)
    ranked = torch.argsort(scores, descending=True).detach().cpu().tolist()

    final: list[int] = []

    for idx in topk_indices.detach().cpu().tolist():
        if idx in forced_set and idx not in final:
            final.append(idx)

    for idx in ranked:
        if idx in forced_set and idx not in final:
            final.append(idx)

    for idx in ranked:
        if idx not in final:
            final.append(idx)
        if len(final) >= k:
            break

    return final[:k]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
