"""Patch-level image utilities for visualising and materialising pruning decisions."""
from __future__ import annotations

import numpy as np
from PIL import Image


def build_pruned_image(
    image_path: str,
    kept_indices: list[int],
    grid_size: int,
    fill_value: int = 0,
) -> Image.Image:
    """Return a copy of the image with non-retained patches blacked out."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    cell_w = w / grid_size
    cell_h = h / grid_size

    kept = set(int(x) for x in kept_indices)
    for idx in range(grid_size * grid_size):
        if idx in kept:
            continue
        r, c = divmod(idx, grid_size)
        x0 = int(c * cell_w)
        x1 = int((c + 1) * cell_w)
        y0 = int(r * cell_h)
        y1 = int((r + 1) * cell_h)
        arr[y0:y1, x0:x1] = fill_value
    return Image.fromarray(arr)


def draw_patch_rects(
    ax,
    coords: list[tuple[int, int]],
    image_size_wh: tuple[int, int],
    grid_size: int,
    edgecolor: str = "red",
    linewidth: float = 1.5,
) -> None:
    """Draw patch rectangles onto a matplotlib Axes instance."""
    import matplotlib.patches as mpatches

    w, h = image_size_wh
    cell_w = w / grid_size
    cell_h = h / grid_size
    for r, c in coords:
        rect = mpatches.Rectangle(
            (c * cell_w, r * cell_h),
            cell_w,
            cell_h,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor="none",
        )
        ax.add_patch(rect)
