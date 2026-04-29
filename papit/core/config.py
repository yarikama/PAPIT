from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

AnchorStrategy = Literal["none", "global_mean", "dropped_mean"]
# gradcam    : GradCAM backward through LLaVA's ViT (~100 ms overhead, highest accuracy at k≤25%)
# value_free : value-feature cosine scoring using v_proj values captured during the
#              already-running ViT forward pass (~0 ms extra overhead beyond text encoding)
ScoringMode = Literal["gradcam", "value_free"]


@dataclass(slots=True)
class PAPITConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    retention_ratio: float = 0.5
    anchor_strategy: AnchorStrategy = "global_mean"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scoring_mode: ScoringMode = "gradcam"


@dataclass(slots=True)
class PAPITOutput:
    patch_tokens: torch.Tensor
    text_embedding: torch.Tensor
    scores: torch.Tensor
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor
    pruned_tokens: torch.Tensor
    coords: list[tuple[int, int]]
    new_position_ids: torch.Tensor
    selected_position_ids: torch.Tensor
