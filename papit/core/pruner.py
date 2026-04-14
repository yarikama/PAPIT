"""Prompt-Aware Pruner: CLIP-based cross-modal token scoring and selection."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import AnchorStrategy, PAPITConfig, PAPITOutput


class PromptAwarePruner:
    def __init__(self, config: PAPITConfig) -> None:
        self.config = config
        self.processor = CLIPProcessor.from_pretrained(config.model_id)
        self.model = CLIPModel.from_pretrained(config.model_id).to(config.device)
        self.model.eval()

    @torch.no_grad()
    def run(self, image_path: str, prompt: str) -> PAPITOutput:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[prompt], images=image, return_tensors="pt")
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        vision_outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=False,
            return_dict=True,
        )
        # Drop CLS token; project patch tokens into the shared CLIP embedding space.
        raw_patch_tokens = vision_outputs.last_hidden_state[:, 1:, :].squeeze(0)
        patch_tokens = self.model.visual_projection(raw_patch_tokens)

        text_embedding = self._extract_text_embedding(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Hybrid scoring: GradCAM for aggressive pruning, value features otherwise
        if self.config.retention_ratio <= 0.375:
            scores = self._gradcam_scores(inputs["pixel_values"], text_embedding)
        else:
            scores = self._value_scores(inputs["pixel_values"], text_embedding)
        topk_scores, topk_indices = self._select_topk(scores)
        selected_tokens = patch_tokens[topk_indices]
        pruned_tokens = self._append_anchor(selected_tokens, patch_tokens, topk_indices)
        coords = self._indices_to_coords(topk_indices.cpu())
        selected_position_ids, new_position_ids = self._remap_positions(topk_indices)

        return PAPITOutput(
            patch_tokens=patch_tokens,
            text_embedding=text_embedding,
            scores=scores,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            pruned_tokens=pruned_tokens,
            coords=coords,
            new_position_ids=new_position_ids,
            selected_position_ids=selected_position_ids,
        )

    def _extract_text_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        text_features = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        if isinstance(text_features, torch.Tensor):
            return text_features.squeeze(0)

        # Fallback for older transformers versions.
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return self.model.text_projection(text_outputs.pooler_output).squeeze(0)

    def _gradcam_scores(
        self,
        pixel_values: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Per-patch saliency via GradCAM on CLIP CLS text-image similarity."""
        saved: dict[str, torch.Tensor] = {}

        def fwd_hook(module: nn.Module, inp: object, out: object) -> None:
            saved["act"] = out[0] if isinstance(out, tuple) else out

        hook = self.model.vision_model.encoder.layers[-2].register_forward_hook(fwd_hook)
        try:
            with torch.enable_grad():
                vision_out = self.model.vision_model(pixel_values, return_dict=True)
                cls_feat = self.model.visual_projection(vision_out.pooler_output)
                sim = (
                    F.normalize(cls_feat, dim=-1)
                    @ F.normalize(text_embedding, dim=-1)
                ).squeeze()
                saved["act"].retain_grad()
                sim.backward()
        finally:
            hook.remove()

        act = saved["act"][0, 1:].detach()
        grad = saved["act"].grad[0, 1:]
        scores = (grad * act).sum(dim=-1).relu()

        if scores.sum() == 0:
            scores = torch.ones(scores.shape[0], device=self.config.device)

        return scores

    def _value_scores(
        self,
        pixel_values: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Per-patch saliency via MaskCLIP value features.

        Hooks the last CLIP attention layer's v_proj, projects V features to
        the CLIP shared space, and computes cosine similarity with the text
        embedding.  No backprop required.

        Returns
        -------
        scores : [N] float32 tensor
        """
        saved: dict[str, torch.Tensor] = {}
        hook = self.model.vision_model.encoder.layers[-1].self_attn.v_proj \
            .register_forward_hook(lambda m, inp, out: saved.update({"v": out}))

        with torch.no_grad():
            self.model.vision_model(pixel_values, return_dict=True)
        hook.remove()

        v_feats = saved["v"][0, 1:, :]                             # [N, D_vit]
        clip_feats = self.model.visual_projection(v_feats)          # [N, D_clip]
        scores = (
            F.normalize(clip_feats, dim=-1)
            @ F.normalize(text_embedding, dim=-1)
        )  # [N]
        return scores

    def _compute_scores(
        self,
        patch_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.shape[-1] != text_embedding.shape[-1]:
            raise ValueError(
                f"Dimension mismatch: patches={patch_tokens.shape[-1]}, "
                f"text={text_embedding.shape[-1]}"
            )
        return F.normalize(patch_tokens, p=2, dim=-1) @ F.normalize(
            text_embedding, p=2, dim=-1
        )

    def _select_topk(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = max(0.0, min(1.0, self.config.retention_ratio))
        k = max(1, int(round(scores.shape[0] * ratio)))
        return torch.topk(scores, k=k, largest=True, sorted=True)

    def _append_anchor(
        self,
        selected_tokens: torch.Tensor,
        patch_tokens: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        strategy: AnchorStrategy = self.config.anchor_strategy
        if strategy == "none":
            return selected_tokens
        if strategy == "global_mean":
            anchor = patch_tokens.mean(dim=0, keepdim=True)
            return torch.cat([selected_tokens, anchor], dim=0)
        if strategy == "dropped_mean":
            mask = torch.ones(patch_tokens.shape[0], device=patch_tokens.device, dtype=torch.bool)
            mask[selected_indices] = False
            anchor = (
                patch_tokens[mask].mean(dim=0, keepdim=True)
                if mask.any()
                else patch_tokens.mean(dim=0, keepdim=True)
            )
            return torch.cat([selected_tokens, anchor], dim=0)
        raise ValueError(f"Unknown anchor strategy: {strategy!r}")

    def _indices_to_coords(self, indices: torch.Tensor) -> list[tuple[int, int]]:
        vision_cfg = self.model.vision_model.config
        grid_size = max(
            int(getattr(vision_cfg, "image_size", 224))
            // int(getattr(vision_cfg, "patch_size", 16)),
            1,
        )
        return [(idx // grid_size, idx % grid_size) for idx in indices.tolist()]

    def _remap_positions(
        self, selected_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_position_ids = selected_indices.clone()
        new_position_ids = torch.arange(
            selected_indices.shape[0],
            device=selected_indices.device,
            dtype=torch.long,
        )
        return selected_position_ids, new_position_ids

    def project_for_llm(
        self,
        pruned_tokens: torch.Tensor,
        llm_hidden_dim: int = 4096,
        projector: Optional[nn.Module] = None,
    ) -> tuple[torch.Tensor, nn.Module]:
        """Project pruned image tokens into an LLM embedding space."""
        if projector is None:
            projector = nn.Linear(pruned_tokens.shape[-1], llm_hidden_dim).to(
                pruned_tokens.device
            )
        return projector(pruned_tokens), projector

    @property
    def grid_size(self) -> int:
        vision_cfg = self.model.vision_model.config
        return max(
            int(getattr(vision_cfg, "image_size", 224))
            // int(getattr(vision_cfg, "patch_size", 16)),
            1,
        )
