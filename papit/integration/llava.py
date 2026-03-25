"""True token-level PAPIT integration for LLaVA-1.5.

Pruning is inserted between the vision tower and the MLP projector:

    image → ViT → [N patches] → PAPIT top-k → [k patches] → MLP projector → LLM

Key insight: LlavaModel.forward skips its own image-merging logic when
``pixel_values=None`` is passed alongside a pre-assembled ``inputs_embeds``.
We exploit this to replace N image-token embeddings with our k pruned ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from papit.core.config import PAPITConfig


@dataclass
class PAPITPruningInfo:
    total_patches: int
    selected_patches: int
    retention_ratio: float
    selected_indices: list[int]  # original patch indices chosen
    scores: list[float]          # top-k cosine similarity scores


@dataclass
class PAPITLlavaOutput:
    answer: str
    pruning_info: PAPITPruningInfo


class PAPITLlavaRunner:
    """Run LLaVA-1.5 with PAPIT token-level pruning.

    Parameters
    ----------
    llava_model_id:
        HuggingFace model ID, e.g. ``"llava-hf/llava-1.5-7b-hf"``.
    clip_model_id:
        Full CLIP model used for cross-modal scoring.  Must share the same
        vision encoder family as LLaVA's vision tower so patch grids align.
        Default: ``"openai/clip-vit-large-patch14"`` (matches LLaVA-1.5).
    config:
        PAPIT pruning configuration.  ``model_id`` is ignored here (LLaVA's
        own vision tower is used instead); only ``retention_ratio`` and
        ``anchor_strategy`` are read.
    device:
        Compute device.  Defaults to CUDA if available.

    Notes
    -----
    Memory layout:

    * LLaVA weights are loaded at float16 on GPU (``device_map="auto"``).
    * CLIP text encoder + projections are kept at float32 on the same device.
      The vision encoder weights are NOT duplicated — scoring uses features
      already produced by LLaVA's vision tower.
    """

    def __init__(
        self,
        llava_model_id: str = "llava-hf/llava-1.5-7b-hf",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        config: PAPITConfig | None = None,
        device: str | None = None,
    ) -> None:
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or PAPITConfig(device=self.device)

        # --- LLaVA ---------------------------------------------------------
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.llava: LlavaForConditionalGeneration = (
            LlavaForConditionalGeneration.from_pretrained(
                llava_model_id,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            ).eval()
        )
        self.processor = AutoProcessor.from_pretrained(llava_model_id)

        # --- CLIP (scoring only) -------------------------------------------
        # We only need visual_projection + text encoder from the full CLIP model.
        # LLaVA's vision tower already encodes images — no duplication needed.
        clip = CLIPModel.from_pretrained(clip_model_id)
        self.visual_projection = clip.visual_projection.to(self.device)
        self.text_model = clip.text_model.to(self.device)
        self.text_projection = clip.text_projection.to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        del clip  # free the full model; we keep only the lightweight components

        for m in (self.visual_projection, self.text_model, self.text_projection):
            m.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_vit_features(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run LLaVA's vision tower once and return two feature sets.

        Returns
        -------
        patch_for_projector : [N, D_vit]
            Hidden state from the layer configured for LLaVA's MLP projector
            (usually -2).  CLS token removed.
        patch_for_scoring : [N, D_vit]
            Final-layer hidden state.  Used with CLIP visual_projection for
            cross-modal cosine scoring.  CLS token removed.
        """
        vision_tower = self.llava.model.vision_tower
        feat_layer: int = getattr(self.llava.config, "vision_feature_layer", -2)
        llava_dtype = next(self.llava.parameters()).dtype

        out = vision_tower(
            pixel_values.to(dtype=llava_dtype),
            output_hidden_states=True,
            return_dict=True,
        )
        # [1, N+1, D_vit] → drop CLS → [N, D_vit]
        patch_for_projector = out.hidden_states[feat_layer][:, 1:].squeeze(0).float()
        patch_for_scoring = out.last_hidden_state[:, 1:].squeeze(0).float()
        return patch_for_projector, patch_for_scoring

    @torch.no_grad()
    def _get_text_embedding(self, prompt: str) -> torch.Tensor:
        """CLIP text embedding for the prompt.  Returns [D_clip]."""
        inputs = self.clip_processor(
            text=[prompt], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_out = self.text_model(**inputs, return_dict=True)
        return self.text_projection(text_out.pooler_output).squeeze(0)

    def _score_and_prune(
        self,
        patch_for_projector: torch.Tensor,
        patch_for_scoring: torch.Tensor,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(pruned_vit_features [k, D_vit], selected_indices [k])``.

        Scoring is performed in CLIP shared space; pruned features stay in ViT
        space so LLaVA's MLP projector can consume them unchanged.
        """
        # Project final-layer features to CLIP shared space for scoring.
        clip_feats = self.visual_projection(patch_for_scoring)  # [N, D_clip]
        text_emb = self._get_text_embedding(prompt)              # [D_clip]

        scores = (
            F.normalize(clip_feats, p=2, dim=-1)
            @ F.normalize(text_emb, p=2, dim=-1)
        )  # [N]

        N = scores.shape[0]
        k = max(1, int(round(N * self.config.retention_ratio)))
        _, indices = torch.topk(scores, k=k, largest=True, sorted=True)

        selected = patch_for_projector[indices]  # [k, D_vit]

        # Anchor appended in ViT space (same space as MLP projector input).
        strategy = self.config.anchor_strategy
        if strategy == "global_mean":
            anchor = patch_for_projector.mean(0, keepdim=True)
            selected = torch.cat([selected, anchor], dim=0)
        elif strategy == "dropped_mean":
            mask = torch.ones(N, dtype=torch.bool, device=patch_for_projector.device)
            mask[indices] = False
            anchor = (
                patch_for_projector[mask].mean(0, keepdim=True)
                if mask.any()
                else patch_for_projector.mean(0, keepdim=True)
            )
            selected = torch.cat([selected, anchor], dim=0)

        return selected, indices

    def _project_through_mlp(self, pruned_vit: torch.Tensor) -> torch.Tensor:
        """Apply LLaVA's MLP projector.  Returns [1, k, D_llm]."""
        llava_dtype = next(self.llava.parameters()).dtype
        return self.llava.model.multi_modal_projector(
            pruned_vit.to(dtype=llava_dtype).unsqueeze(0)
        )

    def _build_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        pruned_llm: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace the N consecutive image-token slots in input_ids with k
        pruned LLM-space features.

        LLaVA's processor inserts exactly N ``image_token_id`` values into
        ``input_ids`` (one per patch), forming a contiguous block.  We:
        1. Embed all tokens normally via ``embed_tokens``.
        2. Find the image block boundaries (first and last ``image_token_id``).
        3. Splice: ``[text_before | pruned_llm | text_after]``.
        4. Rebuild attention_mask to match the new sequence length.

        Returns ``(inputs_embeds, attention_mask)``.
        """
        image_token_id: int = self.llava.config.image_token_id
        embed_fn = self.llava.model.get_input_embeddings()

        # Build text embeddings for all positions (image slots get dummy id=0).
        text_ids = input_ids.clone()
        text_ids[input_ids == image_token_id] = 0
        text_embeds = embed_fn(text_ids)  # [1, L, D_llm]

        # Locate the image token block.
        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if image_positions.numel() == 0:
            # No image tokens — return text-only embeddings unchanged.
            attn = (
                attention_mask
                if attention_mask is not None
                else torch.ones(1, text_embeds.shape[1], dtype=torch.long, device=self.device)
            )
            return text_embeds, attn

        img_start = int(image_positions[0])
        img_end = int(image_positions[-1])
        k = pruned_llm.shape[1]

        merged_embeds = torch.cat(
            [text_embeds[:, :img_start], pruned_llm, text_embeds[:, img_end + 1 :]],
            dim=1,
        )

        if attention_mask is not None:
            img_mask = torch.ones(1, k, dtype=attention_mask.dtype, device=self.device)
            merged_mask = torch.cat(
                [attention_mask[:, :img_start], img_mask, attention_mask[:, img_end + 1 :]],
                dim=1,
            )
        else:
            merged_mask = torch.ones(
                1, merged_embeds.shape[1], dtype=torch.long, device=self.device
            )

        return merged_embeds, merged_mask

    def _format_prompt(self, prompt: str) -> str:
        """Apply LLaVA chat template if available, else fall back to legacy format."""
        try:
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            return self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        except Exception:
            return f"USER: <image>\n{prompt}\nASSISTANT:"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> PAPITLlavaOutput:
        """Generate an answer with PAPIT token-level pruning.

        Parameters
        ----------
        image:
            PIL image or path to an image file.
        prompt:
            Question / instruction text.
        max_new_tokens:
            Maximum number of tokens to generate.

        Returns
        -------
        PAPITLlavaOutput
            Decoded answer and pruning metadata.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        text = self._format_prompt(prompt)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 1. Extract ViT features (vision tower runs once).
        patch_for_proj, patch_for_score = self._extract_vit_features(
            inputs["pixel_values"]
        )

        # 2. Score patches and select top-k.
        pruned_vit, selected_indices = self._score_and_prune(
            patch_for_proj, patch_for_score, prompt
        )

        # 3. Project pruned ViT features through LLaVA's MLP projector.
        pruned_llm = self._project_through_mlp(pruned_vit)  # [1, k, D_llm]

        # 4. Build inputs_embeds: replace N image slots with k pruned tokens.
        inputs_embeds, attention_mask = self._build_inputs_embeds(
            inputs["input_ids"], pruned_llm, inputs.get("attention_mask")
        )

        # 5. Generate.
        # Pass inputs_embeds + pixel_values=None → LLaVA skips its own
        # image-merging logic and uses our pre-assembled embeddings directly.
        output_ids = self.llava.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)

        N = patch_for_proj.shape[0]
        k_actual = int(selected_indices.shape[0])
        clip_feats = self.visual_projection(patch_for_score)
        text_emb = self._get_text_embedding(prompt)
        all_scores = (
            F.normalize(clip_feats, p=2, dim=-1)
            @ F.normalize(text_emb, p=2, dim=-1)
        )
        topk_scores = all_scores[selected_indices]

        return PAPITLlavaOutput(
            answer=answer,
            pruning_info=PAPITPruningInfo(
                total_patches=N,
                selected_patches=k_actual,
                retention_ratio=self.config.retention_ratio,
                selected_indices=selected_indices.cpu().tolist(),
                scores=topk_scores.cpu().tolist(),
            ),
        )

    @torch.no_grad()
    def generate_unpruned(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int = 128,
        **generate_kwargs: Any,
    ) -> str:
        """Run standard LLaVA (no pruning) for comparison."""
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        text = self._format_prompt(prompt)
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output_ids = self.llava.generate(
            **inputs, max_new_tokens=max_new_tokens, **generate_kwargs
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True)

    def compare(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:
        """Run both pruned and unpruned and return a side-by-side comparison."""
        pruned_out = self.generate(image, prompt, max_new_tokens)
        unpruned_ans = self.generate_unpruned(image, prompt, max_new_tokens)

        info = pruned_out.pruning_info
        return {
            "prompt": prompt,
            "answer_unpruned": unpruned_ans,
            "answer_pruned": pruned_out.answer,
            "total_patches": info.total_patches,
            "selected_patches": info.selected_patches,
            "tokens_saved": info.total_patches - info.selected_patches,
            "retention_ratio": info.retention_ratio,
        }
