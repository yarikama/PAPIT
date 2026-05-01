"""CLI entry point: ``papit``.

Two modes:

1. ``papit <image> <prompt>``                          → score-only (PAPIT-CLIP)
2. ``papit <image> <prompt> --generate --predictor P`` → end-to-end demo:
   runs LLaVA-1.5-7B with each scorer (Unpruned / Random / PAPIT-CLIP /
   PAPIT-Distill) and prints the four answers side by side. Optionally
   saves a 4-panel PNG showing which patches each method kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _score_only(args) -> None:
    """Score-only path — no LLaVA, just CLIP saliency."""
    import torch
    from papit.core.config import PAPITConfig
    from papit.core.pruner import PromptAwarePruner

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = PAPITConfig(
        retention_ratio=args.retention,
        anchor_strategy=args.anchor,
        device=device,
    )
    pruner = PromptAwarePruner(config)
    result = pruner.run(args.image, args.prompt)

    summary = {
        "image": args.image,
        "prompt": args.prompt,
        "device": device,
        "retention_ratio": args.retention,
        "total_patch_tokens": int(result.patch_tokens.shape[0]),
        "selected_patch_tokens": int(result.topk_indices.shape[0]),
        "top20_scores": [round(float(s), 4) for s in result.topk_scores[:20].cpu()],
        "top20_coords": [[int(r), int(c)] for r, c in result.coords[:20]],
    }
    print(json.dumps(summary, indent=2))


def _generate_demo(args) -> None:
    """End-to-end demo: load LLaVA + (optionally) the distilled predictor,
    run each method on the same (image, question), print four answers."""
    import torch
    from PIL import Image
    from papit.core.config import PAPITConfig
    from papit.integration.llava import PAPITLlavaRunner

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from train_distill import PatchScorePredictor  # noqa: E402
    from train_distill_arch import ARCHS  # noqa: E402

    methods = (
        ["unpruned", "random", "papit_clip", "papit_distill"]
        if args.method == "all"
        else [args.method]
    )
    if "papit_distill" in methods and not args.predictor:
        sys.exit("--predictor PATH is required when method includes 'distill'")

    print(f"Loading LLaVA-1.5-7B (device={args.device or 'auto'}) …")
    runner = PAPITLlavaRunner(
        config=PAPITConfig(retention_ratio=args.retention),
        device=args.device,
        attn_implementation="eager",
    )
    device = runner.device

    predictor = None
    if args.predictor:
        ckpt = torch.load(args.predictor, weights_only=True, map_location=device)
        arch = ckpt.get("arch", "mlp2")
        p, t = ckpt.get("patch_dim", 1024), ckpt.get("text_dim", 768)
        if arch == "mlp2" and "hidden" in ckpt:
            predictor = PatchScorePredictor(patch_dim=p, text_dim=t, hidden=ckpt["hidden"])
        else:
            predictor = ARCHS[arch](patch_dim=p, text_dim=t)
        predictor.load_state_dict(ckpt["model_state"])
        predictor = predictor.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    text = runner._format_prompt(args.prompt)
    inputs = runner.processor(images=image, text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
    n_patches = patch_for_proj.shape[0]
    k = int(round(n_patches * args.retention))

    text_emb = runner._get_text_embedding(args.prompt)
    s_clip = runner._gradcam_scores(inputs["pixel_values"], args.prompt, n_patches)
    if predictor is not None:
        with torch.no_grad():
            s_distill = predictor(
                patch_for_proj.unsqueeze(0).float(), text_emb.unsqueeze(0).float()
            )[0]
    rand_indices = torch.randperm(n_patches, device=device)[:k]

    selections: dict[str, torch.Tensor | None] = {}
    if "unpruned" in methods:
        selections["unpruned"] = None
    if "random" in methods:
        selections["random"] = rand_indices
    if "papit_clip" in methods:
        selections["papit_clip"] = torch.topk(s_clip, k).indices
    if "papit_distill" in methods:
        selections["papit_distill"] = torch.topk(s_distill, k).indices

    answers: dict[str, str] = {}
    for name, idx in selections.items():
        print(f"  [{name}] generating …")
        if name == "unpruned":
            raw = runner.generate_unpruned(image, args.prompt, max_new_tokens=32)
        else:
            out = _generate_with_indices(
                runner, image, args.prompt, idx, patch_for_proj, inputs
            )
            raw = out.answer
        answers[name] = (
            raw.split("ASSISTANT:")[-1].strip()
            if "ASSISTANT:" in raw
            else raw.strip()
        )

    print()
    print(f"Image:    {args.image}")
    print(f"Question: {args.prompt}")
    print(f"k:        {int(args.retention * 100)}% ({k}/{n_patches} patches)")
    print()
    for name in methods:
        print(f"  {name:14s} → {answers[name]}")

    if args.save_viz:
        _save_viz(image, selections, answers, args.save_viz, n_patches)
        print(f"\nVisualization → {args.save_viz}")


def _generate_with_indices(runner, image, prompt, indices, patch_for_proj, inputs):
    """Generate using a pre-computed selected_indices tensor (skips
    runner's internal scorer). Mirrors PAPITLlavaRunner.generate but
    inserts our own selection."""
    import torch
    from papit.integration.llava import PAPITLlavaOutput, PAPITPruningInfo

    pruned_vit = patch_for_proj[indices]
    pruned_llm = runner._project_through_mlp(pruned_vit)
    inputs_embeds, attention_mask = runner._build_inputs_embeds(
        inputs["input_ids"], pruned_llm, inputs.get("attention_mask")
    )
    with torch.no_grad():
        output_ids = runner.llava.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=None,
            max_new_tokens=32,
        )
    answer = runner.processor.decode(output_ids[0], skip_special_tokens=True)
    return PAPITLlavaOutput(
        answer=answer,
        pruning_info=PAPITPruningInfo(
            total_patches=patch_for_proj.shape[0],
            selected_patches=int(indices.shape[0]),
            retention_ratio=runner.config.retention_ratio,
            selected_indices=indices.cpu().tolist(),
            scores=[],
        ),
    )


def _save_viz(image, selections, answers, out_path, n_patches):
    """Save a 1-row × N-method PNG: each panel shows the input image
    with non-selected patches blacked out, plus the model's answer."""
    import matplotlib.pyplot as plt
    import numpy as np

    grid = int(round(n_patches**0.5))
    patch_px = 336 // grid

    fig, axes = plt.subplots(1, len(selections), figsize=(3 * len(selections), 3.4))
    if len(selections) == 1:
        axes = [axes]
    img_resized = image.resize((336, 336))
    img_arr = np.asarray(img_resized).astype(np.float32) / 255.0

    for ax, (name, idx) in zip(axes, selections.items()):
        if idx is None:
            arr = img_arr
        else:
            mask = np.zeros(n_patches, dtype=np.float32)
            mask[idx.cpu().numpy()] = 1.0
            mask_2d = np.kron(mask.reshape(grid, grid), np.ones((patch_px, patch_px)))
            arr = img_arr * mask_2d[..., None]
        ax.imshow(arr)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(name, fontsize=10)
        ax.text(0.5, -0.06, f"→ {answers[name][:30]}", transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAPIT — Prompt-Aware Pruning for Image Tokens"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("prompt", help="Text prompt / question")
    parser.add_argument("--retention", type=float, default=0.5,
                        help="Token retention ratio (default: 0.5)")
    parser.add_argument("--anchor",
                        choices=["none", "global_mean", "dropped_mean"],
                        default="global_mean")
    parser.add_argument("--device", default=None,
                        help="Compute device (default: auto)")
    parser.add_argument("--generate", action="store_true",
                        help="End-to-end demo: load LLaVA and print actual answers")
    parser.add_argument("--method",
                        choices=["unpruned", "random", "papit_clip",
                                 "papit_distill", "all"],
                        default="all",
                        help="Which scorer to run when --generate (default: all)")
    parser.add_argument("--predictor", default=None,
                        help="Path to PAPIT-Distill predictor .pt "
                             "(required for --method papit_distill or all)")
    parser.add_argument("--save-viz", default=None,
                        help="Save a side-by-side PNG of pruned patches "
                             "(only with --generate)")
    args = parser.parse_args()

    if args.generate:
        _generate_demo(args)
    else:
        _score_only(args)


if __name__ == "__main__":
    main()
