"""CLI entry point: ``papit``."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAPIT — Prompt-Aware Pruning for Image Tokens"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("prompt", help="Text prompt for cross-modal scoring")
    parser.add_argument("--retention", type=float, default=0.5, help="Token retention ratio (default: 0.5)")
    parser.add_argument(
        "--anchor",
        choices=["none", "global_mean", "dropped_mean"],
        default="global_mean",
    )
    parser.add_argument("--device", default=None, help="Compute device (default: auto)")
    args = parser.parse_args()

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
        "final_sequence_length": int(result.pruned_tokens.shape[0]),
        "top20_scores": [round(float(s), 4) for s in result.topk_scores[:20].cpu()],
        "top20_coords": [[int(r), int(c)] for r, c in result.coords[:20]],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
