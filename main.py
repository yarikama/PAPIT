from __future__ import annotations

import argparse
import json
from pathlib import Path

from papit.pipeline import PAPITConfig, PromptAwarePruner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt-aware pruning for image tokens (PAPIT baseline)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Prompt used for token scoring")
    parser.add_argument("--retention", type=float, default=0.5, help="Retention ratio in [0, 1]")
    parser.add_argument(
        "--anchor",
        choices=["none", "global_mean", "dropped_mean"],
        default="global_mean",
        help="Anchor strategy after pruning",
    )
    parser.add_argument(
        "--model-id",
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model id",
    )
    parser.add_argument("--device", default=None, help="Device override, e.g. cpu or cuda")
    parser.add_argument("--json-out", default=None, help="Optional output path for JSON summary")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = PAPITConfig(
        model_id=args.model_id,
        retention_ratio=args.retention,
        anchor_strategy=args.anchor,
        device=args.device or PAPITConfig().device,
    )

    pruner = PromptAwarePruner(config)
    output = pruner.run(args.image, args.prompt)

    summary = {
        "image": str(Path(args.image)),
        "prompt": args.prompt,
        "model_id": args.model_id,
        "device": config.device,
        "retention_ratio": args.retention,
        "anchor_strategy": args.anchor,
        "total_patch_tokens": int(output.patch_tokens.shape[0]),
        "selected_patch_tokens": int(output.topk_indices.shape[0]),
        "final_sequence_length": int(output.pruned_tokens.shape[0]),
        "topk_indices": [int(i) for i in output.topk_indices.detach().cpu().tolist()],
        "topk_scores": [float(s) for s in output.topk_scores.detach().cpu().tolist()],
        "topk_coords": [{"row": int(r), "col": int(c)} for r, c in output.coords],
    }

    print(json.dumps(summary, indent=2))

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
