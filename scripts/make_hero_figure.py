#!/usr/bin/env python3
"""Render Fig 1 for the paper: per-method patch selection on hero
examples, with the model's actual generated answer underneath each
panel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_distill import PatchScorePredictor  # noqa: E402
from train_distill_arch import ARCHS  # noqa: E402

from papit.core.config import PAPITConfig  # noqa: E402
from papit.integration.llava import PAPITLlavaRunner  # noqa: E402

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Charter", "Bitstream Charter", "Times"]
rcParams["pdf.fonttype"] = 42

GRID = 24            # patches per side
PATCH_PX = 336 // GRID
N_PATCHES = GRID * GRID

HERO = [
    {"dataset": "textvqa", "retention": 0.25,
     "question_substr": "brand of this camera",
     "method_short": "TextVQA k=25%: camera brand"},
    {"dataset": "textvqa", "retention": 0.25,
     "question_substr": "is this denny's",
     "method_short": "TextVQA k=25%: is this Denny's"},
]

METHOD_ORDER = ["unpruned", "random", "papit_clip", "papit_distill"]
METHOD_LABEL = {
    "unpruned":      "Unpruned",
    "random":        "Random",
    "papit_clip":    "PAPIT-CLIP",
    "papit_distill": "PAPIT-Distill",
}


def build_predictor(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    arch = ckpt.get("arch", "mlp2")
    p, t = ckpt.get("patch_dim", 1024), ckpt.get("text_dim", 768)
    if arch == "mlp2" and "hidden" in ckpt:
        m = PatchScorePredictor(patch_dim=p, text_dim=t, hidden=ckpt["hidden"])
    else:
        m = ARCHS[arch](patch_dim=p, text_dim=t)
    m.load_state_dict(ckpt["model_state"])
    return m.to(device).eval()


def find_hero_row(csv_dir: Path, dataset: str, question_substr: str, retention: float):
    csv = pd.read_csv(csv_dir / f"distill100k_attn_L8_{dataset}.csv")
    csv = csv[csv.retention == retention]
    matches = csv[csv.question.str.contains(question_substr, case=False, regex=False)]
    if len(matches) == 0:
        raise RuntimeError(f"no example matching {question_substr!r} in {dataset}")
    return matches.iloc[0]


def fetch_hero_image(dataset: str, question_substr: str) -> Image.Image:
    """Re-download the hero example image directly from HF parquet by
    matching question substring. Avoids needing val_subset image paths."""
    import io, os
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HOME", "/opt/dlami/nvme/hf_cache")
    if dataset == "textvqa":
        pq = hf_hub_download("lmms-lab/textvqa",
            "data/validation-00000-of-00003.parquet", repo_type="dataset")
        df = pd.read_parquet(pq)
        m = df[df["question"].str.contains(question_substr, case=False, regex=False)]
        if len(m) == 0:
            raise RuntimeError(f"no textvqa match for {question_substr!r}")
        field = m.iloc[0]["image"]
    elif dataset == "gqa":
        # Question in instructions parquet, image bytes in images parquet,
        # joined on imageId.
        inst = hf_hub_download("lmms-lab/GQA",
            "testdev_balanced_instructions/testdev-00000-of-00001.parquet",
            repo_type="dataset")
        imgs = hf_hub_download("lmms-lab/GQA",
            "testdev_balanced_images/testdev-00000-of-00001.parquet",
            repo_type="dataset")
        dq = pd.read_parquet(inst)
        di = pd.read_parquet(imgs).drop_duplicates("id").set_index("id")
        m = dq[dq["question"].str.contains(question_substr, case=False, regex=False)]
        if len(m) == 0:
            raise RuntimeError(f"no gqa match for {question_substr!r}")
        iid = str(m.iloc[0]["imageId"])
        if iid not in di.index:
            raise RuntimeError(f"gqa imageId {iid} missing from images parquet")
        field = di.loc[iid, "image"]
    elif dataset == "vqa_v2":
        pq = hf_hub_download("lmms-lab/VQAv2",
            "data/validation-00000-of-00068.parquet", repo_type="dataset")
        df = pd.read_parquet(pq)
        m = df[df["question"].str.contains(question_substr, case=False, regex=False)]
        if len(m) == 0:
            raise RuntimeError(f"no vqav2 match for {question_substr!r}")
        field = m.iloc[0]["image"]
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    img_bytes = field["bytes"] if isinstance(field, dict) else field
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def patches_to_mask(top_k: torch.Tensor) -> np.ndarray:
    """Return [336, 336] mask, 1 where patches are kept, 0 where blacked."""
    mask = np.zeros(N_PATCHES, dtype=np.float32)
    mask[top_k.cpu().numpy()] = 1.0
    mask_2d = mask.reshape(GRID, GRID)
    return np.kron(mask_2d, np.ones((PATCH_PX, PATCH_PX), dtype=np.float32))


def render_panel(ax, image_pil: Image.Image, mask: np.ndarray | None,
                 label: str, answer: str, correct: bool):
    img = image_pil.resize((336, 336))
    img_arr = np.asarray(img).astype(np.float32) / 255.0
    if mask is not None:
        img_arr = img_arr * mask[..., None]
    ax.imshow(img_arr, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if label:
        ax.set_title(label, fontsize=7, pad=2)
    color = "#1A8043" if correct else "#C0392B"
    mark = "✓" if correct else "✗"
    ax.text(0.5, -0.04, f"{mark} {answer}", transform=ax.transAxes,
            ha="center", va="top", fontsize=7.5, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", type=Path, required=True)
    ap.add_argument("--csv-dir", type=Path,
                    default=Path("outputs"))
    ap.add_argument("--retention", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("report/fig_hero.pdf"))
    ap.add_argument("--llava-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    args = ap.parse_args()

    runner = PAPITLlavaRunner(
        llava_model_id=args.llava_id,
        config=PAPITConfig(retention_ratio=args.retention),
        attn_implementation="eager",
    )
    device = runner.device
    predictor = build_predictor(args.predictor, device)

    rows = [find_hero_row(args.csv_dir, h["dataset"], h["question_substr"],
                          h.get("retention", args.retention)) for h in HERO]

    # Single-row layout: 2 examples × 4 methods = 8 panels, with a
    # visible gap between the two example groups. Total height ~1.5 in,
    # which fits page-1 bottom of a CVPR 2-column letter page when the
    # paper text overflows to the next column / page.
    n_examples = len(rows)
    n_methods = len(METHOD_ORDER)
    n_cols = n_examples * n_methods
    # Width ratios: ones for normal panels, larger gap between examples
    # implemented by inserting a "gap" column (we'll hide its frame).
    fig, axes_full = plt.subplots(
        1, n_cols, figsize=(7.4, 1.55),
        gridspec_kw={"wspace": 0.02},
    )
    # axes_full has shape (n_cols,). Reshape into (n_examples, n_methods)
    # for indexing convenience.
    axes = np.array(axes_full).reshape(n_examples, n_methods)
    # Adjust subplots to leave a small gap between the two examples.
    fig.subplots_adjust(left=0.005, right=0.995, top=0.78, bottom=0.18)
    # Manually shift example 2 (cols 4..7) to the right by a small amount.
    GAP = 0.025
    for c in range(n_methods, n_cols):
        bb = axes_full[c].get_position()
        axes_full[c].set_position(
            [bb.x0 + GAP, bb.y0, bb.width, bb.height]
        )
    for c in range(n_methods):
        bb = axes_full[c].get_position()
        axes_full[c].set_position(
            [bb.x0 - GAP, bb.y0, bb.width, bb.height]
        )

    for r_idx, (row, hero) in enumerate(zip(rows, HERO)):
        # Pull the image fresh from the HF parquet by question match.
        img = fetch_hero_image(hero["dataset"], hero["question_substr"])
        text = runner._format_prompt(row["question"])
        inputs = runner.processor(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
        text_emb = runner._get_text_embedding(row["question"])

        s_clip = runner._gradcam_scores(inputs["pixel_values"], row["question"], 576)
        with torch.no_grad():
            s_distill = predictor(patch_for_proj.unsqueeze(0).float(),
                                  text_emb.unsqueeze(0).float())[0]
        ret = float(hero.get("retention", args.retention))
        k = int(round(576 * ret))
        seed = int(row["idx"])
        gen = torch.Generator(); gen.manual_seed(seed)
        rand_indices = torch.randperm(576, generator=gen)[:k]
        topk_clip    = torch.topk(s_clip,    k).indices
        topk_distill = torch.topk(s_distill, k).indices

        for c_idx, method in enumerate(METHOD_ORDER):
            ax = axes[r_idx, c_idx]
            if method == "unpruned":
                mask = None
            elif method == "random":
                mask = patches_to_mask(rand_indices)
            elif method == "papit_clip":
                mask = patches_to_mask(topk_clip)
            elif method == "papit_distill":
                mask = patches_to_mask(topk_distill)

            ans_col = "ans_unpruned" if method == "unpruned" else f"ans_{method.replace('papit_clip','papit').replace('papit_distill','distill')}"
            acc_col = "acc_unpruned" if method == "unpruned" else f"acc_{method.replace('papit_clip','papit').replace('papit_distill','distill')}"
            ans = str(row[ans_col]).split("ASSISTANT:")[-1].strip().rstrip(".") if "ASSISTANT:" in str(row[ans_col]) else str(row[ans_col]).strip().rstrip(".")
            ans = ans[:18]
            correct = float(row[acc_col]) > 0
            # Method label printed once per group (first row only)
            label = METHOD_LABEL[method]
            render_panel(ax, img, mask, label, ans, correct)

    # Centred caption above each example group, drawn after layout.
    fig.canvas.draw()
    for r_idx, (row, hero) in enumerate(zip(rows, HERO)):
        ax_first = axes[r_idx, 0]
        ax_last  = axes[r_idx, -1]
        b0 = ax_first.get_position(); b1 = ax_last.get_position()
        cx = 0.5 * (b0.x0 + b1.x1)
        cy = b0.y1 + 0.07
        question = row["question"]
        question_short = question[:46] + ("..." if len(question) > 46 else "")
        ret = float(hero.get("retention", 0.25))
        cap = (f'{hero["dataset"].upper()} $k\\!=\\!{int(ret*100)}\\%$   '
               f'Q: “{question_short}”')
        fig.text(cx, cy, cap, ha="center", va="bottom",
                 fontsize=8.5, fontweight="medium")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.06, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
