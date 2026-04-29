#!/usr/bin/env python3
"""Train a small MLP that predicts LLaVA's mid-layer attention from PAPIT's
ViT patch features and CLIP text embedding (Scope-A distillation).

Loss: KL divergence between softmax-over-576 of predicted scores and the
cached target attention distribution.

Usage:
    PYTHONPATH=. uv run python scripts/train_distill.py \
        --cache /opt/dlami/nvme/distill_cache \
        --out  /opt/dlami/nvme/distill_predictor.pt \
        --epochs 10 --batch 64
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PatchScorePredictor(nn.Module):
    """Per-patch score predictor:
        input  patch_hidden [B,576,P]  +  text_emb [B,T] (broadcast)
        output scores       [B,576]
    """

    def __init__(self, patch_dim: int = 1024, text_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, patches: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        p = self.patch_proj(patches.float())                           # [B,576,h]
        t = self.text_proj(text.float()).unsqueeze(1).expand_as(p)     # [B,576,h]
        return self.head(torch.cat([p, t], dim=-1)).squeeze(-1)        # [B,576]


def load_cache(cache_dir: Path):
    shard_files = sorted(cache_dir.glob("shard_*.pt"))
    if not shard_files:
        raise FileNotFoundError(f"No shards in {cache_dir}")
    parts = [torch.load(f, weights_only=True) for f in shard_files]
    vit    = torch.cat([p["vit"]    for p in parts], dim=0)
    text   = torch.cat([p["text"]   for p in parts], dim=0)
    target = torch.cat([p["target"] for p in parts], dim=0)
    print(f"Loaded {vit.shape[0]} samples from {len(shard_files)} shards "
          f"(vit={tuple(vit.shape)} {vit.dtype}, target={tuple(target.shape)})")
    return vit, text, target


def kl_loss(pred_logits: torch.Tensor, target_prob: torch.Tensor) -> torch.Tensor:
    """KL(target || softmax(pred)) over the 576-patch dimension."""
    log_pred = F.log_softmax(pred_logits, dim=-1)
    return F.kl_div(log_pred, target_prob, reduction="batchmean")


def topk_overlap(pred_logits: torch.Tensor, target: torch.Tensor, frac: float = 0.10) -> float:
    """Mean Jaccard between predicted and target top-k% sets."""
    k = max(1, int(round(pred_logits.shape[-1] * frac)))
    p_set = torch.topk(pred_logits, k, dim=-1).indices
    t_set = torch.topk(target,      k, dim=-1).indices
    js = []
    for i in range(p_set.shape[0]):
        a, b = set(p_set[i].tolist()), set(t_set[i].tolist())
        inter = len(a & b); union = len(a | b)
        js.append(inter / union if union else 0.0)
    return sum(js) / len(js)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out",   type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    vit, text, target = load_cache(args.cache)
    n = vit.shape[0]
    n_val = max(1, int(round(n * args.val_frac)))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"Split: train={len(tr_idx)}  val={len(val_idx)}")

    tr_ds  = TensorDataset(vit[tr_idx],  text[tr_idx],  target[tr_idx])
    val_ds = TensorDataset(vit[val_idx], text[val_idx], target[val_idx])
    tr_ld  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = PatchScorePredictor(
        patch_dim=vit.shape[-1], text_dim=text.shape[-1], hidden=args.hidden,
    ).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Predictor parameters: {n_param/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        losses = []
        for v, t, y in tr_ld:
            v, t, y = v.to(device), t.to(device), y.to(device)
            logits = model(v, t)
            loss = kl_loss(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            v_loss, v_jac = [], []
            for v, t, y in val_ld:
                v, t, y = v.to(device), t.to(device), y.to(device)
                logits = model(v, t)
                v_loss.append(kl_loss(logits, y).item())
                v_jac.append(topk_overlap(logits, y, frac=0.10))
        print(f"  epoch {ep+1:2d}: train_kl={sum(losses)/len(losses):.4f}  "
              f"val_kl={sum(v_loss)/len(v_loss):.4f}  "
              f"val_top10_jac={sum(v_jac)/len(v_jac):.3f}")

    torch.save({
        "model_state": model.state_dict(),
        "patch_dim": vit.shape[-1],
        "text_dim":  text.shape[-1],
        "hidden": args.hidden,
    }, args.out)
    print(f"\nSaved to {args.out}  (training took {(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
