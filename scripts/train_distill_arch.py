#!/usr/bin/env python3
"""Train multiple predictor architectures on a cache; report val metrics.

Architectures:
  - bilinear : score_i = (W_text @ text)·(W_patch @ patch_i). ~1M params.
  - mlp2     : per-patch + broadcast text -> 2-layer MLP. ~0.6M (Scope-A baseline).
  - mlp4     : per-patch + broadcast text -> 4-layer MLP, residual. ~2M.
  - xattn    : text vector cross-attends to patches, then per-patch MLP. ~3M.

Usage:
    PYTHONPATH=. uv run python scripts/train_distill_arch.py \
        --cache /opt/dlami/nvme/distill_cache \
        --target attn_L16  --epochs 10 --batch 64
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

class BilinearScorer(nn.Module):
    def __init__(self, patch_dim=1024, text_dim=768, hidden=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden)
        self.patch_proj = nn.Linear(patch_dim, hidden)
    def forward(self, patches, text):
        p = self.patch_proj(patches.float())              # [B,576,h]
        t = self.text_proj(text.float())                  # [B,h]
        return torch.einsum("bnh,bh->bn", p, t)           # [B,576]


class MLP2Scorer(nn.Module):  # baseline (Scope-A)
    def __init__(self, patch_dim=1024, text_dim=768, hidden=256):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, hidden)
        self.text_proj  = nn.Linear(text_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, patches, text):
        p = self.patch_proj(patches.float())
        t = self.text_proj(text.float()).unsqueeze(1).expand_as(p)
        return self.head(torch.cat([p, t], dim=-1)).squeeze(-1)


class MLP4Scorer(nn.Module):
    def __init__(self, patch_dim=1024, text_dim=768, hidden=384):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, hidden)
        self.text_proj  = nn.Linear(text_dim, hidden)
        self.b1 = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, hidden), nn.GELU())
        self.b2 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.b3 = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 1)
    def forward(self, patches, text):
        p = self.patch_proj(patches.float())
        t = self.text_proj(text.float()).unsqueeze(1).expand_as(p)
        h = self.b1(torch.cat([p, t], dim=-1))
        h = h + self.b2(h)
        h = h + self.b3(h)
        return self.out(h).squeeze(-1)


class XAttnScorer(nn.Module):
    """Text vector serves as a single query; patches as keys/values; one attn
    block + per-patch MLP head over (patch_proj || patch_attended).
    """
    def __init__(self, patch_dim=1024, text_dim=768, hidden=384, n_heads=4):
        super().__init__()
        self.patch_proj = nn.Linear(patch_dim, hidden)
        self.text_proj  = nn.Linear(text_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, patches, text):
        p = self.patch_proj(patches.float())                 # [B,576,h]
        t = self.text_proj(text.float()).unsqueeze(1)        # [B,1,h]
        # Patches attend to text -> per-patch attention-conditioned vector
        out, _ = self.attn(p, t, t)                          # [B,576,h]
        return self.head(torch.cat([p, out], dim=-1)).squeeze(-1)


ARCHS = {
    "bilinear": BilinearScorer,
    "mlp2": MLP2Scorer,
    "mlp4": MLP4Scorer,
    "xattn": XAttnScorer,
}


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------

def kl_loss(pred_logits, target_prob):
    log_pred = F.log_softmax(pred_logits, dim=-1)
    return F.kl_div(log_pred, target_prob, reduction="batchmean")


def top10_jaccard(pred_logits, target):
    k = max(1, int(round(pred_logits.shape[-1] * 0.10)))
    p = torch.topk(pred_logits, k, dim=-1).indices
    t = torch.topk(target,      k, dim=-1).indices
    js = []
    for i in range(p.shape[0]):
        a, b = set(p[i].tolist()), set(t[i].tolist())
        u = len(a | b)
        js.append(len(a & b) / u if u else 0.0)
    return sum(js) / len(js)


# ---------------------------------------------------------------------------
# Cache loader (handles both Scope-A and multi-target shards)
# ---------------------------------------------------------------------------

def load_cache(cache_dir: Path, target_key: str):
    shards = sorted(cache_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards in {cache_dir}")
    parts = [torch.load(f, weights_only=True) for f in shards]
    vit  = torch.cat([p["vit"]  for p in parts], dim=0)
    text = torch.cat([p["text"] for p in parts], dim=0)
    if target_key in parts[0]:
        target = torch.cat([p[target_key] for p in parts], dim=0)
    elif "target" in parts[0]:
        # Scope-A cache stored a single 'target' tensor (= L=16 attention).
        if target_key not in ("attn_L16", "target"):
            raise KeyError(f"target_key={target_key} not found; cache only has 'target'.")
        target = torch.cat([p["target"] for p in parts], dim=0)
    else:
        raise KeyError(f"Cache missing target {target_key}; available: {list(parts[0].keys())}")
    return vit, text, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--target", type=str, default="attn_L16",
                    choices=["attn_L8", "attn_L16", "attn_L24", "attn_rollout", "attn_answer"])
    ap.add_argument("--archs", nargs="+", default=["bilinear", "mlp2", "mlp4", "xattn"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch",  type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-best", type=Path, default=None,
                    help="Save the lowest-val-KL model checkpoint here.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    vit, text, target = load_cache(args.cache, args.target)
    n = vit.shape[0]
    n_val = max(1, int(round(n * args.val_frac)))
    perm = torch.randperm(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr_ld  = DataLoader(TensorDataset(vit[tr_idx],  text[tr_idx],  target[tr_idx]),
                        batch_size=args.batch, shuffle=True)
    val_ld = DataLoader(TensorDataset(vit[val_idx], text[val_idx], target[val_idx]),
                        batch_size=args.batch, shuffle=False)
    print(f"Cache n={n}  target={args.target}  train={len(tr_idx)}  val={len(val_idx)}")

    rows = []
    best_overall = (None, float("inf"), None)  # (arch, val_kl, state_dict)

    for arch_name in args.archs:
        Arch = ARCHS[arch_name]
        model = Arch(patch_dim=vit.shape[-1], text_dim=text.shape[-1]).to(device)
        n_p = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        t0 = time.time()
        last_val_kl, last_jac = None, None
        best_state, best_kl = None, float("inf")
        for ep in range(args.epochs):
            model.train()
            for v, t_, y in tr_ld:
                v, t_, y = v.to(device), t_.to(device), y.to(device)
                loss = kl_loss(model(v, t_), y)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vk, vj = [], []
                for v, t_, y in val_ld:
                    v, t_, y = v.to(device), t_.to(device), y.to(device)
                    logits = model(v, t_)
                    vk.append(kl_loss(logits, y).item())
                    vj.append(top10_jaccard(logits, y))
                last_val_kl = sum(vk) / len(vk)
                last_jac    = sum(vj) / len(vj)
            if last_val_kl < best_kl:
                best_kl = last_val_kl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elapsed = (time.time() - t0)
        print(f"  {arch_name:<10} params={n_p/1e6:5.2f}M  "
              f"val_kl={last_val_kl:.4f}  best_kl={best_kl:.4f}  "
              f"top10_jac={last_jac:.3f}  ({elapsed:.0f}s)")
        rows.append({
            "arch": arch_name, "params_M": n_p/1e6,
            "val_kl_last": last_val_kl, "val_kl_best": best_kl,
            "val_top10_jac": last_jac, "secs": elapsed,
        })
        if best_kl < best_overall[1]:
            best_overall = (arch_name, best_kl, best_state)

    print("\n=== summary ===")
    import pandas as pd
    print(pd.DataFrame(rows).to_string(index=False))

    if args.save_best is not None and best_overall[2] is not None:
        # Save best-overall checkpoint
        Arch = ARCHS[best_overall[0]]
        # Recover dims
        torch.save({
            "arch": best_overall[0],
            "patch_dim": vit.shape[-1],
            "text_dim":  text.shape[-1],
            "model_state": best_overall[2],
            "target": args.target,
            "val_kl_best": best_overall[1],
        }, args.save_best)
        print(f"\nSaved best ({best_overall[0]}, val_kl={best_overall[1]:.4f}) to {args.save_best}")


if __name__ == "__main__":
    main()
