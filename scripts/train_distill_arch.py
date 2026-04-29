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
# Sharded dataset for big caches (100K samples × 1.18MB = 118GB doesn't fit
# in g5.2xlarge's 32GB RAM, so we keep at most a few shards loaded at once).
# Falls back to in-memory tensors when the cache is small.
# ---------------------------------------------------------------------------
from torch.utils.data import Dataset


import random as _random
from torch.utils.data import IterableDataset


class ShardedCacheDataset(IterableDataset):
    """Worker-aware streaming sharded cache.
    Each DataLoader worker iterates a disjoint subset of shards. With
    `num_workers >= 2` and `prefetch_factor >= 2`, the next shard is
    loaded in parallel with the current shard's training, hiding the
    ~200 ms torch.load latency behind the ~100 ms compute.
    Call `epoch_shuffle()` on the main-process dataset before each epoch
    to randomise shard visit order; workers see the updated state via
    fork-on-iter (default DataLoader behaviour)."""

    def __init__(self, cache_dir: Path, target_key: str, indices):
        self.cache_dir = cache_dir
        self.target_key = target_key
        self.shards = sorted(cache_dir.glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No shards in {cache_dir}")
        first = torch.load(self.shards[0], weights_only=True)
        if target_key not in first and "target" in first:
            if target_key not in ("attn_L16", "target"):
                raise KeyError(
                    f"target_key={target_key} not in cache; only 'target' available")
            self._target_alias = "target"
        else:
            self._target_alias = target_key
        self._per_shard = first["vit"].shape[0]
        self.patch_dim = first["vit"].shape[-1]
        self.text_dim = first["text"].shape[-1]
        del first
        self._by_shard: dict[int, list[int]] = {}
        for gi in indices:
            sh = gi // self._per_shard
            self._by_shard.setdefault(sh, []).append(gi % self._per_shard)
        self.epoch_shuffle()

    def epoch_shuffle(self):
        order = list(self._by_shard.keys())
        _random.shuffle(order)
        self._shard_order = order

    def __len__(self):
        return sum(len(v) for v in self._by_shard.values())

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        order = self._shard_order
        if info is None:
            my_shards = order
        else:
            # Round-robin shard assignment so workers progress in parallel.
            my_shards = order[info.id::info.num_workers]
        for sh in my_shards:
            d = torch.load(self.shards[sh], weights_only=True)
            in_shard_indices = self._by_shard[sh][:]
            _random.shuffle(in_shard_indices)
            for in_shard in in_shard_indices:
                yield (d["vit"][in_shard],
                       d["text"][in_shard],
                       d[self._target_alias][in_shard])
            del d


def cache_total_size(cache_dir: Path):
    shards = sorted(cache_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards in {cache_dir}")
    first = torch.load(shards[0], weights_only=True)
    per = first["vit"].shape[0]
    last = torch.load(shards[-1], weights_only=True)["vit"].shape[0]
    return per * (len(shards) - 1) + last


def load_cache(cache_dir: Path, target_key: str):
    """Legacy in-memory loader. Used only by Scope-A 5K cache path."""
    shards = sorted(cache_dir.glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No shards in {cache_dir}")
    parts = [torch.load(f, weights_only=True) for f in shards]
    vit  = torch.cat([p["vit"]  for p in parts], dim=0)
    text = torch.cat([p["text"] for p in parts], dim=0)
    if target_key in parts[0]:
        target = torch.cat([p[target_key] for p in parts], dim=0)
    elif "target" in parts[0]:
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
    ap.add_argument("--in-memory-threshold", type=int, default=20000,
                    help="Below this many samples, load all of cache into "
                         "RAM (faster). Above, stream from disk.")
    ap.add_argument("--shard-cache", type=int, default=8,
                    help="Number of shards kept in RAM by the streaming loader.")
    ap.add_argument("--subset", type=int, default=None,
                    help="Use only this many samples from the cache (random "
                         "subset). 30K fits in OS file cache so subsequent "
                         "epochs are RAM-bound rather than disk-bound.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    n_avail = cache_total_size(args.cache)
    n_total = min(args.subset, n_avail) if args.subset else n_avail
    n_val = max(1, int(round(n_total * args.val_frac)))
    # Use the first n_total samples (= first ceil(n_total/100) shards) so the
    # streaming loader only touches those shards. The cache itself was
    # randomly shuffled when built, so contiguous-prefix == random subset.
    perm_full = torch.randperm(n_total)             # shuffle within subset
    val_idx_all = perm_full[:n_val].tolist()
    tr_idx_all  = perm_full[n_val:].tolist()

    # Stream when the underlying cache is too big for RAM, even if the
    # subset we're using would fit — load_cache() reads the full cache.
    streaming = n_avail > args.in_memory_threshold
    if not streaming:
        # Small cache (Scope-A 5K): keep the original fast in-memory path.
        vit, text, target = load_cache(args.cache, args.target)
        tr_ds  = TensorDataset(vit[torch.tensor(tr_idx_all)],
                               text[torch.tensor(tr_idx_all)],
                               target[torch.tensor(tr_idx_all)])
        val_ds = TensorDataset(vit[torch.tensor(val_idx_all)],
                               text[torch.tensor(val_idx_all)],
                               target[torch.tensor(val_idx_all)])
        patch_dim, text_dim = vit.shape[-1], text.shape[-1]
        del vit, text, target
    else:
        # Big cache (100K): stream from disk; only shards whose samples
        # appear in tr_idx_all/val_idx_all are loaded.
        tr_ds  = ShardedCacheDataset(args.cache, args.target, tr_idx_all)
        val_ds = ShardedCacheDataset(args.cache, args.target, val_idx_all)
        patch_dim, text_dim = tr_ds.patch_dim, tr_ds.text_dim

    if streaming:
        tr_ld  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=False,
                            num_workers=2, prefetch_factor=4,
                            persistent_workers=False)
        val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=1, prefetch_factor=2,
                            persistent_workers=False)
    else:
        tr_ld  = DataLoader(tr_ds,  batch_size=args.batch, shuffle=True,
                            num_workers=0)
        val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0)
    print(f"Cache n={n_total}  target={args.target}  "
          f"train={len(tr_idx_all)}  val={len(val_idx_all)}  "
          f"streaming={n_total > args.in_memory_threshold}")

    rows = []
    best_overall = (None, float("inf"), None)  # (arch, val_kl, state_dict)

    for arch_name in args.archs:
        Arch = ARCHS[arch_name]
        model = Arch(patch_dim=patch_dim, text_dim=text_dim).to(device)
        n_p = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        t0 = time.time()
        last_val_kl, last_jac = None, None
        best_state, best_kl = None, float("inf")
        for ep in range(args.epochs):
            if streaming:
                tr_ds.epoch_shuffle()
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
        torch.save({
            "arch": best_overall[0],
            "patch_dim": patch_dim,
            "text_dim":  text_dim,
            "model_state": best_overall[2],
            "target": args.target,
            "val_kl_best": best_overall[1],
        }, args.save_best)
        print(f"\nSaved best ({best_overall[0]}, val_kl={best_overall[1]:.4f}) to {args.save_best}")


if __name__ == "__main__":
    main()
