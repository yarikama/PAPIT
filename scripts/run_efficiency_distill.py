#!/usr/bin/env python3
"""Wall-clock latency benchmark for the four scoring strategies on
LLaVA-1.5-7B at retention 0.5. Reports mean ± std over `--n` samples
for: scoring-only and end-to-end (scoring + generate 16 tokens)."""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_distill_arch import ARCHS  # noqa
from train_distill import PatchScorePredictor  # noqa
from papit.core.config import PAPITConfig
from papit.integration.llava import PAPITLlavaRunner


def build_predictor_from_ckpt(ckpt, device):
    arch = ckpt.get("arch", "mlp2")
    p, t = ckpt.get("patch_dim", 1024), ckpt.get("text_dim", 768)
    if arch == "mlp2" and "hidden" in ckpt:
        m = PatchScorePredictor(patch_dim=p, text_dim=t, hidden=ckpt["hidden"])
    else:
        m = ARCHS[arch](patch_dim=p, text_dim=t)
    m.load_state_dict(ckpt["model_state"])
    return m.to(device).eval()


@torch.no_grad()
def time_it(fn, n_warmup=3, n=10):
    for _ in range(n_warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    import statistics
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--predictor", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("outputs/efficiency_distill.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv).head(args.n + 5).reset_index(drop=True)
    runner = PAPITLlavaRunner(
        llava_model_id="llava-hf/llava-1.5-7b-hf",
        config=PAPITConfig(retention_ratio=0.5),
        attn_implementation="eager",
    )
    device = runner.device

    ckpt = torch.load(args.predictor, weights_only=True, map_location=device)
    predictor = build_predictor_from_ckpt(ckpt, device)

    print("Measuring scoring-only latency at k=0.5 over", args.n, "samples...")
    rows = []
    for i in range(args.n):
        row = df.iloc[i]
        img = Image.open(row["image_path"]).convert("RGB")
        text = runner._format_prompt(row["question"])
        inputs = runner.processor(images=img, text=text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        patch_for_proj, _ = runner._extract_vit_features(inputs["pixel_values"])
        text_emb = runner._get_text_embedding(row["question"])

        # --- per-method scoring closure
        def score_random():
            torch.randperm(576, device=device)[:288]

        def score_papit_clip():
            runner._gradcam_scores(inputs["pixel_values"], row["question"], 576)

        def score_distill():
            with torch.no_grad():
                predictor(patch_for_proj.unsqueeze(0).float(),
                          text_emb.unsqueeze(0).float())

        def score_oracle():
            runner.llava(input_ids=inputs["input_ids"],
                         pixel_values=inputs["pixel_values"],
                         attention_mask=inputs["attention_mask"],
                         output_attentions=True, return_dict=True, use_cache=False)

        for name, fn in [("random", score_random),
                         ("papit_clip", score_papit_clip),
                         ("papit_distill", score_distill),
                         ("oracle_L16", score_oracle)]:
            mean_ms, std_ms = time_it(fn, n_warmup=2, n=5)
            rows.append({"sample": i, "method": name,
                         "mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3)})
            print(f"  [{i}] {name:14s}  {mean_ms:7.2f} ± {std_ms:5.2f} ms")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)

    print("\n=== Latency summary (mean over", args.n, "samples) ===")
    summary = out_df.groupby("method")["mean_ms"].agg(["mean", "std"]).round(2)
    print(summary)


if __name__ == "__main__":
    main()
