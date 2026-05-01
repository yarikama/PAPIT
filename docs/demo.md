# 30-second AWS demo

Three copy-pasteable shell blocks that run PAPIT-Distill end-to-end on a
single TextVQA "Dakota camera" example, print the four method answers,
and save a 4-panel PNG. Designed for the 5-minute submission video.

## Prerequisites

- AWS instance is **running** (g5.2xlarge with the EBS volume that has
  `~/PAPIT/outputs/aws_artifacts/mlp4_attn_L8_20k.pt`).
- The `aws` SSH alias is configured locally.

If you need to start the instance, do it from the AWS console first.

## Step 1 — fetch a known-good test image (≈ 10 s)

Reuses [`scripts/fetch_demo_image.py`](../scripts/fetch_demo_image.py)
which pulls TextVQA val from the HF parquet by question substring:

```bash
ssh aws 'export PATH="$HOME/.local/bin:$PATH" HF_HOME=/opt/dlami/nvme/hf_cache &&
    cd ~/PAPIT && PYTHONPATH=. uv run python scripts/fetch_demo_image.py \
        --dataset textvqa --question "brand of this camera" \
        --out /tmp/dakota.jpg'
```

## Step 2 — run the demo (≈ 20 s on A10G; first run also downloads LLaVA-1.5-7B ~13 GB)

```bash
ssh aws 'export PATH="$HOME/.local/bin:$PATH" HF_HOME=/opt/dlami/nvme/hf_cache &&
    cd ~/PAPIT && PYTHONPATH=. uv run papit /tmp/dakota.jpg \
        "what is the brand of this camera?" \
        --retention 0.25 --generate \
        --predictor outputs/aws_artifacts/mlp4_attn_L8_20k.pt \
        --save-viz /tmp/demo.png --device cuda'
```

Expected stdout:

```text
Image:    /tmp/dakota.jpg
Question: what is the brand of this camera?
k:        25% (144/576 patches)

  unpruned       → Dakota digital
  random         → Delta
  papit_clip     → Nikon
  papit_distill  → Dakota digital
```

## Step 3 — pull the visualisation back (1 s)

```bash
scp aws:/tmp/demo.png . && open demo.png
```

`demo.png` is a 4-panel side-by-side: Unpruned / Random / PAPIT-CLIP /
PAPIT-Distill, each showing which 25% of the patches the method kept
and the answer LLaVA produced underneath.

## Why this case

At *k* = 25% the camera-brand text occupies a tiny fraction of the
image. PAPIT-CLIP discards the brand patch and LLaVA hallucinates
"Polaroid"; PAPIT-Distill, trained against LLaVA's mid-layer attention,
keeps it and answers "Dakota" — the same answer Unpruned gives at 4×
the FLOPs. Random pruning happens to land on no informative patches and
guesses "Canon". One example, four very different answers, in 30
seconds.

## Variations for the video

Different example, different *k*:

```bash
# GQA "leather bag side" at k=75%
ssh aws 'export PATH="$HOME/.local/bin:$PATH" HF_HOME=/opt/dlami/nvme/hf_cache &&
    cd ~/PAPIT && PYTHONPATH=. uv run python scripts/fetch_demo_image.py \
        --dataset gqa --question "side of the picture is the leather bag" \
        --out /tmp/bag.jpg'
ssh aws 'export PATH="$HOME/.local/bin:$PATH" HF_HOME=/opt/dlami/nvme/hf_cache &&
    cd ~/PAPIT && PYTHONPATH=. uv run papit /tmp/bag.jpg \
        "On which side of the picture is the leather bag?" \
        --retention 0.75 --generate \
        --predictor outputs/aws_artifacts/mlp4_attn_L8_20k.pt \
        --save-viz /tmp/demo_bag.png --device cuda'
```

Single-method (faster, ~5 s):

```bash
ssh aws '... uv run papit ... --generate --method papit_distill \
    --predictor ... --device cuda'
```

## Troubleshooting

- **`uv: command not found`** — the login shell didn't pick up
  `~/.local/bin`. Always export `PATH="$HOME/.local/bin:$PATH"` at the
  start of the ssh command (the blocks above already do this).
- **`bash: -c: option requires an argument`** — caused by trying to
  pass multi-line python to `bash -lc` over ssh. We avoid this by
  putting the fetch logic in a real script (`scripts/fetch_demo_image.py`)
  and using single-line ssh invocations.
- **`predictor not found`** — the EBS path is
  `outputs/aws_artifacts/mlp4_attn_L8_20k.pt`. The NVMe copy at
  `/opt/dlami/nvme/predictors_a2/` is volatile and will be gone after
  any instance stop.
