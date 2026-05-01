# 30-second AWS demo

A copy-pasteable script that runs PAPIT-Distill end-to-end on a single
TextVQA "Dakota camera" example, prints the four method answers, and
saves a 4-panel comparison PNG. Designed for the 5-minute video.

## Prerequisites

- AWS instance is **running** (g5.2xlarge with the EBS volume that has
  `~/PAPIT/outputs/aws_artifacts/mlp4_attn_L8_20k.pt`).
- `aws` SSH alias is configured locally.

If you need to start the instance, do it from the AWS console first.

## Step 1 — fetch a known-good test image (10 s)

The hero figure pipeline already pulls TextVQA images from HF parquet.
We reuse that to grab the Dakota camera frame on the fly:

```bash
ssh aws bash -lc '
cd ~/PAPIT && export HF_HOME=/opt/dlami/nvme/hf_cache &&
PYTHONPATH=. uv run python -c "
import io
from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image
pq = hf_hub_download(\"lmms-lab/textvqa\",
    \"data/validation-00000-of-00003.parquet\", repo_type=\"dataset\")
df = pd.read_parquet(pq)
m = df[df.question.str.contains(\"brand of this camera\", case=False)]
field = m.iloc[0][\"image\"]
buf = field[\"bytes\"] if isinstance(field, dict) else field
Image.open(io.BytesIO(buf)).convert(\"RGB\").save(\"/tmp/dakota.jpg\")
print(\"saved /tmp/dakota.jpg\")
"
'
```

## Step 2 — run the demo (≈ 20 s on A10G)

```bash
ssh aws bash -lc '
cd ~/PAPIT && export PATH="$HOME/.local/bin:$PATH" &&
export HF_HOME=/opt/dlami/nvme/hf_cache &&
PYTHONPATH=. uv run papit /tmp/dakota.jpg "what is the brand of this camera?" \
    --retention 0.25 \
    --generate \
    --predictor ~/PAPIT/outputs/aws_artifacts/mlp4_attn_L8_20k.pt \
    --save-viz /tmp/demo.png \
    --device cuda
'
```

Expected stdout:

```text
Image:    /tmp/dakota.jpg
Question: what is the brand of this camera?
k:        25% (144/576 patches)

  unpruned       → Dakota
  random         → Canon
  papit_clip     → Polaroid
  papit_distill  → Dakota
```

## Step 3 — pull the visualisation back (1 s)

```bash
scp aws:/tmp/demo.png .
open demo.png
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
ssh aws "cd ~/PAPIT && PYTHONPATH=. uv run papit /tmp/bag.jpg \
    'On which side of the picture is the leather bag?' \
    --retention 0.75 --generate \
    --predictor outputs/aws_artifacts/mlp4_attn_L8_20k.pt \
    --save-viz /tmp/demo_bag.png --device cuda"
```

Single-method (faster, ~5 s):

```bash
papit ... --generate --method papit_distill --predictor ... --device cuda
```
