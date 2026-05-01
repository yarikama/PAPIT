# Submission links — COMP 646 Final Project

**Authors:** Chao Hsuan Ho (ch218) · Heng Jui Hsu (hh83) · Kerstin Sun (ks256)
· Rice University

**Title:** PAPIT: Prompt-Aware Pruning for Image Tokens in Efficient Multimodal
Inference

## Deliverables

| Item | Link |
| ---- | ---- |
| Paper (4-page PDF) | [`report/PAPIT.pdf`](report/PAPIT.pdf) |
| Codebase | <https://github.com/yarikama/PAPIT> |
| 5-min walkthrough (YouTube, unlisted) | <https://youtu.be/LIKLcf7pN7E> |
| 5-min walkthrough (Google Drive mirror) | <https://drive.google.com/file/d/19uzhvj-3BEnP4Gv6LNXSwUOrvJJySTaI/view?usp=drive_link> |

## Headline result

PAPIT-Distill is a 1.3M-parameter MLP distilled from LLaVA-1.5-7B's mid-layer
attention. At *k* = 25 % retention on TextVQA it gains **+12.3** points over
random pruning (36.0 vs. 23.7), essentially recovering the unpruned baseline
(36.7), while scoring a 576-patch image in **0.43 ms** — ~150× faster than its
CLIP-GradCAM analogue and ~500× faster than the two-pass oracle it learns
from. See [`report/PAPIT.pdf`](report/PAPIT.pdf) for the full numbers and the
negative-result diagnostic that motivated distillation.
