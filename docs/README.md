# PAPIT documentation

Index for everything that isn't the paper or runnable code.

| File | What it is |
| ---- | ---------- |
| [`demo.md`](demo.md) | 30-second AWS demo recipe — three copy-pasteable shell blocks that produce the 4-panel comparison PNG used in the submission video. |
| [`scope_c_postmortem.md`](scope_c_postmortem.md) | Engineering journal for the Scope-C distillation work: what was shipped, headline numbers, and the six mistakes worth not repeating (cache sizing, OOM diagnosis, instance-store volatility, etc.). |
| [`experiment_log.md`](experiment_log.md) | Older log of scoring-method experiments preceding Scope C. Kept for context; superseded by `scope_c_postmortem.md` for the final-report numbers. |
| [`roadmap.md`](roadmap.md) | Pre-paper roadmap mapping each rubric item to a concrete artifact. Useful as a checklist of what made it into the paper vs. what was descoped. |
| [`proposal/`](proposal/) | Original NeurIPS-format project proposal (`proposal.tex` / `.pdf`) and its style file. The starting point for the paper; included for traceability. |

The paper itself lives at [`../report/PAPIT.pdf`](../report/PAPIT.pdf).
The deployed predictor (5 MB MLP₄) is checked in at
[`../outputs/mlp4_attn_L8_20k.pt`](../outputs/mlp4_attn_L8_20k.pt).
