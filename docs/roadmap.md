# PAPIT — Roadmap to a Workshop-Submittable Paper

Snapshot date: 2026-04-29.
Author: this is a planning doc, not a status report. For per-experiment
results see `docs/experiment_log.md` and the per-CSV files in `outputs/`.

---

## 1. Where we are

| Item | Status |
|---|---|
| Pipeline (ViT → top-k → splice → LLaVA) | done, in `papit/integration/llava.py` |
| Original PAPIT result (700-sample × 3 datasets) | done; PAPIT ≈ Random |
| §7 Cross-dataset alignment diagnostic (ρ ≈ 0) | done, Table 4 in `final_report.tex` |
| §7 Multi-layer FastV-style oracle | done, Table 5 |
| Scope-A distillation MVP (5K cache, mlp2 @ L=16) | done, beats Random |
| Architecture ablation (bilinear / mlp2 / mlp4 / xattn) | done, mlp4 wins |
| **100K mixed cache (Scope-C Phase 1)** | running on AWS, ETA ~7 hr |
| **Phase 3+4 watcher** | armed; auto-launches when cache index.csv appears |

---

## 2. Remaining experiments

### Phase 3 (auto, ~20 min once cache ready)
Train `mlp4` on the 100K cache against each of 4 supervision targets:
`attn_L8`, `attn_L16`, `attn_L24`, `attn_rollout`. One predictor per
target → `predictors_100k/mlp4_<target>.pt`.

### Phase 4 (auto, ~30 min)
Eval each predictor on 100 samples per dataset
(`gqa_val_subset.csv`, `textvqa_val_subset.csv`, `vqa_v2_val_subset.csv`)
× 3 retentions, against {Unpruned, Random, PAPIT-CLIP, Oracle L=16}.
Result: 12 CSVs at `outputs/distill100k_<target>_<ds>.csv` plus a
`scope_c_p34.log` summary table.

### Phase 5 — Pick winner & full eval (~7 hr GPU after Phase 4)
1. Inspect Phase 4 summary, pick the supervision target whose downstream
   accuracy averaged across {3 datasets × 3 retentions} is highest.
2. Run a 700-sample-per-dataset eval with that single predictor against
   {Unpruned, Random, PAPIT-CLIP, Oracle L=16}.
3. This is the headline number for the paper. Reuses
   `scripts/eval_distill.py`; just bump `--max-samples` to 700.

### Phase 6 — Efficiency benchmark (~30 min GPU)
Wall-clock latency on g5.2xlarge for one image at retention 0.5:
- Random (no scoring; just `randperm`)
- PAPIT-CLIP (CLIP forward + GradCAM backward)
- PAPIT-distill (predictor MLP forward; ~1 ms expected)
- FastV-style oracle (full LLaVA forward)
Reuse `scripts/run_efficiency_benchmark.py`; add a `--scorer` flag if
needed. Average over ≥30 samples per scorer to get stable numbers.

### Phase 7 (optional, ~3 hr GPU, $4)
Cross-dataset training-data ablation: train 3 predictors on (a) GQA-only
50K, (b) VQAv2-only 30K, (c) full 100K mix. Eval on TextVQA-val subset.
Tests whether the mix actually helps generalisation.

### Phase 8 (optional, ~30 min GPU)
Predictor size ablation: re-train `mlp4` at hidden ∈ {128, 256, 384, 512}.
Plot params vs val Jaccard vs downstream accuracy. Argues that even a
smaller predictor (faster inference) is sufficient.

**Deliverables from §2:** 12 + 9 + 4 + (optional 9 + 16) ≈ 25–50 CSV
files, four predictor checkpoints, one efficiency CSV.

---

## 3. Paper restructuring

Current `final_report.tex` is framed as **negative-result + positive-control**.
Scope-C makes it a **method paper**. The minimal-edit path:

### Sections to keep verbatim
- §2 Related Work (already cites everything we need)
- §3 Methodology (PAPIT pipeline figure stays; this *is* the pipeline
  the distilled predictor plugs into)
- §4 Qualitative Results

### Sections to revise

| Section | Action |
|---|---|
| Abstract | Lead with the distillation result, not the negative result; keep negative result as "explains why CLIP-based pruners fail" |
| §1 Introduction | Three contributions: pipeline, negative result, distilled predictor |
| §5 Quantitative Results | Add a column for PAPIT-distill in the accuracy table; update Pareto figure to add a third curve |
| §6 Experiments | Add ablation pointer to §8 |
| §7 Alignment Diagnostic | Stays. Add 1–2 sentences about why the diagnostic motivates distillation |
| **§8 (new) Distillation Method** | Architecture, supervision target, training, hyperparams. Reuse the L=16 oracle figure as motivation |
| **§9 (new) Distillation Results** | Tables for: (i) supervision-target ablation, (ii) full 700-sample × 3-dataset accuracy, (iii) efficiency benchmark including PAPIT-distill |
| §10 Discussion | Reframe: misalignment was the problem, distillation is the fix; what's next |
| Conclusion | Three-contribution close: pipeline + negative result + distillation |

### Tables (final layout target)

1. **Efficiency** — keep as is, add a `PAPIT-distill (predictor)` row
2. **Main accuracy** — add PAPIT-distill column
3. **Scoring ablation (existing)** — keep; expand caption to note these
   are all *CLIP-aligned* signals, hence the parity with random
4. **Alignment diagnostic** — keep
5. **Multi-layer oracle** — keep
6. **(new) Supervision target ablation** — `mlp4` × {L8/L16/L24/rollout}
   × 3 retentions on GQA-100, val_kl + Jaccard + downstream
7. **(new) Architecture ablation** — bilinear/mlp2/mlp4/xattn × val
   metrics

### Figures
- Pipeline figure: stays
- Qualitative figure: stays
- Pareto figure: update to include PAPIT-distill curve (a 3rd line)
- (optional) New figure: per-target val_kl over training epochs, on
  the 100K cache; demonstrates the four targets converge to different
  Jaccard plateaus

---

## 4. Page-budget sketch

| Venue | Page limit | Approach |
|---|---|---|
| **CVPR/ICCV workshop** | 4 + refs | Hard cut: drop §6 ablation, drop multi-layer oracle table (cite ours from supplementary), keep new §8 + §9 |
| **NeurIPS workshop** | 4–6 + refs | Same as above; can keep the multi-layer oracle table as Table 4 |
| **EMNLP findings / short** | 4 + refs | Possible if we re-frame as "training-free scoring is misaligned; we fix it with cheap distillation"; needs a more language-model-y intro |
| **arXiv only** | unlimited | Default. 8–10 pages including all ablations |

Recommend writing the 8–10 page arXiv version first, then producing a
4-page workshop cut from it (compress §3 and §6, drop §6 ablation table).

---

## 5. Risks and contingencies

| Risk | Likelihood | Mitigation |
|---|---|---|
| 100K cache fails partway | low | Watcher only triggers on `index.csv` which exists only on success; partial shards reusable for retraining |
| Phase 4 winner target underperforms Scope-A 5K mlp2 | low | Phase 4 result CSVs let us inspect; fall back to mlp2 + L16 from Scope-A if so. (Scope-A 5K beat random, so this is a soft floor.) |
| Distill barely beats random in Phase 5 (e.g. +0.01) | medium | Honest framing: "small but consistent gain explained by the alignment-diagnostic predictor". Strengthens the *empirical* case for distillation. Paper still publishable as a careful study. |
| AWS instance cost overrun | medium | Stop instance once Phase 6 done; total realistic spend $50–80 |
| HF dataset rate-limit during prep | low | Token already configured; if hit, retry next hour |
| Cross-dataset (TextVQA) generalisation fails | medium | Honest reporting: predictor may need TextVQA-only fine-tune (Phase 7 covers this) |

---

## 6. Submission timeline (suggested)

Assuming today is 2026-04-29 and Scope C cache + Phase 3+4 finishes by
2026-04-30 morning:

| Week | Focus |
|---|---|
| W1 (now → 05-04) | Phases 5+6, write §8 §9, update tables/figures |
| W2 (05-05 → 05-11) | Optional Phases 7+8 if results suggest; tighten Discussion + Conclusion |
| W3 (05-12 → 05-18) | Internal review pass, ablation polish, arXiv-ready 8-page draft |
| W4 (05-19 →) | 4-page workshop cut + camera-ready prep when venue decided |

---

## 7. Decision points (require user input when reached)

1. **After Phase 4 summary** — confirm winner supervision target before
   committing 7 hr GPU to Phase 5.
2. **After Phase 5** — decide Phase 7 / Phase 8 yes/no based on whether
   the headline numbers need additional ablation muscle.
3. **After draft** — pick venue (CVPR-W vs NeurIPS-W vs EMNLP-findings)
   so we cut the right pages.
4. **AWS lifecycle** — after Phase 6 done, stop or terminate the
   instance. Stop preserves EBS root (~$1.8/mo) but loses NVMe
   ephemeral data; terminate is final.

---

## 8. What is intentionally *not* in this plan

- New baselines beyond what's needed for the negative-and-positive story
  (e.g. SparseVLM, VisionZip head-to-head). These are good for a
  full-conference paper, not a workshop submission.
- Cross-model validation (LLaVA-1.6, Qwen-VL, InstructBLIP). One model
  is enough for the workshop story. Add later if extending to a full
  conference paper.
- Joint-training a predictor inside the LLaVA forward (single-pass
  FastV-style). Keeps the paper focused on the two-pass distillation
  story; single-pass is follow-up work.

---

## 9. Files of record

| File | Role |
|---|---|
| `report/final_report.tex` | the paper draft |
| `docs/experiment_log.md` | historical scoring-method log |
| `outputs/alignment_*.csv` | per-sample ρ + Jaccard for §7 |
| `outputs/fastv_oracle_multilayer.csv` | per-sample multi-layer oracle |
| `outputs/distill_eval_gqa.csv` | Scope-A 100-sample eval |
| `outputs/distill100k_<target>_<ds>.csv` | (Phase 4) per-dataset eval per target |
| `outputs/distill700_<ds>.csv` | (Phase 5, planned) 700-sample headline |
| `outputs/efficiency_with_distill.csv` | (Phase 6, planned) wall-clock |

---

## 10. North star

The paper's core claim is a 3-step empirical chain:

1. **CLIP-aligned saliency carries no useful patch-importance signal**
   for LLaVA — measured directly (Spearman ρ ≈ 0, top-10% Jaccard below
   random) on 3 benchmarks.
2. **LLM-internal mid-layer attention does carry the signal** — the
   FastV-style two-pass oracle beats random.
3. **A 1-million-parameter MLP can distil that signal cheaply enough to
   deploy** — `PAPIT-distill` matches the oracle within noise on 700
   samples × 3 datasets, at single-pass inference cost.

Every section, table, and figure in the paper should be evaluable
against "does this advance one of those three claims?" If not, cut.
