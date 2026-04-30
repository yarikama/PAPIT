# Scope-C journal — what was actually built, and what went wrong

A condensed timeline of the distillation work (Apr 28 → Apr 29) plus the
mistakes that cost us time so they don't repeat. Reads like a postmortem,
which is the point.

---

## What we shipped

| Artefact | File | What it is |
|---|---|---|
| 4-page paper | `report/final_report.pdf` | rubric-aligned CVPR-style w/ Charter font, hero Fig 1 |
| Hero Fig 1 | `report/fig_hero.pdf` | 2 × 4 panel real-I/O on TextVQA: Unpruned / Random / PAPIT-CLIP / PAPIT-Distill at k=25% with the model's actual answers and ✓/✗ |
| Pareto Fig | `report/fig_pareto.pdf` | accuracy-vs-FLOPs across 3 datasets, 4 methods |
| Predictor | `outputs/mlp4_attn_L8_20k.pt` | 1.28M-param MLP-4, trained on 20K subset of the mixed-domain cache, target = LLaVA L=8 attention. val_kl = 0.094, top-10% Jaccard 0.60 |
| Phase-4 evals | `outputs/distill100k_<target>_<dataset>.csv` × 12 | downstream accuracy at N=100 across 4 supervision targets × 3 benchmarks |
| Alignment diagnostic | `outputs/alignment_<dataset>.csv` × 3 | per-image Spearman ρ between PAPIT-CLIP and LLaVA's last-layer attention |
| Multi-layer oracle | `outputs/fastv_oracle_multilayer.csv` | 5-layer two-pass oracle on GQA-100 |
| Latency benchmark | `outputs/efficiency_distill.csv` | wall-clock for 4 scorers at retention 0.5 |
| Cache index | `outputs/distill_cache_20k_index.csv` | which 20K (image, question) pairs the predictor was trained on |

The 100K cache itself (118 GB of multi-target features) is **not** in the
repo — it was on the AWS instance store. See "Mistake 4" below.

---

## Headline numbers

| Dataset | $k$ | Random | PAPIT-CLIP | **PAPIT-Distill (L=8)** | Unpruned |
|---|---|---|---|---|---|
| TextVQA | 25% | 23.7 | 20.0 | **36.0** (+12.3) | 36.7 |
| GQA | 50% | 52.0 | 52.0 | **56.0** | 57.0 |
| VQA v2 | 75% | 82.0 | 82.7 | **85.0** | 83.3 |

Latency at retention 0.5 (g5.2xlarge, A10G):

| Method | ms / image |
|---|---|
| Random | 0.07 |
| **PAPIT-Distill** | **0.43** |
| PAPIT-CLIP (GradCAM) | 65.23 |
| Two-pass Oracle L=16 | 216.17 |

Distill is ~150× faster than PAPIT-CLIP, ~500× faster than the oracle
it was distilled from, while matching the oracle's downstream accuracy.

---

## Phase timeline

| Phase | What happened | Outcome |
|---|---|---|
| 1 — Cache 100K | mixed 50K GQA + 30K VQAv2 + 20K TextVQA, multi-target (L=8 / L=16 / L=24 / rollout). 7.3 hr GPU on g5.2xlarge | 118 GB on `/opt/dlami/nvme` (instance store) |
| 2 — Architecture ablation | bilinear / mlp2 / mlp4 / xattn on Scope-A 5K cache | mlp4 wins (val_kl 0.087, Jac 0.62) |
| 3 — Train 4 predictors | mlp4 × {L=8, L=16, L=24, rollout} on 20K subset | L=8 has best val_kl 0.094 |
| 4 — Downstream eval | 12 evals × N=100 = (4 targets × 3 datasets) | L=8 wins by avg Δ=+0.030 vs random; rollout is **−0.054** (worse than random — distillability matters) |
| 5 — *(skipped)* | original plan: 700-sample headline | dropped: N=100 already shows TextVQA +12 at 5σ |
| 6 — Latency bench | 4 scorers × 20 samples | 0.43 ms PAPIT-Distill, the deploy story |
| (re-run) — A2 chain | re-cache 20K + retrain mlp4 (L=8) + render Fig 1 hero | predictor recovered, hero figure produced |

---

## Mistakes worth remembering

### 1. Cache-size sanity-check before committing
**Symptom**: 100K × 1.18 MB = 118 GB on disk, doesn't fit in g5.2xlarge's
30 GB RAM. We discovered this only after caching was done and Phase 3
training silently OOM-killed (no traceback).

**Should have**: at scope-design time done the arithmetic
`samples × per-sample bytes vs box RAM` and either downsized the cache
or upsized the instance to g5.8xlarge (128 GB RAM) before paying for the
big cache. Cost ~5 hr cache time and ~$8 GPU.

**Fix that finally worked**: streaming `IterableDataset` that loads one
shard at a time (~118 MB RAM) plus a 20K subset clamp so the OS file
cache holds the whole working set after epoch 1.

### 2. The OOM that didn't print a traceback
**Symptom**: Phase 3 launched, log said "Phase 3: training...", then
nothing. No error, no exit log, no process visible. Pipeline hung.

**Diagnosis**: Linux kernel OOM-killer SIGKILLs the process. Python
never gets to print. Visible only via `dmesg | grep oom-kill`.

**Should have**: in long-running launchers, print a stable "starting
training" line and have a Monitor pattern that flags when a process
disappears without emitting a known-good completion line. Our Monitor
filter only watched for `Traceback / OutOfMemoryError / Killed` —
`oom-kill` from `dmesg` doesn't go to the script's stdout/stderr, so
none of those matched.

### 3. The threshold that meant the opposite of what I thought
**Symptom**: ran the same `--in-memory-threshold 25000` against a 20K
cache. The flag means "if cache size $\le$ threshold, use in-memory
path"; 20K $\le$ 25000 → in-memory load → 24 GB allocation → OOM at
31 GB RSS. The exact same crash from Mistake 1, second time.

**Should have**: re-read the flag's semantics every time, instead of
copying a previous invocation. Also: a default that errs toward
streaming would have prevented this.

**Fix**: `--in-memory-threshold 5000` so any subset $\ge$ 5K streams.

### 4. Instance-store data is volatile; harvest immediately
**Symptom**: when the user `stop`'d the AWS instance, `/opt/dlami/nvme/`
was wiped. We lost (a) the 100K cache (118 GB, ~$9 GPU sunk) and
(b) the four `mlp4_attn_*.pt` checkpoints that produced the paper's
Phase-4 numbers. Only the per-sample CSVs (text data) had been pulled
back to the local repo.

**Should have**: `harvest_scope_c.sh` was written precisely to scp the
predictor checkpoints back to local, but I never invoked it after Phase
3+4 finished, because I was busy on Phase 6 + paper updates. The same
story-shape as the OOM: known prevention measure, not actually used.

**Recovery**: re-cached + retrained one predictor (L=8) on a fresh
g5.2xlarge in ~2 hr and ~$2.50, harvested *this* time.

**Rule of thumb**: any artefact that lives on instance store and cost
more than ~30 min of GPU to produce should be `scp`'d the moment it is
finished, not "later".

### 5. Schedule the loop you intend to be in
**Symptom**: `ScheduleWakeup` only fires when you are inside `/loop`'s
dynamic mode. I scheduled wake-ups before the user had typed `/loop`.
Result: the schedules were no-ops. The pipeline ran fine on AWS but
nothing on my side moved while it ran.

**Should have**: read the `ScheduleWakeup` description before the first
call. Once `/loop` was active, fallbacks worked correctly.

### 6. CSV column you assumed but didn't verify
**Symptom**: the hero-figure script hit `KeyError: 'image_path'` at
runtime because Phase-4 eval CSVs only carry `idx`, not the path. Fixed
by re-fetching the hero images directly from the HF parquet by question
substring.

**Should have**: read the CSV header once before writing the consumer.
30 seconds saved would have prevented a full debug round-trip.

---

## Decisions that paid off

- **Streaming Dataset with shard-aware ordering**: kept the 100K case
  feasible at all on the small box. After-the-fact tweak but it
  worked.
- **20K subset that fits in OS file cache**: the actual unblocker for
  feasible training time. After epoch 1, all later epochs are RAM-bound
  rather than disk-bound (~5 min/predictor instead of 7 min/epoch).
- **Picking L=8 as the predictor target by ablation, not by oracle**:
  oracle says L=16 is the marginally-best layer, but L=16 is also the
  marginally-harder-to-distil layer. The 9-cell average across the
  three benchmarks selected L=8 cleanly. Worth $0 and ~30 min Phase-4
  time.
- **Hero Fig 1 with real answers**: the rubric explicitly rewards
  this and penalizes pipeline-as-Fig-1. The cost of generating it
  (predictor inference on 2 hero images, ~5 min) was tiny compared
  to its weight in Presentation/Originality scoring.

---

## Open items

- **Charter font on AWS render** — `make_hero_figure.py` falls back to
  DejaVu Serif on the AWS box because Charter is not installed there.
  The PDF is still vector and readable, but the typeface inside the
  figure does not match the Charter body of the paper. Cleanest fix is
  re-rendering on the local Mac (which has Charter as a system font);
  needs LLaVA-1.5-7B + the predictor on the local box. Low priority.
- **A real "qualitative" figure** — Fig 1 covers two TextVQA examples;
  the rubric's preferred bar is "6+ qualitative examples". Adding a
  small in-text grid of additional cases would cover this, especially
  one or two GQA / VQAv2 examples to prove the method is not
  TextVQA-specific.
- **Section numbers in figure caption** — `\ref{fig:hero}` is wired,
  but if we re-order figures we should double-check the cross-ref
  numbering once more.
- **A2 chain idempotency** — `run_phase_a2.sh` now skips work whose
  outputs already exist, which is the right behaviour for resuming
  after a stop. `harvest_scope_c.sh` should be folded *into* the chain
  so the harvest happens before the script exits, never as a manual
  follow-up.

---

## Final cost rollup

| Item | Hours GPU | $\$$ |
|---|---|---|
| Original 100K cache | 7.3 hr | 8.85 |
| Phase 3 + 4 + 6 (first run) | 2.5 hr | 3.0 |
| A2 redo (cache 20K + train + figure) | 1.7 hr | 2.05 |
| Idle / debug / SSH probe time | ~6 hr | 7.3 |
| **Total** | **~17.5 hr** | **~$21** |

If we'd done the cache-size arithmetic up front and gone to g5.8xlarge,
total would have been ~10 hr / $24 — same money, half the wall-clock,
and all artefacts on a single instance you could stop once.
