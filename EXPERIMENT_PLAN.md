# PAPIT — Experiment Plan

> **Chao Hsuan Ho · Heng Jui Hsu · Kerstin Sun**
> COMP 646, Rice University

---

## 1. 專案背景與核心假設

### 我們在解決什麼問題

LLaVA 等 Multimodal LLM 會把一張圖切成 576 個 patch token（24×24 grid），全部送進 LLM 的 attention 層。然而大部分的 patch 和當前問題無關——問「手指有幾根」時，背景的天空 patch 對回答毫無幫助，卻佔用了相同的計算資源。

**Global pruning**（丟掉固定比例的 patch）無法保證保留和 prompt 相關的 patch。

### PAPIT 的解法

用 CLIP 的 text encoder 把 prompt 編成 embedding，再計算每個 patch 和 prompt 的 cosine similarity，只保留分數最高的 top-k 個 patch。這樣一來，patch 的選擇是**由 prompt 驅動**的（Prompt-Aware），而不是固定規則。

```
Image → ViT → 576 patch tokens
Prompt → CLIP Text Encoder → text embedding
                 ↓
      cosine similarity scoring (576 scores)
                 ↓
      top-k selection + spatial anchor
                 ↓
      k patch tokens → LLaVA MLP projector → LLaVA LLM → Answer
```

### 我們要驗證的三個假設

| # | 假設 | 驗證方式 |
|---|---|---|
| H1 | PAPIT 在相同 token budget 下，準確率高於 random pruning | VQA soft accuracy: PAPIT vs Random at k=25%/50%/75% |
| H2 | PAPIT 在大幅削減 token 數的情況下，仍能維持接近 unpruned 的準確率 | Pareto curve: accuracy vs relative FLOPs |
| H3 | 在含文字的圖（TextVQA），PAPIT 選到文字區域 patch 的比例遠高於 random | TextVQA patch recall: PAPIT vs Random |

---

## 2. 已完成的元件

| 模組 | 位置 | 說明 |
|---|---|---|
| CLIP cross-modal pruner | `papit/core/pruner.py` | 核心：cosine scoring + top-k 選取 + spatial anchor |
| OCR forced retention | `papit/ocr/retention.py` | 用 EasyOCR 偵測文字框，強制保留對應 patch |
| Risk awareness | `papit/risk/awareness.py` | Safety-keep（安全標誌）+ instruction-block（jailbreak 圖） |
| LLaVA integration | `papit/integration/llava.py` | 真正的 token-level 整合：pruning 發生在 ViT → MLP projector 之間 |
| Dataset loaders | `papit/data/` | GQA / VQA v2 / TextVQA → unified CSV |
| LLaVA benchmark runner | `papit/benchmark/llava_runner.py` | 跑 PAPIT + Random + Unpruned，輸出 detailed + summary CSV |
| Efficiency benchmark | `papit/benchmark/efficiency.py` | 量 latency、GPU memory（已跑完，見 `outputs/efficiency_benchmark.csv`） |

---

## 3. 執行流程總覽

```
Step 0  跑 demo.ipynb          驗證環境 + 理解 pipeline
  ↓
Step 1  下載資料集              GQA / VQA v2 / TextVQA 原始檔案
  ↓
Step 2  執行 eval.ipynb §1     把原始檔案轉成 CSV
  ↓
Step 3  執行 eval.ipynb §2     LLaVA 大規模 benchmark（需 GPU ≥16GB）
  ↓
Step 4  執行 eval.ipynb §3     產生論文圖表和 Table
  ↓
Step 5  (Optional) eval.ipynb §4   Risk awareness ablation
```

---

## Step 0 — 跑 demo.ipynb

`notebooks/demo.ipynb` 是整個 pipeline 的互動式展示，依序示範：

| Section | 內容 | 結果 |
|---|---|---|
| §1 | 載入一張 TextVQA 真實照片 | 看到含文字的街景圖 |
| §2 | CLIP scoring + top-k pruning | 選中的 patch 集中在文字/目標物附近 |
| §3 | LLM 維度對齊確認 | shape `(k+1, 4096)` |
| §4 | Retention ratio sweep（單圖） | 視覺上看 25%/50%/75% 的剪枝差異 |
| §5 | OCR-guided retention | 文字 patch 強制保留，coverage 提升 |
| §6 | Risk awareness | Safety patch 保留，jailbreak patch 塗黑 |
| §7 | Efficiency benchmark (BLIP) | Latency 隨 k 線性下降，已存 CSV |
| §9 | LLaVA 真實 token-level 推理 | 單張圖的 pruned vs unpruned 比較 |

---

## Step 1 — 下載資料集

> 所有原始檔案放在 `data/raw/`（不 commit 進 git）。
> **下載指令與解壓步驟詳見 `eval.ipynb` Section 1-A。**

| Dataset | 大小 | 為何選用 |
|---|---|---|
| **GQA** | ~20 GB | 多步空間推理，對 patch 選擇品質要求最高 |
| **VQA v2** | ~6 GB | 學術標準 benchmark，可與發表文獻直接對比；每題有 10 個 human annotation 支援 soft accuracy |
| **TextVQA** | ~7 GB | 文字通常只佔 1–3 個 patch 但對回答至關重要，最直接測試 prompt-awareness；含 OCR bounding box 可算 patch recall |

---

## Step 2 — 資料前處理（eval.ipynb §1-B）

> **轉換指令詳見 `eval.ipynb` Section 1-B。**
> 執行後產生三個 CSV 檔（各 500–1000 筆）供 Step 3 使用。

每個 CSV 的統一格式：

| 欄位 | 說明 |
|---|---|
| `image_path` | 圖片絕對/相對路徑 |
| `question` | 問題文字 |
| `answer` | canonical 答案（最多人給的） |
| `answer_list` | 所有 annotator 的答案 JSON list（VQA soft 評分用） |
| `ocr_boxes` | （TextVQA only）OCR bounding box list，用來算 patch recall |

---

## Step 3 — LLaVA Accuracy Benchmark（eval.ipynb §2）

**需要 GPU ≥ 16 GB VRAM**（LLaVA-1.5-7B）。三個 dataset 的 cell 各自獨立，可以分批跑。

### 跑了什麼

對每張圖、每個 retention ratio，跑三個 variant：

```
同一張圖、同一個問題：
  1. Unpruned LLaVA    → 所有 576 patches 進 LLM
  2. PAPIT             → top-k patches by CLIP score
  3. Random pruning    → 隨機選 k patches（seed per sample）
```

三個 variant 共用同一個 LLaVA model，只有 patch 選法不同，確保對比公平。

### 計算的指標

| 指標 | 說明 |
|---|---|
| **VQA soft accuracy** | `min(count_matching_answers / 3, 1.0)`，VQA v2 官方標準 |
| **Token keep ratio** | 實際 k / N（含 anchor token） |
| **Relative FLOPs** | `(k + L_text)² / (N + L_text)²`，attention 複雜度的二次方關係 |
| **Latency** | wall-clock 時間（pruning + generation） |
| **TextVQA patch recall** | PAPIT/random 選中的 patch 中，有多少覆蓋到 OCR 文字區域 |

### 預估執行時間

| Dataset | Samples | 預估時間 (A100) |
|---|---|---|
| GQA | 1 000 | ~25 min |
| VQA v2 | 1 000 | ~25 min |
| TextVQA | 500 | ~15 min |

---

## Step 4 — 產生論文圖表（eval.ipynb §3）

### 預期結果與解讀

**H1（PAPIT > Random 準確率）**

我們預期在所有 retention ratio 下，PAPIT 都優於 random，尤其在 k=25% 時差距最大。原因：random 丟棄 patch 不分青紅皂白；PAPIT 保留的 patch 在語意上與問題相關。

**H2（PAPIT 接近 Unpruned）**

- k=75%：PAPIT ≈ Random ≈ Unpruned（大部分 patch 都留下，差異小）
- k=50%：PAPIT 略低於 Unpruned，Random 再低一點
- k=25%：兩者都下降，但 PAPIT 下降幅度應顯著小於 Random

**H3（TextVQA patch recall）**

PAPIT 的 text patch recall 應遠高於 random（隨機選的話，只有 k 比例的文字 patch 會被選到）。這是最直接的 prompt-awareness 佐證。

---

### Figure 1 — Accuracy–Efficiency Pareto Curve

- **橫軸：** Relative FLOPs（1.0 = unpruned）
- **縱軸：** VQA Soft Accuracy
- **三條線：** PAPIT（實線）/ Random（虛線）/ Unpruned（灰色水平線）
- **三個子圖：** GQA / VQA v2 / TextVQA
- **每點標 k 值：** 25% / 50% / 75%

期望看到：PAPIT 的 Pareto frontier 優於 Random（在相同 FLOPs 下，PAPIT 的 accuracy 更高）。

---

### Figure 2 — TextVQA Patch Recall

- **橫軸：** Retention ratio (k / N)
- **縱軸：** 文字區域 patch recall
- **兩條線：** PAPIT vs Random

期望看到：PAPIT 的曲線明顯高於 Random（e.g., k=50% 時 PAPIT recall ~80% vs Random ~50%）。

---

### Table 1 — 主要結果表

適合放進論文 Section 4（Experiments）。

| Dataset | Method | k=25% | k=50% | k=75% | Unpruned |
|---|---|---|---|---|---|
| GQA | PAPIT | ? | ? | ? | ? |
| GQA | Random | ? | ? | ? | — |
| VQA v2 | PAPIT | ? | ? | ? | ? |
| VQA v2 | Random | ? | ? | ? | — |
| TextVQA | PAPIT | ? | ? | ? | ? |
| TextVQA | Random | ? | ? | ? | — |

---

### Table 2 — 效率指標

適合放在論文 Section 4 或 Appendix。

| k | Tokens kept | Relative FLOPs | Prune latency (ms) | E2E latency (ms) |
|---|---|---|---|---|
| 100% (unpruned) | 576 | 1.000 | 0 | — |
| 75% | 432 | ~0.62 | — | — |
| 50% | 288 | ~0.32 | — | — |
| 25% | 144 | ~0.12 | — | — |

> Relative FLOPs 的數值是估算（`(k + 50)² / (576 + 50)²`），跑完 benchmark 後填入實測 latency。

---

### Table 3（Optional）— TextVQA Patch Recall

| Method | k=25% | k=50% | k=75% |
|---|---|---|---|
| PAPIT | ? | ? | ? |
| Random | ~25% | ~50% | ~75% |

> Random 的 recall 理論值等於 k（隨機選 k 比例，期望就覆蓋 k 比例的文字 patch），可作為合理性檢查。

---

## Step 5 — Risk Awareness Ablation（Optional，eval.ipynb §4）

**目標：** 驗證 Risk Awareness module 確實能保留安全關鍵 patch 並封鎖 jailbreak patch。

**資料：** 需要自行準備一小批圖片（10–20 張）：
- 含 STOP / WARNING / HAZARD 等安全標誌的圖（測 safety-keep）
- 含「ignore previous instructions」等注入文字的圖（測 instruction-block）

**對比：** Base PAPIT（無 risk module）vs Risk-aware PAPIT
**指標：** 安全 patch 保留率、jailbreak patch 封鎖率、最終回答的行為差異

---

## 進度 Checklist

### Step 0 — demo.ipynb
- [x] 環境確認（CUDA / MPS / CPU）
- [x] CLIP pruner + OCR + Risk awareness pipeline
- [x] Efficiency benchmark 跑完（`outputs/efficiency_benchmark.csv` 存在）
- [x] LLaVA single-image smoke test

### Step 1 — 下載資料集
- [ ] GQA images + val_balanced_questions.json
- [ ] VQA v2 val2014 images + questions + annotations
- [ ] TextVQA val JSON + train_val_images

### Step 2 — eval.ipynb §1 資料前處理
- [ ] `data/gqa_val.csv` (1 000 rows)
- [ ] `data/vqa_v2_val.csv` (1 000 rows)
- [ ] `data/textvqa_val.csv` (500 rows，含 `ocr_boxes` 欄位)

### Step 3 — eval.ipynb §2 LLaVA Benchmark
- [ ] `outputs/gqa_eval/llava_benchmark_summary.csv`
- [ ] `outputs/vqa_v2_eval/llava_benchmark_summary.csv`
- [ ] `outputs/textvqa_eval/llava_benchmark_summary.csv`

### Step 4 — eval.ipynb §3 圖表
- [ ] `outputs/pareto_curve.pdf` (Figure 1)
- [ ] `outputs/textvqa_patch_recall.pdf` (Figure 2)
- [ ] `outputs/table1.csv` (Table 1 主要結果)
- [ ] 填入 Table 2 效率數字（從 `efficiency_benchmark.csv` + LLaVA benchmark 取）

### Step 5 — Optional
- [ ] 收集 adversarial images
- [ ] Risk ablation 結果

---

## 檔案結構

```
PAPIT/
├── papit/
│   ├── core/           CLIP scorer, config
│   ├── integration/    LLaVA token-level integration  ← 核心
│   ├── benchmark/      llava_runner.py, efficiency.py
│   ├── data/           GQA / VQA v2 / TextVQA loaders
│   ├── ocr/            EasyOCR forced retention
│   └── risk/           Safety-keep + instruction-block
│
├── notebooks/
│   ├── demo.ipynb      Pipeline 展示
│   └── eval.ipynb      實驗執行（Steps 2–5 在這裡跑）
│
├── data/
│   ├── raw/            原始 dataset
│   ├── gqa_val.csv
│   ├── vqa_v2_val.csv
│   └── textvqa_val.csv
│
├── outputs/
│   ├── efficiency_benchmark.csv
│   ├── gqa_eval/
│   ├── vqa_v2_eval/
│   ├── textvqa_eval/
│   ├── pareto_curve.pdf
│   ├── textvqa_patch_recall.pdf
│   └── table1.csv
│
├── scripts/
│   └── prepare_datasets.py   CLI 版的資料前處理（同 eval.ipynb §1）
│
└── proposal/
    └── proposal.pdf
```
