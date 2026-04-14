# PAPIT — AWS Experiment Plan

> 在 AWS EC2 GPU instance 上跑完整實驗，無時間限制，費用約 $2–5。

---

## 快速開始

```bash
# 1. 本機先確認 code 都 push 了
git status
git push

# 2. 啟動 EC2（見 Step 1），SSH 進去後：
git clone https://github.com/yarikama/PAPIT.git papit
cd papit

# 3. 安裝環境
pip install uv
uv sync --extra ocr --extra llava
# 或是直接用 pip：
# pip install -e ".[ocr,llava]"

# 4. 下載資料
nohup bash scripts/download_data.sh > download.log 2>&1 &
tail -f download.log

# 5. 準備 CSVs
source .venv/bin/activate
python scripts/prepare_datasets.py gqa \
    --questions data/raw/gqa/val_balanced_questions.json \
    --images data/raw/gqa/images --output data/gqa_val.csv
python scripts/prepare_datasets.py vqa_v2 \
    --questions data/raw/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations data/raw/vqa_v2/v2_mscoco_val2014_annotations.json \
    --images data/raw/vqa_v2/val2014 --output data/vqa_v2_val.csv
python scripts/prepare_datasets.py textvqa \
    --annotations data/raw/textvqa/TextVQA_0.5.1_val.json \
    --images data/raw/textvqa/train_val_images \
    --max-samples 1000 --output data/textvqa_val.csv

# 6. 跑實驗（背景執行，斷線不中斷）
mkdir -p logs
nohup python scripts/run_eval.py --dataset gqa --max-samples 500 \
    --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_500 > logs/gqa.log 2>&1 &
nohup python scripts/run_eval.py --dataset vqa_v2 --max-samples 500 \
    --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_500 > logs/vqa.log 2>&1 &
nohup python scripts/run_eval.py --dataset textvqa --max-samples 500 \
    --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_500 > logs/textvqa.log 2>&1 &

# 7. 下載結果（本機執行）
rsync -av -e "ssh -i your-key.pem" \
    ubuntu@<EC2_IP>:~/papit/outputs/ \
    "/home/yuhan/workspace/comp 646/PAPIT/outputs/aws_results/"

# 8. 關掉 instance（省錢！）
# AWS Console → EC2 → Stop Instance
```

---

## Step 1 — 啟動 EC2 Instance

### 推薦機型

| 機型 | GPU | VRAM | Spot 費用 | 推薦用途 |
|------|-----|------|----------|---------|
| `g5.xlarge` | A10G 24GB | 24GB | ~$0.4/hr | **推薦**，LLaVA 跑很順 |
| `g4dn.xlarge` | T4 16GB | 16GB | ~$0.2/hr | 省錢但稍慢 |

LLaVA-1.5-7B 需要 ~14GB VRAM，兩種都能跑。

### 在 AWS Console 操作

1. EC2 → **Launch Instance**
2. **AMI**：搜尋 `Deep Learning AMI GPU PyTorch`，選最新版（已預裝 CUDA、conda）
3. **Instance type**：`g5.xlarge`
4. **Key pair**：建立或選現有 `.pem`
5. **Storage**：改成 **100 GB** gp3（GQA ~20GB + VQA ~6GB + TextVQA ~7GB + models ~30GB）
6. **Security group**：確認有 SSH (port 22)
7. Launch

### SSH 進去

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>

# 確認 GPU
nvidia-smi
```

---

## Step 2 — 安裝環境

```bash
# 安裝 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# clone repo
git clone https://github.com/yarikama/PAPIT.git papit
cd papit

# 安裝套件（包含 EasyOCR 和 LLaVA 依賴）
uv sync --extra ocr --extra llava

# HuggingFace token（避免 rate limit，可選）
export HF_TOKEN=hf_xxx
```

---

## Step 3 — 下載資料集

```bash
source .venv/bin/activate

# 用 nohup 背景下載，斷線不中斷（總共約 33GB，~1 小時）
nohup bash scripts/download_data.sh > download.log 2>&1 &
echo "PID: $!"

# 追蹤進度
tail -f download.log

# 確認完成
du -sh data/raw/gqa/images/ data/raw/vqa_v2/val2014/ data/raw/textvqa/train_val_images/
# 預期：~20G  ~6G  ~7G
```

---

## Step 4 — 準備 CSVs

```bash
source .venv/bin/activate

# GQA（快，~30秒）
python scripts/prepare_datasets.py gqa \
    --questions data/raw/gqa/val_balanced_questions.json \
    --images    data/raw/gqa/images \
    --output    data/gqa_val.csv

# VQA v2（快，~30秒）
python scripts/prepare_datasets.py vqa_v2 \
    --questions    data/raw/vqa_v2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations  data/raw/vqa_v2/v2_mscoco_val2014_annotations.json \
    --images       data/raw/vqa_v2/val2014 \
    --output       data/vqa_v2_val.csv

# TextVQA（有 GPU 約 5 分鐘，1000 samples）
python scripts/prepare_datasets.py textvqa \
    --annotations data/raw/textvqa/TextVQA_0.5.1_val.json \
    --images      data/raw/textvqa/train_val_images \
    --max-samples 1000 \
    --output      data/textvqa_val.csv
```

---

## Step 5 — 主實驗：Hybrid 500 samples

**目的：** Final report 主要結果表（Table 3）。

```bash
mkdir -p logs

# 三個 dataset 同時背景執行（各自獨立的 nohup）
nohup python scripts/run_eval.py \
    --dataset gqa --max-samples 500 \
    --retention 0.25 0.5 0.75 \
    --output-dir outputs/hybrid_500 > logs/gqa.log 2>&1 &

nohup python scripts/run_eval.py \
    --dataset vqa_v2 --max-samples 500 \
    --retention 0.25 0.5 0.75 \
    --output-dir outputs/hybrid_500 > logs/vqa.log 2>&1 &

nohup python scripts/run_eval.py \
    --dataset textvqa --max-samples 500 \
    --retention 0.25 0.5 0.75 \
    --output-dir outputs/hybrid_500 > logs/textvqa.log 2>&1 &

# 追蹤進度
tail -f logs/gqa.log
```

輸出：
```
outputs/hybrid_500/gqa_eval/llava_benchmark_summary.csv
outputs/hybrid_500/vqa_v2_eval/llava_benchmark_summary.csv
outputs/hybrid_500/textvqa_eval/llava_benchmark_summary.csv
```

---

## Step 6 — OCR-forced TextVQA

**目的：** 驗證 OCR-forced 能否補救 TextVQA accuracy gap。

```bash
nohup python scripts/run_eval.py \
    --dataset textvqa \
    --force-ocr \
    --max-samples 500 \
    --retention 0.25 0.5 0.75 \
    --output-dir outputs/ocr_forced_500 > logs/ocr_forced.log 2>&1 &

tail -f logs/ocr_forced.log
```

---

## Step 7 — Anchor Strategy Ablation

**目的：** 驗證 `global_mean` 優於其他 anchor 策略。

```bash
nohup python scripts/run_eval.py \
    --dataset gqa --anchor dropped_mean \
    --max-samples 300 \
    --retention 0.25 0.5 0.75 \
    --output-dir outputs/anchor_dropped_mean > logs/anchor_dropped.log 2>&1 &

nohup python scripts/run_eval.py \
    --dataset gqa --anchor none \
    --max-samples 300 \
    --retention 0.25 0.5 0.75 \
    --output-dir outputs/anchor_none > logs/anchor_none.log 2>&1 &
```

---

## Step 8 — Efficiency Benchmark

**目的：** 修正 k=100% 的 cold-start 問題，填 Table 1 TBD。

```bash
python scripts/generate_figures.py \
    --image data/raw/textvqa/train_val_images/train_images/a50551c2199738ce.jpg \
    --prompt "What does the sign say?" \
    --output-dir outputs
```

產生：`outputs/efficiency_benchmark.csv`、`outputs/fig_efficiency.pdf`、`outputs/fig_qualitative.pdf`

---

## Step 9 — 確認所有實驗完成

```bash
# 確認所有 nohup process 結束
ps aux | grep run_eval | grep -v grep

# 確認 CSV 存在
ls outputs/hybrid_500/*/llava_benchmark_summary.csv
ls outputs/ocr_forced_500/textvqa_eval/llava_benchmark_summary.csv
ls outputs/anchor_dropped_mean/gqa_eval/llava_benchmark_summary.csv
ls outputs/anchor_none/gqa_eval/llava_benchmark_summary.csv
ls outputs/efficiency_benchmark.csv
```

---

## Step 10 — 下載結果到本機

```bash
# 本機執行
rsync -av -e "ssh -i your-key.pem" \
    ubuntu@<EC2_IP>:~/papit/outputs/ \
    "/home/yuhan/workspace/comp 646/PAPIT/outputs/aws_results/"
```

然後在本機生成 Pareto 圖（不需要 GPU）：

```bash
python scripts/generate_figures.py \
    --skip-efficiency \
    --output-dir outputs/aws_results/hybrid_500
```

---

## Step 11 — 關掉 Instance（重要！）

```bash
# AWS Console → EC2 → Instances → 選 instance → Instance State → Stop
```

**Stop（停止）≠ Terminate（終止）**
- Stop：保留磁碟，下次可以繼續用（磁碟費用約 $0.1/GB/月）
- Terminate：永久刪除，資料消失

---

## 費用估算

| 工作 | 時間 | 費用（g5.xlarge On-Demand） |
|------|------|--------------------------|
| 資料下載 | ~1 hr | $0.48 |
| 主實驗 × 3 datasets | ~3 hr | $1.44 |
| OCR-forced + Anchor | ~2 hr | $0.96 |
| Efficiency benchmark | ~0.5 hr | $0.24 |
| **總計** | **~6.5 hr** | **~$3.12** |

> Spot instance 可省 60–70%，但有可能被中斷。On-Demand 更穩定，建議用 On-Demand。

---

## 實驗結果整理（跑完後）

| 實驗 | 輸出目錄 | 用途 |
|------|---------|------|
| Hybrid 500 | `outputs/hybrid_500/` | Final report Table 3 主結果 |
| OCR-forced 500 | `outputs/ocr_forced_500/` | TextVQA 分析 |
| Anchor ablation | `outputs/anchor_*/` | Design choice 驗證 |
| Efficiency | `outputs/efficiency_benchmark.csv` | Table 1 TBD 填入 |
| 圖表 | `outputs/fig_*.pdf` | Report Figure 1/2/3 |

現有的 ablation 資料（300 samples）保留不動：
- `outputs/gradcam_eval/` → Table 2 GradCAM row
- `outputs/value_eval/` → Table 2 Value features row
