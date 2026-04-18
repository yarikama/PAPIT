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

# 6. 跑實驗（用 tmux，斷線不中斷）
mkdir -p logs
tmux new-session -d -s gqa     "python scripts/run_eval.py --dataset gqa     --max-samples 700 --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_700 2>&1 | tee logs/gqa.log"
tmux new-session -d -s vqa     "python scripts/run_eval.py --dataset vqa_v2  --max-samples 700 --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_700 2>&1 | tee logs/vqa.log"
tmux new-session -d -s textvqa "python scripts/run_eval.py --dataset textvqa --max-samples 700 --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_700 2>&1 | tee logs/textvqa.log"

# 查看所有 session / 進入看進度（Ctrl+B, D 離開）
tmux ls
tmux attach -t gqa

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
# clone repo
git clone https://github.com/yarikama/PAPIT.git papit
cd papit

# 安裝套件（用系統 pip，Amazon Linux 上用 pip-3.13）
pip-3.13 install -e ".[ocr,llava]"

# HuggingFace token（避免 rate limit，可選）
export HF_TOKEN=hf_xxx
```

---

## Step 3 — 下載資料集

```bash
# 用 tmux 背景下載，斷線不中斷（總共約 33GB，~1 小時）
tmux new-session -d -s download "bash scripts/download_data.sh 2>&1 | tee download.log"

# 進入看進度（Ctrl+B, D 離開不中斷）
tmux attach -t download

# 確認完成
du -sh data/raw/gqa/images/ data/raw/vqa_v2/val2014/ data/raw/textvqa/train_val_images/
# 預期：~20G  ~6G  ~7G
```

---

## Step 4 — 準備 CSVs

```bash
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

## Step 5 — 主實驗：Hybrid 700 samples

**目的：** Final report 主要結果表（Table 3）。

```bash
mkdir -p logs

# 三個 dataset 各開一個 tmux session 同時跑
tmux new-session -d -s gqa \
    "python scripts/run_eval.py --dataset gqa --max-samples 700 \
    --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_700 2>&1 | tee logs/gqa.log"

tmux new-session -d -s vqa \
    "python scripts/run_eval.py --dataset vqa_v2 --max-samples 700 \
    --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_700 2>&1 | tee logs/vqa.log"

tmux new-session -d -s textvqa \
    "python scripts/run_eval.py --dataset textvqa --max-samples 700 \
    --retention 0.25 0.5 0.75 --output-dir outputs/hybrid_700 2>&1 | tee logs/textvqa.log"

# 查看所有 session
tmux ls

# 進入某個 session 看即時輸出（Ctrl+B, D 離開）
tmux attach -t gqa
```

輸出：
```
outputs/hybrid_700/gqa_eval/llava_benchmark_summary.csv
outputs/hybrid_700/vqa_v2_eval/llava_benchmark_summary.csv
outputs/hybrid_700/textvqa_eval/llava_benchmark_summary.csv
```

---

## Step 6 — OCR-forced TextVQA

**目的：** 驗證 OCR-forced 能否補救 TextVQA accuracy gap。

```bash
tmux new-session -d -s ocr \
    "python scripts/run_eval.py --dataset textvqa --force-ocr --max-samples 700 \
    --retention 0.25 0.5 0.75 --output-dir outputs/ocr_forced_700 2>&1 | tee logs/ocr_forced.log"

tmux attach -t ocr
```

---

## Step 7 — Anchor Strategy Ablation

**目的：** 驗證 `global_mean` 優於其他 anchor 策略。

```bash
tmux new-session -d -s anchor_dropped \
    "python scripts/run_eval.py --dataset gqa --anchor dropped_mean --max-samples 700 \
    --retention 0.25 0.5 0.75 --output-dir outputs/anchor_dropped_mean 2>&1 | tee logs/anchor_dropped.log"

tmux new-session -d -s anchor_none \
    "python scripts/run_eval.py --dataset gqa --anchor none --max-samples 700 \
    --retention 0.25 0.5 0.75 --output-dir outputs/anchor_none 2>&1 | tee logs/anchor_none.log"

tmux ls  # 看所有 session 狀態
```

---

## Step 8 — Efficiency Benchmark

**目的：** 修正 k=100% 的 cold-start 問題，填 Table 1 TBD。

```bash
python scripts/generate_figures.py \
    --image data/raw/textvqa/train_val_images/0000599864fd15b3.jpg \
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
ls outputs/hybrid_700/*/llava_benchmark_summary.csv
ls outputs/ocr_forced_700/textvqa_eval/llava_benchmark_summary.csv
ls outputs/anchor_dropped_mean/gqa_eval/llava_benchmark_summary.csv
ls outputs/anchor_none/gqa_eval/llava_benchmark_summary.csv
ls outputs/efficiency_benchmark.csv
```

---

## Step 10 — 下載結果到本機

```bash
# 本機執行
rsync -av -e "ssh -i ~/kerstin_aws_key.pem" \
    ec2-user@184.72.111.112:~/PAPIT/outputs/ \
    "/home/yuhan/workspace/comp 646/PAPIT/outputs/aws_results/"
```

然後在本機生成 Pareto 圖（不需要 GPU）：

```bash
python scripts/generate_figures.py \
    --skip-efficiency \
    --output-dir outputs/aws_results/hybrid_700
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
| Hybrid 700 | `outputs/hybrid_700/` | Final report Table 3 主結果 |
| OCR-forced 700 | `outputs/ocr_forced_700/` | TextVQA 分析 |
| Anchor ablation | `outputs/anchor_*/` | Design choice 驗證 |
| Efficiency | `outputs/efficiency_benchmark.csv` | Table 1 TBD 填入 |
| 圖表 | `outputs/fig_*.pdf` | Report Figure 1/2/3 |

現有的 ablation 資料（300 samples）保留不動：
- `outputs/gradcam_eval/` → Table 2 GradCAM row
- `outputs/value_eval/` → Table 2 Value features row
