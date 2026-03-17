
![PAPIT progress figure](Pasted%20image%2020260316195244.png)
PAPIT（Prompt-Aware Patch/Image Token Pruning，基于提示的图像 Patch / Token 剪枝）旨在在保持回答质量的同时，降低视觉语言推理中的视觉 token 负载。

核心思想：
- 计算 图像 patch 对当前 prompt 的相关性。
- 在给定的保留预算下，仅保留 top-k 重要 patch。
- 通过 anchor 机制 和可选的 OCR / 风险感知保护机制 保留关键上下文。

目标效果：
- 更高效率（更少 token、更低延迟、更低显存占用）
- 在 VQA 类任务上尽量保持回答质量不下降

# 目前模塊

### PAPIT pipeline
- 基于 **CLIP 的 patch-text 相关性评分**
- 基于 **retention ratio 的 top-k 剪枝**        
- 为选中的 token 输出 **位置 remapping**
- 为 LLM 空间提供 **投影器（projector）**

## Optional

| Section | Anchor                             | OCR                     | Risk Filtering |
| ------- | ---------------------------------- | ----------------------- | -------------- |
| Input   | [k, D]                             | [k, D]                  | [k, D]         |
| Output  | [k+1, D]                           | [k, D]                  | [k, D]         |
| 含意      | 丟棄太多補丁，可能會遺失全域場景上下文，加入全局 or 被丟棄 資料 | CLIP 有時會忽略小文字，強迫文字補丁保留。 | 忽略不安全的視覺提示注射   |
### Anchor
selected_tokens: [k, D]
anchor [1, D]
output (selected + anchor) = [k+1, D]

- Anchor 策略：
    - none
    - global_mean
    - dropped_mean

Strategy 0:
Strategy 1:
    anchor = mean(all_patch_embedding)
Strategy 2: 
    anchor = mean(dropped_patch_embedding)

### OCR 保留机制
用于提升 含文本场景（text-heavy scenes） 的表现。
主要能力：
- OCR 文本框提取
- 将 OCR 框 映射到 patch grid
- 在 相同 token 预算下强制保留 OCR 区域

1. `base_indices = [3,10,15,22,...]`
2. Now OCR detects text. 
	- OCR boxes: box1, box2
	- box1 → patch 50，box2 → patch 51
3. `final_indices =  [3,10,15,22,...,50,51]`
4. But we still enforce `k` remaining, so we replace some low-score patches.

### Risk-Aware Filtering
Example: `patch 75 = malicious`
Filtering: `blocked_indices = [75]`
During selection: 
```
if idx in blocked_indices:
    skip
```

### Batch Benchmark 框架
Baseline 方法：base, OCR-guided, random
評估指標：
- 与原始回答的一致性（consistency）
- OCR 强制区域覆盖率
- EM / F1（当有标准答案时）

### 效率 Benchmark
token 保留比例, 剪枝延迟, QA 推理延迟, 端到端延迟, GPU 峰值显存


### 剩余工作
- 真正的 token-level LLaVA 内部集成
	（在 projector / LLM 之前的 token path 上进行剪枝）
- 大规模官方 benchmark 运行
- 基于大规模实验生成最终论文表格与图表