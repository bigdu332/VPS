# MathVista 评测记录

## 基本信息

| 项目 | 值 |
|------|-----|
| 评测模型 | MMR1-3B-SFT |
| 模型路径 | `/mnt/data/ericdu/models/MMR1-3B-SFT` |
| 数据集 | MathVista testmini 前 100 条（固定，可复现） |
| 数据路径 | `/mnt/data/ericdu/datasets/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet` |
| 服务器 | `ssh -p 51118 root@115.190.60.96` |

---

## 评测结果（2026-05-02，前 100 条）

| 指标 | 值 |
|------|-----|
| **Overall** | **51.0%（51/100）** |
| free_form | 47.8%（22/46） |
| multi_choice | 53.7%（29/54） |

> 使用 vllm V0 engine 推理，batch_size=32，约 3.5 分钟完成。

---

## 脚本说明

| 脚本 | 说明 |
|------|------|
| `run_mathvista_vllm.py` | **推荐**，vllm V0 engine，快（~3.5min/100条） |
| `run_mathvista.py` | transformers 版本，稳定但慢（~40min/100条） |

### 推理命令

```bash
cd /mnt/data/ericdu/mmr1/evaluation/mathvista

# vllm 版（推荐）
CUDA_VISIBLE_DEVICES=3 python run_mathvista_vllm.py infer \
    --model-path /mnt/data/ericdu/models/MMR1-3B-SFT \
    --data-path /mnt/data/ericdu/datasets/MathVista/data/testmini-00000-of-00001-725687bf7a18d64b.parquet \
    --output-file /mnt/data/ericdu/output/mathvista_eval/infer_results_vllm.jsonl \
    --batch-size 32 --num-samples 100

# 评分
python run_mathvista_vllm.py eval \
    --input-file /mnt/data/ericdu/output/mathvista_eval/infer_results_vllm.jsonl \
    --output-file /mnt/data/ericdu/output/mathvista_eval/eval_results_vllm.json
```

---

## 关键 Trick

### vllm V1 vs V0 engine

- **vllm 0.8.4 默认用 V1 engine**，对 Qwen2.5-VL 会 OOM：
  - V1 按最大分辨率预分配 encoder cache（16384 tokens），即使实际图片很小也会占满显存
  - 调 `max_model_len`、`gpu_memory_utilization`、`tensor_parallel_size` 都无效
- **解决：加 `VLLM_USE_V1=0` 强制用 V0 engine**，显存管理正常，80GB 单卡轻松跑

```python
import os
os.environ['VLLM_USE_V1'] = '0'  # 必须在 import vllm 之前设置
from vllm import LLM, SamplingParams

llm = LLM(
    model=model_path,
    dtype='bfloat16',
    max_model_len=8192,
    limit_mm_per_prompt={'image': 1},
    mm_processor_kwargs={'max_pixels': 1280*28*28},
    gpu_memory_utilization=0.9,
    enforce_eager=True,   # 禁 cuda graph，进一步省显存（可选）
)
```

### precision 字段解读（避免虚高分数）

官方 precision 是**小数位数**，不是误差阈值：

```python
# ❌ 错误：把 precision 当绝对误差
abs(float(pred) - float(gt)) < precision  # precision=1.0 → ±1.0 范围内全算对

# ✅ 正确：precision 是小数位数
dec = int(precision) if precision >= 1 else max(0, round(-np.log10(precision)))
round(float(pred), dec) == round(float(gt), dec)  # precision=1.0 → 保留1位小数精确匹配
```

### numpy 版本

vllm 依赖的 numba 要求 `numpy <= 2.2`，环境里是 2.4，需要降级：
```bash
pip install 'numpy<2.2'
```

---

## 当前进度（2026-05-02 20:02）

| GPU | 版本 | 状态 | 输出 |
|-----|------|------|------|
| GPU 2 | transformers | 运行中（38/100） | `infer_results.jsonl` |
| GPU 3 | vllm V0 | **已完成** ✅ | `infer_results_vllm.jsonl` |

---

## 数据集结构

- **总量**：1000 条（testmini），前 100 条固定可复现
- **题型**：multi_choice 540条 / free_form 460条
- **字段**：`pid, question, decoded_image, choices, precision, answer, question_type, answer_type, query`
- `query` 字段是完整 prompt（含 hint + 题目 + choices），直接用
- 图片从 `decoded_image.bytes` 读，无需下载

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `infer_results_vllm.jsonl` | vllm 推理结果（100条）✅ |
| `eval_results_vllm.json` | vllm 评分结果 ✅ |
| `infer_results.jsonl` | transformers 推理结果（进行中） |
