# GRPO 训练记录

## 基本信息

| 项目 | 值 |
|------|-----|
| 算法 | GRPO |
| 基座模型 | MMR1-3B-SFT（Qwen2.5-VL-3B，**3.75B** 参数） |
| 模型路径 | `/mnt/data/ericdu/models/MMR1-3B-SFT` |
| 训练数据 | `/mnt/data/ericdu/datasets/MMR1-RL/data`（2 个 parquet） |
| 验证数据 | `/mnt/data/ericdu/datasets/MathVista/data`（仅 test split，val_freq=-1 不使用） |
| 框架 | verl 0.2.0.dev0（volcengine 版）+ ray + vllm |
| 服务器 | `ssh -p 51118 root@115.190.60.96`（8× A100-SXM4-80GB） |
| 环境 | `conda activate /mnt/data/ericdu/conda_envs/verl-mmr1`（Python 3.12） |
| checkpoint 输出 | `/mnt/data/ericdu/output/mmr1_3b_rl/checkpoints` |
| 日志 | `/mnt/data/ericdu/output/mmr1_3b_rl/train.log` |
| TensorBoard | `http://115.190.60.96:6006`（SSH 隧道 `ssh -p 51118 -L 6006:localhost:6006 root@115.190.60.96`） |

---

## 当前进度（2026-05-03 21:59）

- **状态**：第 16 次，8 卡 + batch=128 + micro_batch=2 + max_resp=4096 + **liger-kernel** + **gc.collect + empty_cache 全面补丁**
- **已验证通过**：fit() ✅、rollout ✅、reward ✅、**compute_log_probs ✅**、**compute_ref_log_probs ✅**、**update_policy 27/64 进行中（无 OOM）**
- **等待**：Step 1 metrics 输出 + checkpoint 保存

### 当前 Plan

1. ✅ fit() 进入确认
2. ✅ rollout 完成
3. ✅ reward 计算完成
4. ✅ compute_log_probs 64/64 通过（empty_cache + .cpu() 补丁）
5. ✅ compute_ref_log_probs 通过
6. ⏳ update_policy 17/64 进行中（无 OOM）
7. ⬜ Step 1 metrics 输出
8. ⬜ checkpoint 保存确认
9. ⬜ 恢复完整 curriculum，正式跑 10 episodes

---

## 启动命令

```bash
ssh -p 51118 root@115.190.60.96
# 清理环境
pkill -9 -f "ray|verl"; sleep 3; ray stop --force; rm -rf /tmp/ray/
# 启动
cd /mnt/data/ericdu/mmr1
rm -f /mnt/data/ericdu/output/mmr1_3b_rl/train.log
nohup bash run_grpo.sh > /mnt/data/ericdu/output/mmr1_3b_rl/train.log 2>&1 &
disown
tail -f /mnt/data/ericdu/output/mmr1_3b_rl/train.log
```

---

## 最终启动脚本 `run_grpo.sh`

```bash
#!/bin/bash
set -x
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0

cd /mnt/data/ericdu/mmr1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/ericdu/conda_envs/verl-mmr1

export VLLM_USE_V1=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TENSORBOARD_DIR=/mnt/data/ericdu/output/mmr1_3b_rl/tensorboard

JOB_NAME=mmr1_3b_rl bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    trainer.total_episodes=1 \
    trainer.n_gpus_per_node=8 \
    worker.actor.fsdp.fsdp_size=8 \
    worker.actor.global_batch_size=128 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    data.rollout_batch_size=128 \
    data.curriculum_rollout_batch_size=128 \
    data.max_response_length=4096 \
    worker.rollout.enforce_eager=true \
    worker.rollout.gpu_memory_utilization=0.75 \
    worker.rollout.max_num_batched_tokens=65536
```

---

## 脚本链路

```
run_grpo.sh                                   # 启动脚本（环境变量 + 配置）
└── examples/mmr1/train_qwen2_5_vl_3b.sh      # 设模型路径、system prompt
    └── examples/mmr1.yaml                     # 训练配置（数据、超参、算法、worker）
        └── verl/trainer/main.py               # 主入口，解析 config 启动 ray
            └── verl/trainer/ray_trainer.py     # 训练主循环（episode → rollout → update）
                ├── verl/workers/fsdp_workers.py
                ├── verl/workers/actor/dp_actor.py     # padding_free forward + compute_log_prob
                ├── verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
                ├── verl/workers/reward/custom.py
                │   └── verl/utils/reward_score/openr1_rewards.py
                ├── verl/trainer/core_algos.py         # GRPO 核心（advantage、KL）
                ├── verl/utils/dataset.py              # 数据加载（parquet + 图片）
                ├── verl/curriculum/samplers.py         # Curriculum sampling
                └── verl/utils/logger/logger.py        # ConsoleLogger + TensorBoardLogger
```

---

## 关键超参

| 参数 | 当前值 | 论文原文 | 说明 |
|------|--------|---------|------|
| total_episodes | 1（试跑） | 10 | 训练轮数 |
| rollout_batch_size | **128** | 512 | 降了防 OOM |
| GRPO rollout n | 8 | 8 | 每个 prompt 生成 8 个回答 |
| actor lr | 1e-6 | 1e-6 | 学习率 |
| max_prompt_length | 2048 | 2048 | prompt 最大 token |
| **max_response_length** | **4096** | **4096** | yaml 默认 8192 有误 |
| FSDP size | 8 | 8 | 8 卡全分片 |
| padding_free | true | — | 用 flash_attn unpad |
| **micro_batch_for_experience** | **2** | **32** | 降了防 OOM + empty_cache 补丁 |
| gpu_memory_utilization | 0.75 | 0.75 | vllm KV cache 占比 |
| kl_penalty | low_var_kl | low_var_kl | KL 惩罚方式 |
| kl_coef | 1e-2 | 1e-2 | KL 系数 |
| sampling_strategy | curriculum | curriculum | 课程学习采样 |
| save_freq | 1 | — | 每 step 存一次 |
| reward | openr1_wo_LMM | format+accuracy | 纯规则 reward |
| PPO epochs | 1 | **20** | ⚠️ 正式训练需调 |
| curriculum_update_freq | 4 | **56** | VAS 更新频率 |

---

## System Prompt

```
A conversation between User and Assistant.
The User provides an image and asks a question.
The Assistant first analyzes both the image and the question,
then carefully thinks about the reasoning process step by step,
and finally provides the User with an accurate answer.
The Assistant must carefully checkout the correctness and validity of each reasoning step.
If any errors or inconsistencies are found during the reasoning process,
the Assistant reflects and corrects them logically.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags,
i.e., <think> reasoning process here, with potential reflections and corrections </think>
<answer> final answer here, with the key result enclosed in \boxed{} </answer>.
```

---

## 已完成的代码修复

### 修复 1：TensorBoard 初始化崩溃
- **文件**：`verl/utils/logger/logger.py`
- **问题**：`add_hparams(flatten_dict(config))` 传入不兼容类型
- **修复**：类型过滤 + try/except

### 修复 2：ConsoleLogger 输出被 Ray buffer 吞
- **修复**：`sys.stdout.flush()` + `PYTHONUNBUFFERED=1` + `RAY_DEDUP_LOGS=0`

### 修复 3：Checkpoint 保存无日志
- **修复**：`_save_checkpoint()` 前后加 `[CKPT]` print

### 修复 4：fit() debug 打印
- **修复**：fit() 入口加 `[DEBUG fit]` print

### 修复 5：验证集 split 错误
- **修复**：val_dataset 创建加 try/except fallback

### 修复 6：Curriculum 跳过（快速测试）
- **修复**：`torch.ones(len(dataset))` 替代 `_update_curriculum_weights()`

### 修复 7：flash_attn 安装
- ✅ 2.7.4.post1 + cu12 + torch2.6 + cxx11abiFALSE + cp312

### 修复 8：compute_log_probs 显存泄漏（OOM 终极解法）
- **文件**：`verl/workers/actor/dp_actor.py`（第 249 行）
- **问题**：`compute_log_prob()` 循环内每个 micro batch 的 FSDP all-gather 产生显存碎片逐步累积，log_probs 结果也留在 GPU 上。循环内没有 `empty_cache()`，verl 官方只在 rollout→compute 切换时清一次
- **现象**：每处理一个 micro batch 显存泄漏 ~4 GB，13/32 时从 5 GB 涨到 56 GB → OOM
- **修复**：
  ```python
  # 原代码
  log_probs_lst.append(log_probs)
  
  # 修复后
  log_probs_lst.append(log_probs.cpu())  # 结果移到 CPU，释放 GPU 显存
  gc.collect()
  torch.cuda.empty_cache()               # 清除 CUDA allocator 碎片
  ```
- **效果**：每次 forward 后 GPU 回落到 ~5 GB，峰值仅 ~28 GB（vs 80 GB 容量），64/64 micro batch 全部通过
- **这是 OOM 问题的根本解法**，比降 micro_batch/batch_size 更有效

### 修复 9：update_policy 循环内显存碎片（第 14 次 OOM 的根因）
- **文件**：`verl/workers/actor/dp_actor.py`（第 348 行）
- **问题**：update_policy 做 forward + backward（有梯度），每个 micro batch 后碎片累积。第 14 次 micro_batch_for_update=8 在 1/16 时 OOM（softmax 需要 38.88 GiB）
- **修复**：`loss.backward()` 后加 `gc.collect() + torch.cuda.empty_cache()`
- **同时降** `micro_batch_size_per_device_for_update` 从 8 到 2

### 修复 10：liger-kernel fused ops 减少激活显存
- **文件**：`verl/workers/fsdp_workers.py`（第 253 行，`from_pretrained` 之前）
- **版本**：liger-kernel 0.5.10（0.7.0 需要 transformers≥4.52.0，不兼容）
- **代码**：
  ```python
  from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
  apply_liger_kernel_to_qwen2_vl()
  ```
- **效果**：替换 Qwen2.5-VL 的 RMSNorm/SwiGLU/CrossEntropy/RoPE 为 fused kernel，减少中间 activation 显存
- **验证**：8 个 worker 全部打印 `[LIGER] Applied liger-kernel fused ops to Qwen2.5-VL`

### 修复 11：全链路 gc.collect + empty_cache（参考 Awesome-ML-SYS-Tutorial 文章）
- **参考**：https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/torch/mem-snapshot/readme.md
- **问题**：Qwen2.5-VL 的 image processor 在 rollout 阶段产生显存碎片（fast tokenizer 泄漏），Python 引用未回收导致 CUDA tensor 无法释放
- **修复位置**：
  - `fsdp_vllm.py` vllm offload 后：`gc.collect()`（已有 `empty_cache()`）
  - `fsdp_workers.py` compute_log_probs 入口：`gc.collect() + empty_cache()`（data.to(cuda) 之前）
  - `fsdp_workers.py` compute_ref_log_probs 入口：同上
  - `fsdp_workers.py` update_actor 入口：同上
- **原理**：`gc.collect()` 回收 Python 层面的循环引用（image tensor 等），让 CUDA tensor 引用计数归零，之后 `empty_cache()` 才能真正释放显存
- ❌ 2.8.3 ABI 不兼容

---

## 踩坑 & Tricks

### 1. vllm V1 engine OOM
- vllm 0.8.4 默认启用 V1 engine，对 Qwen2.5-VL 的 vision encoder cache profiling 有 bug
- V1 engine 按最大可能图片尺寸（max_pixels=1048576）一次性预留 encoder cache，直接爆显存
- 报错类似：`torch.OutOfMemoryError` 在 vllm 初始化阶段，还没开始 generate
- **解决**：`export VLLM_USE_V1=0`（强制回退到 V0 engine）
- V0 engine 不做激进的 encoder cache 预分配，按需分配

### 2. GPU 0/1 占用是临时的（非永久 Persistence Mode）
- 最初 `nvidia-smi` 显示 GPU 0/1 各占 ~75 GB，但 Processes 栏为空
- 误以为是容器/驱动层的 Persistence Mode 预留，无法释放 → 改用 6 卡（GPU 2-7）
- 后来发现实际是另一个 vllm 推理服务（Holo3-35B-A3B，端口 8010）在跑：
  ```
  /mnt/data/ericdu/conda_envs/seagent/bin/python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data/weights/Holo3-35B-A3B --tensor-parallel-size 2 --port 8010
  ```
  tensor-parallel-size=2 占了 GPU 0+1 各 75 GB
- 该服务停掉后 8 卡全空（每张仅 4-7 MiB）
- **教训**：`nvidia-smi` Processes 栏为空不代表没有进程——可能是其他用户的进程或者 MPS 模式下不显示。用 `ps aux | grep vllm` 或 `fuser -v /dev/nvidia*` 确认
- **每次开训前必须**：`nvidia-smi --query-gpu=index,memory.used --format=csv,noheader` 确认全部 GPU 可用

### 3. 6 卡 batch size 整除问题
- 原配置 `global_batch_size=512` 是给 8 卡设计的（512/8=64 per device）
- 6 卡时 512/6=85.33 不整除 → `ValueError: actor global batch size per device must be divisible by the micro batch size`
- `micro_batch_size_per_device_for_update=8`，需要 `global_batch_size / n_gpus` 能被 8 整除
- **解决**：`global_batch_size=384`（384/6=64，每卡不变）
- 同步修改：`data.rollout_batch_size=384`，`data.curriculum_rollout_batch_size=384`
- 8 卡时恢复 512（512/8=64，整除）

### 4. FSDP actor + vllm rollout 共享 GPU 显存
- verl 是 hybrid engine 架构：actor（FSDP 训练）和 rollout（vllm 推理）**交替占用同一组 GPU**
- rollout 阶段：FSDP actor 权重 offload 到 CPU/sleep，vllm 加载模型做 generate
- training 阶段：vllm sleep 释放显存，FSDP actor 恢复做 forward/backward
- 如果两者抢占不彻底，显存会叠加导致 OOM
- 默认 `gpu_memory_utilization=0.75` + cudagraph 占用过多 → rollout 阶段 OOM
- **解决方案（6 卡时代，显存紧张）**：
  - `worker.rollout.enforce_eager=true`（禁用 cudagraph，省 ~2-3 GB per GPU）
  - `worker.rollout.gpu_memory_utilization=0.5`（vllm KV cache 只用 50% 显存）
  - `worker.rollout.max_num_batched_tokens=65536`（减少 batch token 总量，原 196608）
- **8 卡时**：`gpu_memory_utilization=0.75`（论文值），`enforce_eager=true` 仍保留（稳定性）

### 5. 验证集 MathVista 只有 test split
- MathVista 数据集只发布了 `test` 和 `testmini`，没有 `train` split
- `load_dataset(data_path, split="train")` → `ValueError: Unknown split "train". Should be one of ['test']`
- `val_freq=-1` 意味着验证集从不被调用，所以这个错误本不致命
- Ray `fault_tolerance.max_restarts=3` 自动重启后，datasets 库缓存使第二次通过
- **修复**：`ray_trainer.py` 里 val_dataset 创建加 try/except，失败时 fallback 到 train_dataset

### 6. Ray stdout buffer 吞日志（5月2日训练"没有输出"的根因）
- 5月2日第一次训练跑完 curriculum 初始化后，日志在 `Saved curriculum weights` 后戛然而止
- 实际上 `fit()` 大概率跑了 39 个 training step，但 ConsoleLogger 的 `print(f"Step {step}\n" + ...)` 输出全被 Ray actor 的 stdout buffer 吃掉了
- Ray remote actor（Runner）的 stdout 经由 Ray 回传到 driver 进程，有 buffer 延迟
- 进程正常退出后 buffer 里的内容随进程一起丢失，所以 train.log 里什么都没有
- **修复三管齐下**：
  - `export PYTHONUNBUFFERED=1`（Python 全局禁用 stdout buffer）
  - `export RAY_DEDUP_LOGS=0`（禁止 Ray 去重/合并日志）
  - `ConsoleLogger.log()` 末尾加 `import sys; sys.stdout.flush()`

### 7. Curriculum 初始化 ~6 小时
- `_update_curriculum_weights()` 在 `__init__` 里被调用，对全部 14996 条训练数据做 rollout
- 每条 prompt 生成 8 个 rollout（n=8），按 384 batch 分 40 个 batch
- 每个 batch 的 generate 需要 vllm 推理 ~10 分钟，40 batch × 10 min ≈ **6-7 小时**
- 结果保存到 `checkpoints/curriculum_weights/weights_step_0.pt`，下次可直接加载复用
- **快速测试**：代码里把 `_update_curriculum_weights()` 替换为 `self.train_dataset.curriculum_weights = torch.ones(len(self.train_dataset))`，用均匀权重跳过
- ⚠️ 之前一次误操作 `rm -rf checkpoints/*` 把已算好的 weights_step_0.pt 删了

### 8. flash_attn 安装踩坑
- **源码编译会卡死**：`pip install flash-attn` 启动后 CPU 利用率接近 0%，无 .o 文件产出，进程假死在 GPU 架构检测阶段
- **预编译 whl 版本兼容性矩阵**：
  - ❌ `flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp312` → `undefined symbol: _ZN3c105ErrorC2E...`
  - ❌ `flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp312` → 同样 undefined symbol
  - ✅ **`flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312`** → 正常工作
- 来源：https://github.com/Dao-AILab/flash-attention/releases
- 服务器连不了 GitHub → 本地 Mac 下载 whl → `scp -P 51118` 传服务器 → `pip install xxx.whl`
- `torch._C._GLIBCXX_USE_CXX11_ABI` 报 False，但 2.8.3 的 FALSE 版本仍然不兼容（torch 2.6.0+cu124 的 ABI 比较特殊）
- **教训**：预编译 whl 不是版本匹配就能用，必须实际 import 验证

### 9. yaml 默认参数与论文不一致（OOM 根因之一）
- 对比论文 Table 6 和 `examples/mmr1.yaml` 发现两个关键差异：
  - yaml `max_response_length=8192`，论文 **4096** → 每条 response token 数翻倍
  - yaml `micro_batch_size_per_device_for_experience=64`，论文 **32** → 每个 micro batch forward 数据量翻倍
  - 两项叠加 = **4 倍显存需求**
- 其他差异（不影响 OOM 但影响训练效果）：
  - yaml `PPO epochs=1`，论文 **20**
  - yaml `curriculum_update_freq=4`，论文 **56**
- **教训**：开源代码的默认配置不一定和论文一致，必须逐项对比

### 10. compute_log_probs OOM 真正根因
- `fsdp_workers.py:612` 的 `data = data.to(torch.cuda.current_device())` 把**整个 batch 的全部数据**一次性搬到 GPU
- data 包含 `rollout_batch_size × n_rollout / n_gpus` 条样本的所有 tensor
- 配置 batch=512, n=8, 8 卡 → 每卡 512 条 × (2048+4096) tokens 的 input_ids/attention_mask/position_ids/responses + 图片 multi_modal_inputs
- 然后 `compute_log_prob` 内部才按 micro_batch 切分做 forward，但 data 已经全在 GPU 上了
- micro_batch 只控制 forward pass 的峰值，**不控制 data 搬运的总量**
- 论文用 `fsdp_size=8`（8 卡），但论文的 batch=512 在 8 卡上每卡也是 512 条——理论上应该一样 OOM
- **推测论文可能用了更多卡做数据并行（fsdp_size=8 但总卡数 >8），或者有代码层面的显存优化我们没有**
- **当前解法**：降 `rollout_batch_size=256` → 每卡 256 条 → data 搬运量减半
- 详细显存数值分析见 `grpo训练.md`

### 11. Ray 残留进程导致 Raylet 注册失败
- 现象：启动训练后报 `Failed to register worker to Raylet: IOError: End of file`
- 原因：上一次 OOM/kill 后，Ray 的 GCS server、Raylet、dashboard agent 等后台进程没完全退出
- `ray stop` 有时不够，需要暴力清理
- **标准清理流程**：
  ```bash
  pkill -9 -f "ray|verl"
  sleep 3
  ray stop --force
  rm -rf /tmp/ray/
  ```

### 12. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- PyTorch CUDA 分配器默认使用固定大小的 segment，反复分配/释放后产生碎片
- OOM 报错中 `reserved but unallocated` 就是碎片——PyTorch 预留了内存但因为不连续而无法使用
- 未加此参数前：`27.22 GiB reserved but unallocated`（34% 显存浪费在碎片上）
- 加了此参数后：`722 MiB reserved but unallocated`（碎片降到 <1%）
- 加在 `run_grpo.sh` 的 `export` 里，对所有子进程生效
- **注意**：此参数减少碎片但不减少实际显存需求，如果真的不够还是会 OOM

### 13. compute_log_probs 循环内显存碎片累积（OOM 终极根因）
- verl 的 `dp_actor.py` 里 `compute_log_prob()` 循环 N 个 micro batch 做 forward
- 每次 FSDP forward 会 all-gather 完整参数 → forward → reshard，但 PyTorch CUDA allocator 不会立刻还内存
- 碎片逐 micro batch 累积（~4 GB/batch），到第 13 个就填满 80 GB
- `expandable_segments` 只解决 segment 粒度的碎片，不解决 allocator pool 的碎片
- **正解**：循环内加 `log_probs.cpu()` + `torch.cuda.empty_cache()`
- 这是 verl 官方的遗漏——sharding manager 在 rollout↔training 切换时有 `empty_cache()`，但 micro batch 循环内没有
- liger_kernel 的 fused cross-entropy 也能缓解（不 materialize 完整 logits），但需要改 forward 代码

---

## SFT 模型评测基线（RL 训练前）

| 数据集 | 样本数 | Overall | multi_choice | free_form |
|--------|--------|---------|-------------|-----------|
| MathVista testmini | 100 | 51.0% | 53.7% | 47.8% |

---

## 历史运行记录

### 第 16 次（2026-05-03 21:41，当前 ⏳）
- 8 卡 + batch=128 + micro_batch=2 + max_resp=4096
- **liger-kernel 0.5.10** + empty_cache + gc.collect 全面补丁
- compute_log_probs ✅，update_policy 17/64 进行中，无 OOM

### 第 15 次（2026-05-03 21:37）
- 同 14 但加了 micro_batch_for_update=2 + update 循环 empty_cache
- 未跑完即被重启（为了加 liger-kernel）

### 第 14 次（2026-05-03 21:17 ✅→❌）
- 8 卡 + batch=128 + micro_batch=2 + max_resp=4096 + **empty_cache 补丁**
- compute_log_probs **64/64 通过** 🎉（首次突破！）
- compute_ref_log_probs 通过
- **update_policy 1/16 OOM**（micro_batch_for_update=8，需要 38.88 GiB）
- update 循环也需要 empty_cache

### 第 13 次（2026-05-03 20:57）
- 8 卡 + batch=128 + micro_batch=4 + max_resp=4096
- compute_log_probs **13/32（41%）时 OOM**
- `Tried to allocate 23.18 GiB, free 22.85 GiB`（差 0.33 GiB！）
- 每个 micro batch 泄漏 ~3.9 GB（5 GB → 56 GB in 13 batches）
- reserved unused: 15.49 GiB（碎片累积）

### 第 12 次（2026-05-03 20:37）
- 8 卡 + batch=256 + micro_batch=8 + max_resp=4096
- OOM（`Tried to allocate 35.49 GiB, free 18.82 GiB, in_use 60.29 GiB`）

### 第 11 次
- 8 卡 + batch=256 + micro_batch=16 + max_resp=4096
- OOM（需要 70.76 GiB，差 3 GiB）

### 第 10 次
- 8 卡 + batch=512 + micro_batch=8 + max_resp=4096
- OOM（47 GiB，PyTorch 128 GB allocated）

### 第 9 次
- 8 卡 + batch=512 + micro_batch=32 + max_resp=4096
- OOM（122 GiB）

### 第 8 次
- 6 卡 + batch=384 + micro_batch=4 + max_resp=8192
- OOM

### 第 7 次
- 6 卡 + batch=384 + micro_batch=8 + max_resp=8192 + flash_attn
- OOM（26.8 GiB 碎片）

### 第 6 次
- 6 卡 + padding_free=false + micro_batch=4 + max_resp=8192
- OOM（360+ GiB）

### 第 5 次
- 6 卡 + flash_attn 未装 → NameError

### 第 4 次（跳过 curriculum 首次验证）
- fit() 进入 ✅，rollout ✅，但 flash_attn 缺失

### 第 3 次（2026-05-02 原始版本）
- curriculum 完成，fit() 日志被 buffer 吞，checkpoint 未保存
