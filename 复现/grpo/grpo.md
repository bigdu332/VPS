# GRPO 训练记录

## 基本信息

| 项目 | 值 |
|------|-----|
| 算法 | GRPO |
| 基座模型 | MMR1-3B-SFT（`/mnt/data/ericdu/models/MMR1-3B-SFT`） |
| 训练数据 | `/mnt/data/ericdu/datasets/MMR1-RL/data`（2 个 parquet） |
| 验证数据 | `/mnt/data/ericdu/datasets/MathVista/data` |
| 框架 | verl 0.2.0.dev0（volcengine 版）+ ray + vllm |
| 服务器 | `ssh -p 51118 root@115.190.60.96`（8× A100-SXM4-80GB） |
| 环境 | `conda activate /mnt/data/ericdu/conda_envs/verl-mmr1` |
| checkpoint 输出 | `/mnt/data/ericdu/output/mmr1_3b_rl/checkpoints` |
| 日志 | `/mnt/data/ericdu/output/mmr1_3b_rl/train.log` |
| TensorBoard | `http://115.190.60.96:6006`（或 SSH 隧道 `ssh -p 51118 -L 6006:localhost:6006 root@115.190.60.96`） |

---

## 启动命令

```bash
cd /mnt/data/ericdu/mmr1
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/data/ericdu/conda_envs/verl-mmr1

# 1 episode 试跑
JOB_NAME=mmr1_3b_rl bash examples/mmr1/train_qwen2_5_vl_3b.sh trainer.total_episodes=1

# 正式 10 episodes
JOB_NAME=mmr1_3b_rl bash examples/mmr1/train_qwen2_5_vl_3b.sh
```

---

## 脚本链路

```
examples/mmr1/train_qwen2_5_vl_3b.sh        # 启动入口，设模型路径、system prompt
└── examples/mmr1.yaml                        # 训练配置（数据、超参、算法、worker）
    └── verl/trainer/main.py                  # 主入口，解析 config 启动 ray
        └── verl/trainer/ray_trainer.py        # 训练主循环（episode → rollout → update）
            ├── verl/workers/fsdp_workers.py          # FSDP actor/ref/critic worker
            ├── verl/workers/actor/dp_actor.py        # Actor 前向/反向
            ├── verl/workers/rollout/vllm_rollout/
            │   └── vllm_rollout_spmd.py              # vllm rollout 生成回答
            ├── verl/workers/reward/custom.py          # reward 调度
            │   └── verl/utils/reward_score/
            │       ├── openr1_rewards.py              # 主 reward 函数
            │       └── math.py                        # 数学答案匹配（boxed 格式）
            ├── verl/trainer/core_algos.py             # GRPO 核心（advantage、KL）
            ├── verl/utils/dataset.py                  # 数据加载（parquet + 图片）
            ├── verl/curriculum/samplers.py             # Curriculum sampling
            └── verl/utils/checkpoint/
                └── fsdp_checkpoint_manager.py          # checkpoint 保存
```

---

## 关键超参（mmr1.yaml）

| 参数 | 值 | 说明 |
|------|-----|------|
| total_episodes | 10（试跑用 1） | 训练轮数 |
| rollout_batch_size | 512 | 每轮生成样本数 |
| GRPO rollout n | 8 | 每个 prompt 生成 8 个回答 |
| actor lr | 1e-6 | 学习率 |
| max_prompt_length | 2048 | prompt 最大 token |
| max_response_length | 8192 | 回答最大 token |
| FSDP size | 8 | 8 卡全分片 |
| kl_penalty | low_var_kl | KL 惩罚方式 |
| kl_coef | 1e-2 | KL 系数 |
| sampling_strategy | curriculum | 课程学习采样 |
| save_freq | 1 | 每 episode 存一次 checkpoint |
| reward | openr1_wo_LMM | 纯规则 reward，不用 LLM judge |

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

## 当前进度（2026-05-02 21:22）

- **状态**：1 episode 试跑中（第 4 次尝试，前 3 次分别因 V1 OOM / batch 整除 / rollout OOM 失败）
- **训练 PID**：83030
- **配置**：6 卡 + V0 engine + enforce_eager + gpu_mem 0.5

### 查看进度

```bash
# 训练日志
tail -f /mnt/data/ericdu/output/mmr1_3b_rl/train.log

# GPU 状态
nvidia-smi

# 进程
ps aux | grep verl | grep -v grep
```

---

## 踩坑 & Tricks

### 1. vllm V1 engine OOM → 用 V0
- vllm 0.8.4 默认 V1 engine，对 Qwen2.5-VL 的 encoder cache profiling 有 bug
- 按最大可能图片尺寸一次性预留 encoder cache，直接爆显存
- **解决**：`export VLLM_USE_V1=0`（强制 V0 engine）

### 2. GPU 0/1 Persistence Mode 占用 ~75GB → 用 6 卡
- `nvidia-smi` 显示 75GB 占用，但 Processes 栏为空
- 是容器/驱动层 Persistence Mode 预留，无法释放
- 训练 FSDP 分片到 GPU 0/1 会 OOM（只剩 ~500MB）
- **解决**：`export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7` + `trainer.n_gpus_per_node=6`

### 3. 6 卡 batch size 整除问题
- 原配置 `global_batch_size=512` 是给 8 卡设计的（512/8=64）
- 6 卡：512/6=85.33 不整除 → `ValueError: actor global batch size per device must be divisible by the micro batch size`
- **解决**：`global_batch_size=384`（384/6=64，每卡不变）
- 同步改：`data.rollout_batch_size=384`，`data.curriculum_rollout_batch_size=384`

### 4. FSDP actor + vllm rollout 共享 GPU 显存打架
- verl 是 hybrid engine：actor（FSDP 训练）和 rollout（vllm 推理）交替占用同一张卡
- 默认 `gpu_memory_utilization=0.75` + cudagraph 占用过多，rollout 阶段 OOM
- **解决**：
  - `worker.rollout.enforce_eager=true`（禁用 cudagraph，省 ~2-3GB）
  - `worker.rollout.gpu_memory_utilization=0.5`（vllm KV cache 只用 50%）
  - `worker.rollout.max_num_batched_tokens=65536`（减少 batch token，原 196608）

### 5. 最终启动命令（6 卡版本）

```bash
cd /mnt/data/ericdu/mmr1
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

JOB_NAME=mmr1_3b_rl bash examples/mmr1/train_qwen2_5_vl_3b.sh \
    trainer.total_episodes=1 \
    trainer.n_gpus_per_node=6 \
    worker.actor.fsdp.fsdp_size=6 \
    worker.actor.global_batch_size=384 \
    data.rollout_batch_size=384 \
    data.curriculum_rollout_batch_size=384 \
    worker.rollout.enforce_eager=true \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.rollout.max_num_batched_tokens=65536
```

---

## SFT 模型评测基线（RL 训练前）

| 数据集 | 样本数 | Overall | multi_choice | free_form |
|--------|--------|---------|-------------|-----------|
| MathVista testmini | 100 | 51.0% | 53.7% | 47.8% |

> RL 训练后用相同 100 条对比，衡量提升效果。
