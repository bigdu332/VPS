# MMR1 RL 训练复现记录

## 服务器信息

- **地址**: `ssh -p 51118 root@115.190.60.96`
- **主机名**: `di-20260414154526-fc5fh`
- **GPU**: 8 × NVIDIA A100-SXM4-80GB（80GB 显存）
- **工作目录**: `/mnt/data/ericdu/mmr1/`
- **⚠️ 只能操作 `/mnt/data/ericdu/`，其他目录是同事的，不能动**

---

## 环境配置

| 组件 | 版本 |
|------|------|
| PyTorch | 2.6.0+cu124 |
| veRL | 0.2.0.dev0（volcengine 版） |
| 当前可用环境 | `base`（verl + torch 在此） |
| 目标专用环境 | `verl-mmr1`（待创建，路径 `/mnt/data/ericdu/conda_envs/verl-mmr1`） |

### 已有 conda 环境

| 环境名 | 路径 | 备注 |
|--------|------|------|
| `base` | `/root/miniconda3` | verl + torch，当前可用 |
| `mmr1` | `/root/miniconda3/envs/mmr1` | 无 torch，不可用于训练 |
| `seagent` | `/mnt/data/ericdu/conda_envs/seagent` | 其他项目 |
| `webagent-r1` | `/mnt/data/ericdu/conda_envs/webagent-r1` | 其他项目 |

### 专用环境（verl-mmr1）✅ 已就绪

路径：`/mnt/data/ericdu/conda_envs/verl-mmr1`

**实现方式**：通过 `.pth` 文件继承 base 的 site-packages，再单独装 verl editable install：
```bash
# base site-packages 继承（已配置）
echo '/root/miniconda3/lib/python3.12/site-packages' \
  > /mnt/data/ericdu/conda_envs/verl-mmr1/lib/python3.12/site-packages/base_inherit.pth

# verl editable install（已完成）
cd /mnt/data/ericdu/mmr1 && pip install -e . --no-deps
```

**激活方式**：
```bash
conda activate /mnt/data/ericdu/conda_envs/verl-mmr1
```

验证通过：torch 2.6.0+cu124 ✅ | verl ✅ | vllm ✅

---

## 训练配置（`examples/mmr1.yaml`）

### 模型
- **Base 模型**: `/mnt/data/ericdu/models/MMR1-3B-SFT`
- **输出路径**: `/mnt/data/ericdu/output/mmr1_3b_rl/checkpoints`

### 数据
- **训练集**: `/mnt/data/ericdu/datasets/MMR1-RL/data`（2 个 parquet）
- **验证集**: `/mnt/data/ericdu/datasets/MathVista/data`
- **Curriculum sampling**: 开启，指标为 learnability

### 算法
- **算法**: GRPO
- **KL 惩罚**: `low_var_kl`，kl_coef=1e-2

### 关键超参

| 参数 | 值 |
|------|-----|
| total_episodes | 10 |
| rollout batch size | 512 |
| GRPO rollout n | 8 |
| actor lr | 1e-6 |
| max prompt length | 2048 |
| max response length | 8192 |
| FSDP size | 8（全卡分片） |
| micro_batch_size_per_device (update) | 8 |
| micro_batch_size_per_device (experience) | 64 |

---

## 启动命令

```bash
cd /mnt/data/ericdu/mmr1
conda activate /mnt/data/ericdu/conda_envs/verl-mmr1
JOB_NAME=mmr1_3b_rl bash examples/mmr1/train_qwen2_5_vl_3b.sh
```

> 注：`train_qwen2_5_vl_3b.sh` 模型路径硬编码为 `MMR1-3B-SFT`，直接用即可。`train_qwen2_5_vl_7b.sh` 里模型路径是 fallback（7B），以 yaml 里的配置为准。

---

## 当前状态（2026-04-30）

- ✅ 数据集已就位（MMR1-RL + MathVista）
- ✅ 模型已就位（MMR1-3B-SFT）
- ✅ 环境快照已导出（`base_env_snapshot.yaml`，304 个包）
- ✅ 专用环境 `verl-mmr1` 已就绪（torch + verl + vllm 均验证通过）
- ❌ Checkpoint 目录为空，历史训练未留下 checkpoint
