# verl 框架解读笔记（初学者向）

## 一、背景：为什么要从 SFT 迁移到 RL

- SFT（监督微调）遇到瓶颈时，自然转向 RLHF（基于人类反馈的强化学习）
- 出发点：能否在现有 SFT 框架上"魔改"一下就能跑 RL？

---

## 二、RLHF 的标准流程（PPO）

PPO 有四个模块：**actor / critic / reward / ref**

```
for prompts in dataloader:
    # Stage 1: 生成回答（推理）
    batch = actor.generate_sequences(prompts)

    # Stage 2: 计算训练数据（推理）
    batch = critic.compute_values(batch)
    batch = reference.compute_log_prob(batch)
    batch = reward.compute_reward(batch)
    batch = compute_advantages(batch)

    # Stage 3: 更新模型参数（训练）
    critic_metrics = critic.update_critic(batch)
    actor_metrics = actor.update_actor(batch)
```

- **参数更新只在 Stage 3** 发生，前两个 Stage 都是推理
- 最朴素的实现：每个模块用 `deepspeed.initialize()` 或 `FSDP()` 包一下，重写训练循环（TRL 框架的思路）

**问题**：模型和序列一旦 scale up（7B+），单机 8×80G 就开始 OOM

---

## 三、并行策略基础（Parallelism）

### 3.1 两种切分维度

| 切分方式 | 对应策略 | 说明 |
|---|---|---|
| 切分输入 X（batch 维度） | Data Parallel / Sequence Parallel | 实现简单 |
| 切分权重 W | Tensor Parallel / Pipeline Parallel / Expert Parallel | 实现复杂 |

### 3.2 Data Parallel（DP）标准训练流程

1. **Forward**：各卡用自己的数据独立计算 loss
2. **Backward**：各卡独立计算梯度
3. **Update 前**：`all-reduce` 对各卡梯度求平均，再更新参数

> 本质是分布式的 gradient accumulation。在这种朴素 DP 下，每张卡保存完整的 weight / gradient / optimizer。

### 3.3 ZeRO（DeepSpeed）

在 DP 基础上，对各卡保存的内容做切分以节省显存：

| 阶段 | 每卡只保留 | 通信量 |
|---|---|---|
| ZeRO-1 | 1/n optimizer 参数 | 不变 |
| ZeRO-2 | 1/n gradient | 不变 |
| ZeRO-3 | 1/n model weight | 增加 1.5 倍 |

**ZeRO-3 的代价**：每次 forward/backward 都需要 `all-gather` 获取完整参数，计算后释放。自回归生成时每生成一个 token 都触发一次，通信量极大，因此 ZeRO-3 下 `generate()` 极慢。

**关键区别（ZeRO vs TP/PP）**：
- ZeRO 本质仍是 **Data Parallel**，计算时使用完整参数 + 切分输入
- TP/PP 是 **Model Parallel**，始终只存模型的一部分，计算时用切分参数 + 完整输入
- DP 通信的是模型参数（W）；TP/PP 通信的是中间激活值

---

## 四、SPMD 编程模式

### 4.1 什么是 SPMD

**Single Program Multiple Data**：所有进程运行完全相同的代码，通过环境变量区分各自行为，没有中心调度节点。

```python
# 典型初始化
import os
torch.distributed.init_process_group(backend="nccl")
# 每个进程从环境变量拿到自己的 RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT
torch.cuda.set_device(torch.distributed.get_rank())
```

- `torchrun` 负责设置环境变量，不负责"分配"工作
- `init_process_group` 是阻塞操作，所有进程就绪后才继续

### 4.2 SPMD 的优缺点

| 优点 | 缺点 |
|---|---|
| 无中心控制器，运行高效 | 心智负担重，需考虑不同 rank 的行为差异 |
| 完全 worker 自驱 | 复杂逻辑下容易写出死锁（stuck） |

### 4.3 Tensor Parallel（TP）示例

以 Column TP 为例（2 GPU）：

- 权重矩阵按列切分，每个 rank 持有一半
- 每个 rank 用相同的完整输入计算部分输出
- 通过 `all_gather` 拼接得到完整输出

**TP vs DP 的核心区别**：
- TP 组内各 rank：**输入相同，权重不同**
- DP 组内各 rank：**输入不同，权重相同**

### 4.4 DP ↔ TP 的数据流转换

场景：`world_size=8`，某模型用 `tp=8`，其余模型用 `dp=8`

```
DP → TP：
  各 rank 输入 [bs, d]（内容不同）
  → all-gather 拼接 → [8*bs, d]（所有 rank 内容相同）
  → 送入 TP 模型计算

TP → DP：
  TP 模型输出 [8*bs, d]（所有 rank 相同）
  → 按 batch 维度切分 → 各 rank [bs, d]（内容不同）
  → 送入下一个 DP 模型，避免重复计算
```

---

## 五、Rollout 优化：引入 vLLM

### 5.1 为什么要优化 generate

- ZeRO-3 下，自回归生成每个 token 都触发 all-gather，通信开销巨大
- ZeRO-2 下跑得起来，但 `actor.generate()` 通常占整体耗时的 **50%～80%**
- TRL 的临时方案：`GatheredParameters` 上下文（一次性 gather 全部参数），但生成期间需常驻完整显存

### 5.2 用 vLLM 接管 Rollout

**好处**：
- PagedAttention 等高效推理算子
- KV Cache 内存管理
- 自适应调度
- 训练并行策略与推理并行策略解耦

**代价**：
- 每个 step 训练后需要将 actor 更新的参数同步到 vLLM 模型
- vLLM 原生依赖 Ray，不是完全 SPMD，难以融入现有框架

### 5.3 verl 对 vLLM 的魔改（SPMD 化）

**目标**：脱离 Ray 依赖，用 torchrun 拉起 vLLM

主要改动：

1. **新建 `SPMDGPUExecutor`**：去除 Ray driver 调度逻辑，各 worker 独立执行
2. **分布式初始化**：直接从 `LOCAL_RANK / RANK / WORLD_SIZE` 环境变量初始化，不再由 driver 计算
3. **Worker 输入**：各 worker 接收完全相同的输入，无需 driver 分发
4. **logits 收集**：原逻辑 gather 到 driver，改为 `all-gather` 让所有 worker 保存完整 logits

### 5.4 显存管理：KV Cache

vLLM 会预先 allocate 显存用于 KV Cache：

| | 原生 vLLM | verl 改进版 |
|---|---|---|
| KV Cache 配额 | `total × util - dummy_usage` | `(total - dummy_usage) × util` |

verl 的改动更合理：`util` 参数表示"剩余显存的比例"，不需要手动估算其他模型的占用。

- `init_cache_engine()`：在 generate 开始前分配 KV Cache
- `free_cache_engine()`：generate 结束后释放，节省显存给 stage2/3

---

## 六、FSDP + vLLM 参数同步（FSDPVLLMShardingManager）

### 6.1 整体流程

`FSDPVLLMShardingManager` 是一个 context manager，包裹 rollout 的 generate 过程：

```
__enter__：
  1. 从 FSDP state_dict 获取完整参数（full_tensor()）
  2. 通过 weight_loader 按 TP rank 切分，复制到 vLLM 模型
  3. 释放临时显存（del + empty_cache）

generate...

__exit__：
  1. offload vLLM 模型权重到 CPU
  2. empty_cache 释放显存
```

### 6.2 参数同步的三个难点

1. **命名不匹配**：vLLM 内部的参数命名与 HuggingFace 不同，需要 hard-coded 的 weight loader（如 `llama_dtensor_weight_loader`）做对齐
2. **切分规则不同**：FSDP（ZeRO-3）按 1/8 随机切分；vLLM 需按 TP 规则切分（行/列切分）
3. **完整参数重建**：FSDP 切分的 tensor 不规整，需先 `full_tensor()` 获取完整参数再按 TP 规则切分

### 6.3 ColumnParallelLinear / RowParallelLinear 的 weight_loader

```python
# ColumnParallelLinear：沿 output 维度（dim=0）切分
shard_size = param_data.shape[0]
loaded_weight = loaded_weight.narrow(0, tp_rank * shard_size, shard_size)

# RowParallelLinear：沿 input 维度（dim=1）切分
shard_size = param_data.shape[1]
loaded_weight = loaded_weight.narrow(1, tp_rank * shard_size, shard_size)
```

### 6.4 其他细节

**随机状态管理**：TP group 内所有 rank 在 sampling 时必须用相同随机数，否则各 rank 采样到不同 token，后续计算无意义。verl 为同一 dp rank 内的所有 tp rank 设置相同 seed。

**数据流处理**（world_size=8，vLLM tp=4，dp=2）：
- 其他模型 dp=8，各 rank 持有不同的 batch（bs=4）
- 进入 vLLM 前：tp group 内数据 `all-gather` 拼接，bs=4×4=16，作为相同输入
- vLLM 输出后：结果切分为 4 份 bs=4，返回各 dp rank

**offload 策略**：
- vLLM 模型初始化时即 offload 到 CPU（`offload_model_weights`）
- 按需加载 → 生成完成 → 立即卸载 → `empty_cache`
- 目的：为 stage2/3 的 critic/reward/ref 训练腾出显存

---

## 七、总结：相比朴素实现的改进

| 方面 | 朴素实现（TRL 风格）| verl 的改进 |
|---|---|---|
| Rollout 推理 | `actor.generate()`（HF transformers） | vLLM 接管，PagedAttention + KV Cache |
| 框架依赖 | 纯 torchrun + DeepSpeed | SPMD 化 vLLM，保持 torchrun 不引入 Ray |
| 参数同步 | 直接共享模型对象 | FSDP state_dict → weight_loader → TP 分片加载 |
| 显存管理 | 无特殊处理，靠 ZeRO | CPU offload + free cache + empty_cache，精细管理 |
| 并行策略 | 训练/推理共用同一并行配置 | 训练用 FSDP（ZeRO），推理用 vLLM TP，解耦 |
| 数据流 | 简单按 rank 分发 | DP↔TP 转换，保证 TP group 内输入一致 |

---

## 八、Stage 2 & 3 优化

### 8.1 Remove Padding（去除填充）

传统输入是 `[bs, seq_len]`，其中 PAD token 会参与全程计算，纯属浪费。

**做法**：把输入打平成一维 `[total_seqlen]`，去掉所有 PAD token。

```
原始：[bs=2, seq_len=7]  含 PAD
     [T T T _ _ _ _]
     [T T T T _ _ _]

打平：[T T T T T T T]   total_seqlen=7，去掉 4 个 PAD
cu_seqlens = [0, 3, 7]  每个样本实际长度的累加和
```

- **embedding / lm_head / mlp**：token 间无交互，打平无影响
- **attention**：用 `flash_attn_varlen_func` 替换 `flash_attn_func`，传入 `cu_seqlens`
- **RoPE**：通过 `position_id_unpad` 处理每个 token 的位置编码

**收益**：在 stage2/3 尤其明显，因为 rollout 结果同时包含 prompt 的左 padding 和 response 的右 padding，实际利用率很低。

---

### 8.2 Logits 重计算（为何不复用 generate 的 logits）

verl 在 stage2 会强制重算一遍 logprobs，不使用 vLLM generate 返回的结果，原因有二：

1. **后处理污染**：generate 时若用了 penalty 等采样策略，返回的 logits 已被修改，不是模型原始输出
2. **算子差异**：vLLM 的激进算子优化可能导致与 actor model 有微小 diff，会给不稳定的 RL 训练引入额外噪声

> 多算一次 forward 当保险，时间大头在 generate，这点开销可以接受。

---

### 8.3 Logits 显存优化

`compute_log_prob` 输出 logits，shape 为 `[total_seqlen, vocab_size]`。

**问题**：bs=16，avg_len=1000，vocab_size=150k → logits 占用 **9.6G**（fp32），即使是小模型也容易 OOM。

**早期写法的问题**：
```python
logp = F.log_softmax(logits, dim=-1)  # shape 和 logits 一样，双倍显存
logpy = gather_from_labels(logp, labels)
```

**verl 的解法**：
- 用 `flash_attn` 的 `cross_entropy`，把 log_prob 当 `-cross_entropy_loss` 来算
- 或用恒等式 `log_softmax(x_i) = x_i - logsumexp(x)`，两个操作都是一维 tensor

---

### 8.4 Dynamic Batch Size（动态批大小）

**问题**：即使用了 micro_batch，remove padding 后各 micro_batch 的 token 数量仍不均衡，极端情况下仍会 OOM。

**解法**：切分 batch 的单位从"样本数"改为"token 总数"。

| 参数 | 含义 |
|---|---|
| `use_dynamic_bsz=True` | 启用动态批大小 |
| `max_token_len` / `ppo_max_token_len_per_gpu` | 每个 micro_batch 最多处理多少 token |

每个 micro_batch 的样本数不固定，但 token 总数上限固定，从根本上避免因长序列导致的 OOM。

---

## 九、完整流程梳理（FSDP + vLLM 方案）

verl 推荐 **FSDP + vLLM** 作为主方案：FSDP 不侵入模型结构，ZeRO-3 足以支撑 70B+ 百卡量级，vLLM tp_size 按需调大即可。此外还可以叠加 gradient checkpoint / offload 进一步挤显存。

### 9.1 Init 阶段

**数据分片**：用 `DistributedSampler` 构建 Dataloader，各 rank 拿到不同的数据分区。rollout 以外的模块全部使用 DP，数据分片对它们统一生效。

**模型初始化**：
1. 初始化 transformers 模型
2. 用 liger 的 `_apply_liger_kernel_to_instance` 替换成高性能算子
3. 用 `FSDP()` 包装 actor / critic / ref / reward 四个模型

**HybridEngine 构建**：
- 初始化 vLLM model，传入与 actor 相同的模型结构和参数，确保二者是同一个模型
- 暂时把 vLLM model offload 到 CPU 内存（`offload_model_weights`）
- 初始化 ShardingManager，负责 actor → vLLM 的参数同步和数据处理

### 9.2 Stage 1：Rollout 生成（显存最紧张的阶段）

```
[进入 ShardingManager.__enter__]
① 遍历 actor state_dict，每个 weight 调 full_tensor() 拼出完整参数
② 按 vLLM 命名规范和 TP 切分规则，weight.copy_() 加载到 vLLM model
③ del state_dict + empty_cache，释放临时显存

[数据处理]
④ 当前 dp=8，vLLM tp=4：把 8 份 [bs,d] 的数据在 tp group 内 all-gather，
   拼成 [4*bs, d] 作为 vLLM 的统一输入（TP 组内输入必须相同）

[推理]
⑤ vLLM generate（此时 KV Cache 已分配好）

[资源释放]
⑥ free_cache_engine() 释放 KV Cache 显存
⑦ offload_model_weights() 把 vLLM model 推回 CPU
⑧ empty_cache

[数据切回]
⑨ 把 [4*bs, d] 的生成结果按 tp_size 切分，发回各自的 dp rank [bs, d]

[退出 ShardingManager.__exit__]
```

> GRPO 等需要对同一 prompt 多次采样的算法：stage1 把原始数据 repeat n 份送入 vLLM，stage2 再按 uid 聚合计算同一 prompt 下所有 response 的 mean/var 求 advantage。

### 9.3 Stage 2：数据准备

actor / critic / ref / reward 四个模型分别做推理（不更新参数）：

- **actor**：重新计算 `log_prob`（不复用 vLLM 的结果，原因见 8.2）
- **ref**：计算参考策略的 `log_prob`，用于 KL 约束
- **critic**：计算每个 token 的 value 估计
- **reward**：计算奖励分数

每个模型都用 micro_batch 切分 + `use_dynamic_bsz` 按 token 总数控制批大小，避免大 logits OOM。四个模型可以顺序计算，也可以在显存允许时部分重叠。

### 9.4 Stage 3：模型训练

stage2 已经把所有需要的数据收集齐（log_prob / value / reward / advantage），现在正常做反向传播更新 actor 和 critic。

**三层超参的区别（容易混淆）**：

```
data_batch_size（一个 step 处理多少条样本）
  └─ mini_batch_size（PPO 每次更新用多少样本，一个 step 内可以跑多个 epoch）
       └─ micro_batch_size（实际送进 GPU 的最小单元，多个 micro_batch 做 gradient accumulation）
```

举例：data_batch_size=128，mini_batch_size=64，micro_batch_size=16：
- 每个 step 共 128 条样本，分成 2 个 mini_batch
- 每个 mini_batch 分成 4 个 micro_batch 做 gradient accumulation
- 可以对同一批数据跑多个 epoch（PPO 常见做法）

---

## 十、为什么引入 Ray：Colocate vs Split

### 10.1 两种部署方式对比

| 方式 | 优点 | 缺点 |
|---|---|---|
| **Colocate**（四模型共享同一集群） | 全程 GPU 不空转，无角色间等待 | 每个模型被迫使用更高并行度，通信开销大；并行策略不灵活 |
| **Split**（每个模型独立集群） | 并行度可按角色独立优化，通信更少 | 不同 stage 有角色闲置（GPU 空转） |

> HybridFlow 论文中对 13B 128 卡的最优放置：actor 64 卡，critic 32 卡，ref/reward 各 16 卡。

### 10.2 Split 方案的难题

Split 部署后，角色间数据传输超出了 SPMD 的处理范围：
- SPMD 假设所有进程跑相同程序，但不同角色职责完全不同
- 角色间数据流（stage1 → stage2 → stage3）写起来非常别扭，容易破坏 BP 梯度、触发 NCCL hang

**解法**：引入 Ray 作为"有形的大手"

```
Ray Driver  → 负责数据流编排（轻量算法逻辑）
Ray Worker  → 负责计算流执行（复用 FSDP/Megatron 的 SPMD 逻辑）
```

两者解耦：改算法只改 driver，优化计算只改 worker，互不干扰。

---

## 十一、Ray 的具体实现

### 11.1 为什么不能直接 @ray.remote(num_gpus=4)

```python
@ray.remote(num_gpus=4)
class Actor: pass  # ❌ 本质是单进程，无法用 FSDP/Megatron 等 SPMD 框架
```

需要 init 多个 `num_gpus=1` 的 worker，手动配置分布式环境变量，让它们组成一个 process group。

### 11.2 Ray Worker 的初始化

```python
# torchrun 版本：环境变量由 torchrun 自动配置
torch.distributed.init_process_group(backend="nccl")

# Ray 版本：需要在 options 里手动传入
options = {
    'runtime_env': {
        'env_vars': {
            'WORLD_SIZE': str(num_gpus),
            'RANK': str(rank),
            'MASTER_ADDR': master_addr,
            'MASTER_PORT': str(master_port)
        }
    }
}
worker = MyClass.options(**options).remote(...)
```

### 11.3 Colocate 时的资源管理：RayResourcePool

直接用 `num_gpus=0.5` 无法实现"两个模型各占半张卡"——Ray 会把两个 worker 都塞进同一张卡。

**解法**：`RayResourcePool` + PlacementGroup

```python
# 先把机器资源切成 bundle，每个 bundle = 1 块 GPU
bundles = [{"CPU": N, "GPU": 1} for _ in range(num_gpus)]
pg = placement_group(bundles=bundles, strategy="STRICT_PACK")

# 启动 worker 时指定放在哪个 bundle
ray_options = {
    "placement_group": pg,
    "placement_group_bundle_index": gpu_idx,
    "num_gpus": 0.5  # 同一 bundle 内两个 worker 各占 0.5
}
```

### 11.4 数据流管理：@register 装饰器与 WorkerGroup

**问题**：driver 拿到的是完整 batch，但 DP worker 需要各自一份切片；worker 计算完成后，driver 又需要把各份结果拼回来。每个函数都手写这套 split/gather 很重复。

**verl 的解法**：用 `@register(dispatch_mode)` 给函数打标，在 WorkerGroup 初始化时自动生成包装好的版本。

```python
# ① worker 上的函数只写计算逻辑，打上标记
@register(dispatch_mode=Dispatch.DP_REPLICATED)
def compute_log_prob(self, data):
    return model(data)  # 只关心计算，不管数据从哪来
```

`@register` 本身只是给函数挂了一个 `MAGIC_ATTR`，存了 `dispatch_mode / execute_mode / blocking` 三个值，不改变函数行为。

**真正的魔法在 `_bind_worker_method`**：WorkerGroup 初始化时扫描 worker class 的所有方法，找到带 `MAGIC_ATTR` 的函数，根据 dispatch_mode 选出对应的预制 `dispatch_fn` 和 `collect_fn`，用 `func_generator` 把原始函数替换成完整的数据流版本：

```python
def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)   # 数据切分/分发给各 worker
        output = execute_fn(method_name, *args, **kwargs)    # ray.remote 调用各 worker
        if blocking:
            output = ray.get(output)                          # 等结果（可选异步）
        output = collect_fn(self, output)                     # 结果 gather/合并
        return output
    return func
```

**dispatch_mode 实际做什么（以 DP 为例）**：

```python
# dispatch_fn（DP 模式）：把完整 batch 按 rank 切分
def dp_data_future_process(x):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # 如果收到的是 ObjectRef 列表，先 cat 成完整 tensor
    if isinstance(x, list) and isinstance(x[0], ray.ObjectRef):
        x = torch.cat(ray.get(x), dim=0)
    # 按 rank 切出自己那份
    bs_per_rank = x.shape[0] // world_size
    return x[bs_per_rank * rank : bs_per_rank * (rank + 1)]
```

每种 dispatch_mode 对应不同的数据处理规则（DP 切分、TP 广播、All-to-All 等），verl 把这些都做成"预制菜"，用户只需选模式，不用手写通信逻辑。

**传递 ObjectRef 而非 Tensor**：worker 间传的是 `ray.ObjectRef`（数据引用），`ray.get()` 解引用的动作在 worker 内部发生。这样 driver 可以连续调度多个任务，不会被前一个任务的 `ray.get()` 阻塞——就像下单后不用等外卖到了再点下一单。

### 11.5 Colocated Worker：同一进程跑多个模型

**问题**：Colocate 时 actor 和 ref 同在一块 GPU，但各自是独立的 ray worker（独立进程），进程间显存释放和回收不能协调，offload/empty_cache 的时机很难对齐。

**解法**：`create_colocated_worker_cls` 把需要共享 GPU 的模型合并进同一个进程，让它们在一个进程内分时使用显存。

```python
# 原来：两个独立进程，各自管自己的显存
ActorWorker  →  独立进程A（GPU 0）
RefWorker    →  独立进程B（GPU 0）  ← 显存协调困难

# colocate 后：同一进程，显存统一管理
ActorRefWorker → 同一进程（GPU 0），offload/回收可以精确协调
```

这样做的收益：同一进程内 tensor 的申请和释放对 CUDA allocator 完全可见，`empty_cache()` 能真正释放掉另一个模型 offload 后的空间，不会出现"明明 offload 了但显存没降"的情况。

---

## 十二、关键概念速查（完整版）

| 概念 | 一句话理解 |
|---|---|
| DP | 每张卡有完整模型，处理不同数据，梯度 all-reduce 平均 |
| ZeRO-1/2/3 | DP 的显存优化，依次切分 optimizer/gradient/weight，ZeRO-3 通信最多 |
| TP | 权重切分到多卡，同组输入相同，计算结果拼接 |
| PP | 模型按层切分到多卡，流水线执行 |
| SPMD | 所有进程跑相同代码，靠 rank 区分行为，无中心调度 |
| KV Cache | 存储 attention 中间状态，避免重复计算，vLLM 的核心优化之一 |
| PagedAttention | vLLM 的显存管理机制，类似 OS 的分页，避免 KV Cache 碎片 |
| weight_loader | 将 HF 格式权重转换并按 TP 规则分片加载到 vLLM 模型的适配层 |
| Remove Padding | 去除 PAD token 参与计算，打平为一维，用 flash_attn_varlen_func 处理 |
| cu_seqlens | 各样本实际长度的累加和，remove padding 后替代 attention_mask |
| Dynamic BSZ | 按 token 总数而非样本数切分 micro_batch，彻底避免长序列 OOM |
| Colocate | 多个模型共享同一 GPU 集群，全程无空转但并行度受限 |
| Split | 各模型独占独立集群，并行度灵活但有角色空转 |
| Ray Driver | 负责数据流编排的"有形大手"，不涉及具体计算 |
| Ray Worker | 负责分布式计算的执行单元，内部保持 SPMD 范式 |
| RayResourcePool | 通过 PlacementGroup 实现 GPU 细粒度分配，支持 Colocate |
| @register | verl 的数据流预制装饰器，自动处理 dispatch/collect，避免重复写 gather/split |
| ObjectRef | Ray 的数据引用，跨 worker 传递引用而非实体，实现异步数据流 |
