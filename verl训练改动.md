# veRL 训练流程梳理与改动记录

基于 MMR1 项目实际复现过程，梳理 veRL 0.2.0.dev0 的完整训练流程，并记录为使其在 8× A100-80GB 上跑通 Qwen2.5-VL-3B GRPO 训练所做的所有改动。

---

## 一、一个 GRPO Step 的完整流程

```
ray_trainer.py fit()
  └── for each step:
        ├── generate_sequences()       # fsdp_workers.py:562 → fsdp_vllm.py __enter__/__exit__
        ├── compute_rewards()          # reward/custom.py
        ├── apply_kl_penalty()         # ray_trainer.py:177
        ├── compute_log_probs()        # fsdp_workers.py:617 → dp_actor.py:compute_log_prob()
        ├── compute_ref_log_probs()    # fsdp_workers.py:649 → dp_actor.py:compute_log_prob()
        ├── compute_advantage()        # ray_trainer.py:212 → core_algos.py:compute_grpo_outcome_advantage()
        ├── update_actor()             # fsdp_workers.py:502 → dp_actor.py:update_policy()
        └── _save_checkpoint()         # ray_trainer.py:1062
```

---

## 二、初始化阶段代码流程（`init_model()`）

**入口**：`fsdp_workers.py:406` `init_model()`，由 Ray 在训练开始前调用。

两个 Worker 并行初始化：
- **actor+rollout worker**（`is_actor=True, is_rollout=True`）
- **ref worker**（`is_ref=True`）

两者都调用 `_build_model_optimizer()` → actor worker 额外调用 `_build_rollout()`

### 显存时间线 → 代码对应

---

#### `HuggingFace model init: 1.30 GB` ← actor

`fsdp_workers.py:264` `auto_class.from_pretrained(...)`
打印点：`fsdp_workers.py:304` `print_gpu_memory_usage("After huggingface model init")`

```python
# fsdp_workers.py:261-272
if fsdp_config.enable_rank0_init and self.device_mesh.get_local_rank("fsdp") == 0:
    model = auto_class.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="cpu",          # enable_rank0_init=True → rank0 加载到 CPU
        ...
    )
else:
    with no_init_weights(), init_empty_weights():
        model = auto_class.from_config(...)  # 非rank0只建空壳，不占显存
```

`enable_rank0_init=True`（yaml 默认），rank 0 把模型加载到 CPU，其余 rank 用空壳，GPU 只有框架缓冲区，故 1.30 GB。

---

#### `FSDP module init: 3.72 GB` ← actor

`fsdp_workers.py:338` `FSDP(model, ...)`
打印点：`fsdp_workers.py:351` `print_gpu_memory_usage("After FSDP module init")`

```python
# fsdp_workers.py:338-350
self.fsdp_module = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3：参数/梯度/优化器状态均分片
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision,
    param_init_fn=param_init_fn,       # rank0 → 广播给其余 rank
    device_id=torch.cuda.current_device(),
    sync_module_states=True,           # 把 rank0 CPU 上的权重广播并分片到所有 GPU
    ...
)
```

`FSDP(...)` 做的事：rank0 CPU 权重广播到所有卡，按 FULL_SHARD 每卡只保留 1/8，并分配梯度缓冲区和 flat_param 等内部结构。
3.72 GB = bf16 分片参数（~1 GB）+ FSDP 内部缓冲区。

---

#### `HuggingFace model init: 3.72 GB` ← ref（无增量）

ref worker 独立进程，同样走 `_build_model_optimizer()`，`from_pretrained` 加载到 CPU，GPU 无增量，打印同名日志，显示当前 GPU 总量仍是 3.72 GB。

---

#### `FSDP module init: 7.04 GB` ← ref

同上，ref worker 的 `FSDP(...)` 把 ref 模型分片到 GPU。ref 因 `model.requires_grad_(False)`（`fsdp_workers.py:291`）无梯度缓冲区，比 actor 略小。
7.04 GB = actor 分片（3.72 GB）+ ref 分片（3.32 GB）。

---

#### `Optimizer init: 7.04 GB` ← actor（无增量）

`fsdp_workers.py:355` `self.optimizer = torch.optim.AdamW(...)`
打印点：`fsdp_workers.py:380` `print_gpu_memory_usage("After optimizer init")`

```python
self.optimizer = torch.optim.AdamW(
    self.fsdp_module.parameters(),
    lr=optim_config.lr,
    fused=True,
)
```

AdamW **懒初始化**：创建对象时只注册参数引用，不分配 m/v state。第一次 `optimizer.step()` 才分配 fp32 的 m+v（≈5.7 GB/卡）。所以显存不变。

---

#### `vllm init: 4.62 GB` ← actor offload + vllm sleep

入口：`fsdp_workers.py:384` `_build_rollout()` → `vllm_rollout_spmd.py:83` `LLM(...)`
打印点：`fsdp_workers.py:403` `print_gpu_memory_usage("After vllm init")`

```python
# vllm_rollout_spmd.py:83-103
self.inference_engine = LLM(
    model=model_path,
    gpu_memory_utilization=0.75,
    enable_sleep_mode=True,       # 支持 sleep/wake_up
    ...
)
self.inference_engine.sleep(level=1)  # 初始化后立即 sleep，把模型和 KV cache 释放回 CPU
```

vllm 初始化时加载模型、profiling KV cache，随即 `sleep(level=1)` 全部释放。GPU 回落到只有 FSDP 分片参数的基准值 4.62 GB。

---

## 三、Rollout 阶段代码流程（`fsdp_vllm.py` `__enter__` / `__exit__`）

**调用触发**：`fsdp_workers.py:601`

```python
with self.rollout_sharding_manager:   # 进入 → 触发 __enter__
    ...
    output = self.rollout.generate_sequences(prompts)  # vllm 推理
# 退出 with → 触发 __exit__
```

`FSDPVLLMShardingManager` 是 `fsdp_vllm.py` 里的 context manager，负责 FSDP ↔ vllm 之间的权重同步和显存调度。

---

#### `Before state_dict(): 4.62 GB`

`fsdp_vllm.py:91-92`

```python
def __enter__(self):
    torch.cuda.empty_cache()   # vllm sleep 后清理碎片
    print_gpu_memory_usage("Before state_dict() in sharding manager")
```

此时 vllm 已 sleep，GPU 只有 FSDP 分片参数。Step 2 比 Step 1 高 +3.3 GB，是 Adam m/v state 第一次 `step()` 后的永久增量（懒初始化的一次性开销）。

---

#### `After state_dict(): 8.78 GB`

`fsdp_vllm.py:93-94`

```python
actor_weights = self.module.state_dict()   # FSDP all-gather：聚合所有分片 → 完整参数
print_gpu_memory_usage("After state_dict() in sharding manager")
```

`state_dict()` 触发 FSDP all-gather：逐层把 8 卡分片拼成完整参数，生成一份 bf16 完整参数副本（≈7.5 GB）。
+4.16 GB = 这份临时 state_dict。

---

#### `After sync weights: 60.95 GB`

`fsdp_vllm.py:96-101`

```python
self.inference_engine.wake_up()    # vllm 从 CPU 恢复：加载模型权重 + 预分配 KV cache 池
model = self.inference_engine.llm_engine....model_runner.model
model.load_weights(                # 把刚才 all-gather 的权重写入 vllm 模型
    self._make_weight_iterator(actor_weights)
)
print_gpu_memory_usage("After sync model weights in sharding manager")
```

`wake_up()` 做两件事（一次性完成）：
1. vllm 模型权重（bf16 ≈ 7.5 GB）搬回 GPU
2. 按 `gpu_memory_utilization=0.75` 预分配 KV cache 池（≈80 GB × 0.75 - 其他占用 ≈ 44 GB）

+52 GB = 模型权重 + KV cache 池预分配。

---

#### `After del state_dict: 56.79 GB`

`fsdp_vllm.py:103-107`

```python
del actor_weights          # 释放 all-gather 产生的临时 state_dict 引用
torch.cuda.empty_cache()   # 真正归还 GPU 内存
print_gpu_memory_usage("After del state_dict and empty_cache in sharding manager")
```

−4 GB = 临时 state_dict 释放。vllm 模型权重和 KV cache 池保留。

---

#### `Before vllm offload: 71.04 GB`

`fsdp_vllm.py:114`（`__exit__` 开头）

```python
def __exit__(self, exc_type, exc_value, traceback):
    print_gpu_memory_usage("Before vllm offload in sharding manager")
```

`generate_sequences()` 跑完，KV cache 实际填充（batch=128 × n=8 条长序列）约 14 GB，从 56 → 71 GB。

---

#### `After vllm offload: 4.68 GB` → compute_log_probs 起点

`fsdp_vllm.py:116-123`

```python
self.inference_engine.sleep(level=1)  # vllm CuMemAllocator 全部释放（模型+KV cache）
import gc; gc.collect()               # 回收 Python 循环引用（image processor 等）
self.module.train()                   # FSDP 恢复 train mode
torch.cuda.empty_cache()              # 清理 allocator 碎片
print_gpu_memory_usage("After vllm offload in sharding manager")
```

−66 GB：vllm `sleep(level=1)` 把所有 GPU 内存通过 CuMemAllocator 机制释放回 CPU。
残留 4.68 GB = FSDP 分片参数，此后进入 `compute_log_probs` 阶段。

---

## 四、compute_log_probs / compute_ref_log_probs 阶段

**入口**：`fsdp_workers.py:617` / `fsdp_workers.py:649`

```python
# fsdp_workers.py:620（改动后）
import gc; gc.collect(); torch.cuda.empty_cache()  # 清理 image processor 遗留碎片
data = data.to(torch.cuda.current_device())         # 整个 batch 数据搬上 GPU（含图片 pixel values）
```

**核心 forward**：`dp_actor.py:240-252`

```python
micro_batches = data.split(micro_batch_size)   # 按 micro_batch 切分
for micro_batch in micro_batches:
    log_probs = self._forward_micro_batch(...)  # FSDP forward：all-gather → forward → reshard
    # 改动后：
    log_probs_lst.append(log_probs.cpu())       # 结果立即移 CPU，不在 GPU 累积
    gc.collect()
    torch.cuda.empty_cache()                    # 每批后立即归还碎片
```

每次 `_forward_micro_batch` 内部：FSDP 逐层 all-gather（临时重建完整参数 ~7.5 GB）→ forward → reshard。没有 `empty_cache()` 时碎片按批累积（~3.9 GB/批），13 批后 OOM；加了之后每批回落，峰值稳定 ~28 GB。

---

## 五、update_actor 阶段

**入口**：`fsdp_workers.py:502`

```python
# fsdp_workers.py:504（改动后）
import gc; gc.collect(); torch.cuda.empty_cache()
data = data.to(torch.cuda.current_device())
```

**核心 update**：`dp_actor.py:304-349`

```python
for micro_batch in micro_batches:
    log_probs = self._forward_micro_batch(...)   # forward（有梯度）
    pg_loss, ... = core_algos.compute_policy_loss(...)
    if use_kl_loss:
        kl_loss = ...
        pg_loss = pg_loss + kl_loss * kl_coef
    loss = pg_loss / gradient_accumulation
    loss.backward()                              # backward（gradient checkpointing 保留激活值）
    # 改动后：
    gc.collect()
    torch.cuda.empty_cache()                     # backward 后立即清理
```

update 有梯度 + gradient checkpointing，激活值需在 backward 期间保留，显存压力比 compute_log_probs 大。

---

## 六、日志打印点速查

| 日志关键字 | 文件 | 行号 | 触发函数 |
|-----------|------|------|---------|
| `After huggingface model init` | fsdp_workers.py | 304 | `_build_model_optimizer()` |
| `After FSDP module init` | fsdp_workers.py | 351 | `_build_model_optimizer()` |
| `After optimizer init` | fsdp_workers.py | 380 | `_build_model_optimizer()` |
| `After vllm init` | fsdp_workers.py | 403 | `_build_rollout()` |
| `Before state_dict()` | fsdp_vllm.py | 92 | `__enter__` |
| `After state_dict()` | fsdp_vllm.py | 94 | `__enter__` |
| `After sync model weights` | fsdp_vllm.py | 101 | `__enter__` |
| `After del state_dict` | fsdp_vllm.py | 105 | `__enter__` |
| `Before vllm offload` | fsdp_vllm.py | 114 | `__exit__` |
| `After vllm offload` | fsdp_vllm.py | 119 | `__exit__` |
| `[DEBUG fit]` | ray_trainer.py | 1211 | `fit()` |
| `[CKPT] Saving actor...` | ray_trainer.py | 1068 | `_save_checkpoint()` |

---

## 七、代码改动详情

改动按显存影响从大到小排序。

---

### 改动 1：dp_actor.py — micro_batch 循环内显存碎片修复

**文件**：`verl/workers/actor/dp_actor.py`

### 根因

每次 micro_batch forward 会产生一个巨大的 **logits 张量**：shape 为 `(total_nnz, vocab_size)`，vocab_size=152,064（Qwen2.5-VL 词表），约 12,000 tokens × 152,064 × 2 bytes(bf16) ≈ **3.6 GB**。算完 log_probs 后 logits 被 GC "释放"，但释放只是放进 PyTorch allocator 缓存池——**不归还 CUDA**，`mem_get_info()` 仍将其计入进程占用。

每个 micro_batch 在缓存池里积累 ~3.9 GB（logits ≈ 3.6 GB + 其他中间 tensor），13 批后从 5 GB 涨到 56 GB → OOM。

veRL 官方只在 rollout↔training 切换时调用一次 `empty_cache()`，**micro_batch 循环内没有**，缓存池因此无限累积。

> FSDP 的 all-gather 是**逐层**进行的，每次只重建当前层参数（几十 MB），用完立即 reshard，所有层复用同一块 buffer，不是累积来源。

### 改动

**① 加 `import gc`（第23行）**

```python
import gc
```

**② `compute_log_prob()` 循环末尾（第250-252行）**

```python
# 原代码
log_probs_lst.append(log_probs)

# 改动后
log_probs_lst.append(log_probs.cpu())  # 结果移到 CPU，释放 GPU 显存
gc.collect()
torch.cuda.empty_cache()              # 清除 CUDA allocator 碎片
```

**③ `update_policy()` 循环末尾，`loss.backward()` 之后（第348-349行）**

```python
# loss.backward() 之后加
gc.collect()
torch.cuda.empty_cache()
```

### 效果

- `compute_log_probs`：每次 forward 后 GPU 回落到 ~5 GB，峰值 ~28 GB，64/64 micro_batch 全部通过
- `update_actor`：同理，update_policy 64/64 通过
- 这是 OOM 问题的根本解法。降 micro_batch/batch_size 是治标，碎片按次数累积，不论多小最终都会 OOM

---

### 改动 2：fsdp_workers.py — 函数入口显存清理 + liger-kernel

**文件**：`verl/workers/fsdp_workers.py`

### 根因

Qwen2.5-VL 的 image processor（fast tokenizer）在 rollout 阶段产生大量 Python 循环引用，导致 CUDA tensor 引用计数无法归零。必须先 `gc.collect()` 回收 Python 层循环引用，`empty_cache()` 才能真正释放这部分显存。

### 改动

**① 三处函数入口（`data.to(cuda)` 之前）**

`update_actor`（第504行）、`compute_log_probs`（第620行）、`compute_ref_log_probs`（第651行）各加：

```python
import gc; gc.collect(); torch.cuda.empty_cache()  # clean up before data.to(cuda)
data = data.to(torch.cuda.current_device())
```

**② `_build_model_optimizer()` 加 liger-kernel（第253-259行）**

```python
# Apply liger-kernel fused ops to reduce activation memory
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
    apply_liger_kernel_to_qwen2_vl()
    print("[LIGER] Applied liger-kernel fused ops to Qwen2.5-VL")
except Exception as e:
    print(f"[LIGER] Failed to apply: {e}")
```

替换 Qwen2.5-VL 的 RMSNorm/SwiGLU/CrossEntropy/RoPE 为 fused kernel，减少中间 activation 显存。版本：liger-kernel 0.5.10（0.7.0 需要 transformers≥4.52.0，不兼容当前环境）。

**③ `attn_implementation` 改为 `"eager"`（第268行）**

```python
# 原代码
attn_implementation="flash_attention_2"
# 改动后
attn_implementation="eager"
```

注：flash_attn 通过 whl 单独安装（2.7.4.post1），`from_pretrained` 使用 eager，实际 forward 由 monkey_patch 注入 flash attention。

---

### 改动 3：logger.py — TensorBoard 崩溃修复 + 日志 flush

**文件**：`verl/utils/logger/logger.py`

### 改动

**① `ConsoleLogger.log()` 末尾加 flush（第62行）**

```python
def log(self, data: Dict[str, Any], step: int) -> None:
    print(f"Step {step}\n" + convert_dict_to_str(unflatten_dict(data)))
    import sys; sys.stdout.flush()  # 新增
```

原因：Ray remote actor 的 stdout 有 buffer，进程退出时缓冲内容丢失。需配合 `PYTHONUNBUFFERED=1` + `RAY_DEDUP_LOGS=0` 使用。

**② `TensorBoardLogger.__init__()` 加 try/except + 类型过滤（第80-91行）**

```python
try:
    flat_config = flatten_dict(config)
    safe_config = {}
    for k, v in flat_config.items():
        if isinstance(v, (int, float, str, bool)):
            safe_config[k] = v
        else:
            safe_config[k] = str(v) if v is not None else "None"
    self.writer.add_hparams(safe_config, metric_dict={})
except Exception as e:
    print(f"Warning: Failed to log hparams to TensorBoard: {e}")
```

原因：`add_hparams()` 只接受 `int/float/str/bool`，传入 `list/dict` 直接 crash，导致整个 training 进程退出。

---

### 改动 4：ray_trainer.py — debug 打印 + 容错逻辑

**文件**：`verl/trainer/ray_trainer.py`

### 改动

**① `fit()` 入口 debug print（第1211行）**

```python
print(f"[DEBUG fit] Starting fit(). training_steps={self.training_steps}, global_step={self.global_step}, len(dataloader)={len(self.train_dataloader)}")
```

原因：定位 `fit()` 是否真正进入。Ray buffer 导致日志丢失时，此 print 是判断训练是否启动的基准。

**② `_save_checkpoint()` 前后加 print（第1068-1070行）**

```python
print(f"[CKPT] Saving actor checkpoint to {actor_path}...")
self.actor_rollout_wg.save_checkpoint(actor_path)
print(f"[CKPT] Actor checkpoint saved successfully.")
```

**③ val_dataset 创建加 try/except fallback（第437-453行）**

```python
try:
    self.val_dataset = RLHFDataset(data_path=self.config.data.val_files, ...)
except (ValueError, Exception) as e:
    print(f"Warning: Failed to load val_dataset: {e}. Using train_dataset as fallback.")
    self.val_dataset = self.train_dataset
```

原因：MathVista 只有 `test` split，`split="train"` 会 `ValueError`。`val_freq=-1` 时验证集从不被调用，不应影响训练启动。

**④ curriculum_weights uniform fallback（第388-390行）**

```python
else:
    print("No curriculum state found, using UNIFORM weights (fast test mode)")
    import torch
    self.train_dataset.curriculum_weights = torch.ones(len(self.train_dataset))
```

原因：`_update_curriculum_weights()` 在 `__init__` 里对全部 ~15000 条数据做 rollout，耗时 6-7 小时。快速测试时用 uniform weights 跳过，正式训练时需先单独运行 curriculum 初始化或加载已保存的 `weights_step_0.pt`。

---

### 环境配置（run_grpo.sh）

非代码改动，但直接影响能否跑通：

```bash
export VLLM_USE_V1=0
# vllm 0.8.4 默认启用 V1 engine，对 Qwen2.5-VL 的 vision encoder cache profiling 有 bug，
# 按最大可能图片尺寸一次性预留 encoder cache，直接 OOM。强制回退 V0 engine。

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 减少 segment 粒度显存碎片（reserved but unallocated 从 27 GB 降到 <1 GB）。
# 注意：只减少碎片，不减少实际显存需求，真正不够还是会 OOM。
```

### yaml 关键参数调整

| 参数 | yaml 默认值 | 修改为 | 影响 |
|------|-----------|--------|------|
| `max_response_length` | 8192 | **4096** | token 数减半，forward 显存减半 |
| `micro_batch_size_per_device_for_experience` | 64 | **2** | 每个 micro_batch token 数大幅减少（empty_cache 补丁后可适当调大） |
| `micro_batch_size_per_device_for_update` | 8 | **2** | update 时有梯度，显存压力更大 |
| `worker.rollout.enforce_eager` | false | **true** | 禁用 cudagraph，省 ~2-3 GB/卡（稳定性） |

---

### 改动总览

| 文件 | 改动类型 | 解决的问题 |
|------|---------|----------|
| `workers/actor/dp_actor.py` | 显存管理 | micro_batch 循环碎片累积 → OOM（**根本解法**） |
| `workers/fsdp_workers.py` | 显存管理 + 优化 | image processor 循环引用 + activation 显存 |
| `utils/logger/logger.py` | 日志修复 | TensorBoard crash + Ray buffer 丢日志 |
| `trainer/ray_trainer.py` | 容错 + 调试 | val_dataset crash + curriculum 初始化耗时 + 日志可见性 |
| `run_grpo.sh` / yaml | 配置 | vllm V1 engine bug + 显存碎片 + forward 显存总量 |
