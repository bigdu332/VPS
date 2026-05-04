# MMR1 Reward 学习笔记

## 完整调用链

```
ray_trainer.py: fit()
  │
  ├─ reward_fn(batch)                                    # :1292
  │    └─ CustomRewardManager.__call__()                 # workers/reward/custom.py
  │         └─ openr1_compute_score_batch_wo_LMM()       # utils/reward_score/openr1_rewards_batch.py
  │              ├─ accuracy_reward_batch_wo_LMM()
  │              │    └─ r1v_accuracy_reward()            # utils/reward_score/r1v.py  ← 实际判题
  │              ├─ format_reward_batch()
  │              ├─ get_cosine_scaled_reward_batch()      # 当前配置全为0，无效
  │              └─ get_repetition_penalty_reward()
  │
  ├─ batch["token_level_scores"] = reward_tensor         # :1293
  │
  ├─ use_kl_loss=true → reward 不动，直接透传            # :1333
  │
  └─ compute_advantage(..., adv_estimator=GRPO)          # :1349
       └─ core_algos.compute_grpo_outcome_advantage()    # core_algos.py:131
```

---

## Reward 公式

```
overall = accuracy + format + cosine_len(=0) + repetition_penalty

值域：[-0.5, 2.0]
正确+格式对 = 2.0    错误+格式错+严重重复 = -0.5
```

---

## Reward 在 GRPO 中的流转

**1. reward 写成 tensor**（`custom.py:115`）
```python
reward_tensor[i, valid_response_length - 1] = score["overall"]  # 只写最后一个 token，其余全 0
```

**2. KL 分支**（`ray_trainer.py:1333`）：`use_kl_loss=true` → reward 不含 KL，KL 在 `update_actor` 时作为 loss 项加入（`kl_coef=1e-2`）

**3. GRPO advantage**（`core_algos.py:131`）：
```python
scores = token_level_rewards.sum(dim=-1)          # sum ≈ 最后 token 的标量分数
scores[i] = (scores[i] - mean) / (std + eps)      # 组内 z-score（按 prompt uid 分组）
advantages = scores.unsqueeze(-1).tile([1, L]) * eos_mask  # 广播到每个 token
```
8 条都对/都错 → 方差=0 → advantage=0 → 梯度消失（VAS 要解决的问题）

---

## ✅ 文件1：`verl/utils/reward_score/r1v.py`

**实际判题的核心文件**，accuracy reward 的全部逻辑在此。

### 答案候选提取（第322行 `extract_answer_candidates`）

最多取 3 个，按置信度降序：
```
Priority 1: <answer>...</answer> 标签内容
Priority 2: \boxed{} 内容
Priority 3: "the answer is..." 等短语
Priority 4: 最后一句话（兜底，仅在候选不足时触发）
```

### 多层验证（第493行 `r1v_accuracy_reward`）

对每个候选先做 `extract_boxed_content`，再依次：

```
层1 选择题字母（A-E）：两者都提取出字母 → 比较；字母不同 → continue 试下一候选（不是 return 0）
层2 直接字符串比较
层3 normalize() 后比较：去单位/LaTeX/大小写
层4 progressive_verify()：包含关系 + token overlap ≥80%
层5 math_verify：sympy 符号验证，float_rounding_limit 取两者小数位的较小值
```

**层1 用 `continue` 不用 `return 0` 的原因**：字母候选答错了，但同一条输出里可能还有 `\boxed{}` 候选，不应该直接判 0。

**float_rounding_limit 计算**：
- 两边都是浮点数 → `min(pred小数位, gt小数位)`（以精度低的为准）
- 含 π → 固定 2 位
- 其他代数式 → 固定 4 位

**所有候选失败后的兜底**：
- gt 是选择题字母 → **0.1**（有基础猜中概率，完全给 0 不合理）
- 对完整 `predict_str` 做 `progressive_verify` → **0.5**（答对但格式不规范）
- 否则 **0.0**

### normalize()（第165行）

去 `\text{}`/`°`/`\circ`、去中英文单位、混合数 `7 3/4` → `7+3/4`、统一 `\frac`/`\sqrt`/`\pi` 格式、全转小写。

### progressive_verify()（第430行）

三层硬阈值，不是连续相似度分数：
```
Layer 1: normalize_heavy 后精确匹配
Layer 2: 字符级包含（gt≤10字符 或 长度差<20时）
Layer 3: token overlap ≥ 0.8（分母是 gt 词数）；gt≤3词时放宽到 0.6
```
`normalize_heavy` 比 `normalize` 更重，还处理英文数字转阿拉伯数字、分数转小数等。

---

## ✅ 文件2：`verl/utils/reward_score/openr1_rewards_batch.py`

### format_reward_batch（第23行）
```python
pattern_think  = r"^(\s*)<think>.*?</think>"
pattern_answer = r"<answer>.*?</answer>(\s*)$"
# 两者各出现且仅出现一次 → 1.0，否则 0.0
```

**一句话理解**：回答必须以 `<think>` 开始、以 `</answer>` 结束，且中间不能有多余的成对同名标签。

用 `re.findall` 匹配，`len == 1` 才算合格——多一个或少一个都是 0 分：
- `<think>` 出现两次 → 0（重复）
- `<answer>` 没有 → 0（缺失）
- 顺序反了（answer 在 think 前）→ pattern 匹配失败 → 0

### get_repetition_penalty_reward（第108行）
```python
scaling = 1 - len(unique_40grams) / total_40grams   # 重复率 [0,1]
reward = scaling * (-0.5)                            # 范围 [-0.5, 0]
# 不足 40 词 → 直接 0
```

### get_cosine_scaled_reward_batch（第57行）
按长度余弦缩放，**mmr1 配置全设 0.0，此项无效。**

### openr1_compute_score_batch_wo_LMM（第533行）
三个分量加和，返回字典列表：`{"overall": acc+fmt+rep, "accuracy": ..., "format": ..., ...}`

---

## ✅ 文件3：`verl/workers/reward/custom.py`

胶水层，DataProto → reward_tensor。

### `__call__` 做三件事

**① 初始化 reward_tensor**
```python
reward_tensor = torch.zeros_like(data.batch["responses"])  # (bs, response_length)，全 0
```

**② 逐条打分，只写最后一个 token**
```python
valid_response_length = response_mask.sum()
response_str = tokenizer.decode(response_ids[:valid_response_length])
score = self.compute_score(response_str, ground_truth)
reward_tensor[i, valid_response_length - 1] = score["overall"]
```
只写最后一个位置：后续 `sum(dim=-1)` 自然等于这个标量，其余 0 不影响。

**③ 收集统计量**

| 字段 | 内容 | 实际用途 |
|------|------|---------|
| `reward_metrics["accuracy"]` | 每条样本的 0/1 得分列表 | VAS 计算 learnability：reshape 成 `(batch_size, rollout_n)` 算 pass_rate |
| `accuracy_variance` | 上面列表的 `np.var` | 训练监控，log 到 wandb，反映当前 batch 梯度信号质量 |

`accuracy_variance` 接近 0 说明全对或全错，GRPO 组内方差为 0，梯度消失。

### batch_processing 分支（mmr1 实际走这条）

先把全部 response/gt 收集齐 → 一次性批量传入 `openr1_compute_score_batch_wo_LMM`，比逐条调用效率高。

---

## 待学：文件4、5

| 文件 | 位置 | 学习重点 |
|------|------|---------|
| `ray_trainer.py` | :1292, :1333 | reward_fn 返回值写进 batch；use_kl_loss 分支含义 |
| `core_algos.py` | :131（约45行） | GRPO 组内 z-score；scores.sum() 为何等于最后 token 分数 |
