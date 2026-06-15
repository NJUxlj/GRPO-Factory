# M02: DAPO 算法扩展

> **阶段编号**: M02  
> **对应原里程碑**: M0.5  
> **创建时间**: 2026-06-10  
> **预计工期**: 4-5天  
> **前置阶段**: M01 (GRPO基础框架搭建)

---

## 1. 阶段定位

本阶段在 M01 的 GRPO 基础框架上，扩展 **DAPO (Decoupled Clip + Dynamic Sampling + Overlong Shaping)** 算法的三大核心创新：

1. **非对称 Clip**：解耦上下界（ε_low=0.2, ε_high=0.28），允许更大的正向更新
2. **Dynamic Sampling**：过滤全 0/全 1 的 trivial group，提升训练效率
3. **Overlong Reward Shaping**：对接近长度上限的输出施加渐进惩罚

DAPO 适用于长 CoT（Chain-of-Thought）推理任务，能显著提升训练稳定性和样本利用率。

---

## 2. 阶段目标

### 2.1 业务目标

- 支持长文本推理任务（数学证明、代码生成等）的 RLHF 训练
- 通过 Dynamic Sampling 减少无效样本的梯度计算，提升训练效率 20%-30%

### 2.2 技术目标

- 实现 `compute_dapo_loss`（非对称 clip）
- 实现 `filter_trivial_groups`（Dynamic Sampling 过滤）
- 实现 `apply_overlong_penalty`（Overlong Reward Shaping）
- 在 `finetuning_args.py` 中添加 `dapo_*` 参数
- Trainer 中按 `grpo_loss_mode == "dapo"` 分支调用新逻辑
- 创建 `qwen3_lora_dapo.yaml` 配置模板

---

## 3. 核心任务

### 任务 3.1: 实现 DAPO 损失函数 (`loss.py`)

**任务描述**：在 `loss.py` 中新增 `compute_dapo_loss`，实现非对称 clip。

**技术细节**：

```python
def compute_dapo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
) -> torch.Tensor:
    """DAPO: 非对称 clip + token-mean 聚合"""
    ratio = torch.exp(log_probs - ref_log_probs)
    adv = advantages.unsqueeze(-1)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * adv
    token_loss = -torch.min(surr1, surr2)

    return _aggregate_loss(token_loss, mask, "token-mean")
```

**关键差异**：
- GRPO: `torch.clamp(ratio, 1 - 0.2, 1 + 0.2)` （对称）
- DAPO: `torch.clamp(ratio, 1 - 0.2, 1 + 0.28)` （非对称，允许更大的正向更新）
- DAPO 固定使用 `token-mean` 聚合模式

---

### 任务 3.2: 实现 Dynamic Sampling 过滤 (`sampling.py`)

**任务描述**：新建 `sampling.py`，实现 `filter_trivial_groups` 函数。

**技术细节**：

```python
def filter_trivial_groups(
    rewards: torch.Tensor,
    group_size: int,
    metric: str = "acc",
) -> torch.Tensor:
    """DAPO Dynamic Sampling: 过滤全 0 或全 1 的 group，返回有效 mask"""
    num_groups = rewards.shape[0] // group_size
    grouped = rewards.view(num_groups, group_size)

    if metric == "acc":
        binary = (grouped > 0).float()
        group_sum = binary.sum(dim=-1)
        valid = (group_sum > 0) & (group_sum < group_size)
    else:  # metric == "score" or "seq_reward"
        valid = grouped.std(dim=-1) > 1e-8

    return valid.unsqueeze(-1).expand(-1, group_size).reshape(-1)
```

**过滤逻辑**：
- 若 group 内所有 reward 全为 0（全部错误）或全为 1（全部正确），则过滤
- 返回 valid_mask，用于在 trainer 中屏蔽无效样本的梯度

---

### 任务 3.3: 实现 Overlong Reward Shaping (`reward_shaping.py`)

**任务描述**：新建 `reward_shaping.py`，实现 `apply_overlong_penalty` 函数。

**技术细节**：

```python
def apply_overlong_penalty(
    rewards: torch.Tensor,
    response_lengths: torch.Tensor,
    max_response_length: int,
    buffer_len: int = 256,
    penalty_factor: float = 1.0,
) -> torch.Tensor:
    """DAPO Overlong Reward Shaping: 对接近长度上限的输出施加渐进惩罚"""
    threshold = max_response_length - buffer_len
    over = (response_lengths - threshold).clamp(min=0).float()
    penalty = penalty_factor * (over / buffer_len)
    return rewards - penalty
```

**惩罚机制**：
- 当 `response_length > max_response_length - buffer_len` 时开始惩罚
- 惩罚线性增长：`penalty = factor * (over_length / buffer_len)`
- 最终 reward = `original_reward - penalty`

---

### 任务 3.4: 扩展 `finetuning_args.py` 参数

**任务描述**：在 `finetuning_args.py` 中添加 DAPO 特有参数。

**新增参数**：

```python
@dataclass
class FinetuningArguments:
    # ... (M01 已有参数)
    
    # === DAPO 特有参数 ===
    dapo_clip_ratio_low: float = 0.2       # 非对称 clip 下界 ε_low
    dapo_clip_ratio_high: float = 0.28     # 非对称 clip 上界 ε_high
    dapo_dynamic_sampling: bool = True     # 是否启用动态采样
    dapo_filter_metric: Literal["acc", "score", "seq_reward"] = "acc"
    dapo_max_gen_batches: int = 10         # 动态采样最大重试批次
    dapo_overlong_shaping: bool = True     # 是否启用超长惩罚
    dapo_overlong_buffer_len: int = 256    # 超长缓冲区长度
    dapo_overlong_penalty_factor: float = 1.0  # 超长惩罚因子
```

---

### 任务 3.5: Trainer 中集成 DAPO 分支 (`trainer.py`)

**任务描述**：在 `CustomGRPOTrainer.training_step` 中，根据 `grpo_loss_mode` 分发到不同损失函数和逻辑。

**修改内容**：

```python
class CustomGRPOTrainer(Trainer):
    def __init__(self, ref_model, reward_fn, finetuning_args, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        self.reward_fn = reward_fn
        
        # 损失函数路由
        self.loss_fn = {
            "grpo": compute_grpo_loss,
            "dapo": compute_dapo_loss,
        }[finetuning_args.grpo_loss_mode]

    def training_step(self, model, inputs):
        prompts = inputs["input_ids"]
        
        # 1. Rollout
        responses, log_probs, mask = self._rollout(model, prompts)
        
        # 2. Reward
        rewards = self._compute_rewards(prompts, responses)
        
        # 3. DAPO overlong shaping
        if self.grpo_args.grpo_loss_mode == "dapo" and self.grpo_args.dapo_overlong_shaping:
            lengths = mask.sum(dim=-1)
            rewards = apply_overlong_penalty(
                rewards, lengths, self.grpo_args.grpo_max_response_length,
                self.grpo_args.dapo_overlong_buffer_len,
                self.grpo_args.dapo_overlong_penalty_factor,
            )
        
        # 4. Advantage
        advantages = compute_group_relative_advantage(
            rewards, self.grpo_args.grpo_num_generations,
            self.grpo_args.grpo_norm_adv_by_std,
        )
        
        # 5. DAPO 过滤 trivial groups
        if self.grpo_args.grpo_loss_mode == "dapo" and self.grpo_args.dapo_dynamic_sampling:
            valid_mask = filter_trivial_groups(
                rewards, self.grpo_args.grpo_num_generations,
                self.grpo_args.dapo_filter_metric,
            )
            advantages = advantages * valid_mask
        
        # 6. Ref log probs
        with torch.no_grad():
            ref_log_probs = self._get_log_probs(self.ref_model, responses)
        
        # 7. Policy loss (根据 loss_mode 分发)
        loss = self.loss_fn(log_probs, ref_log_probs, advantages, mask,
                            **self._get_loss_kwargs())
        
        # 8. KL loss
        if self.grpo_args.grpo_use_kl_loss:
            kl = self._compute_kl(log_probs, ref_log_probs, mask)
            loss = loss + self.grpo_args.grpo_kl_coef * kl
        
        return loss

    def _get_loss_kwargs(self):
        mode = self.grpo_args.grpo_loss_mode
        if mode == "grpo":
            return {"clip_ratio": self.grpo_args.grpo_clip_ratio,
                    "loss_agg_mode": self.grpo_args.grpo_loss_agg_mode}
        elif mode == "dapo":
            return {"clip_ratio_low": self.grpo_args.dapo_clip_ratio_low,
                    "clip_ratio_high": self.grpo_args.dapo_clip_ratio_high}
```

---

### 任务 3.6: 创建 DAPO 配置模板

**任务描述**：创建 `examples/train_lora/qwen3_lora_dapo.yaml`。

**配置内容**：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct
stage: grpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
grpo_loss_mode: dapo
grpo_num_generations: 16
grpo_max_response_length: 4096
dapo_clip_ratio_low: 0.2
dapo_clip_ratio_high: 0.28
dapo_dynamic_sampling: true
dapo_filter_metric: acc
dapo_max_gen_batches: 10
dapo_overlong_shaping: true
dapo_overlong_buffer_len: 256
dapo_overlong_penalty_factor: 1.0
grpo_use_kl_loss: true
grpo_kl_coef: 0.001
dataset: grpo_math_demo
template: qwen3
cutoff_len: 2048
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1
bf16: true
output_dir: saves/qwen3-4b/lora/dapo
```

**关键变化**：
- `grpo_loss_mode: dapo`
- `grpo_num_generations: 16`（相比 GRPO 的 8，增加采样数以利用 Dynamic Sampling）
- `grpo_max_response_length: 4096`（支持长文本）

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 类型 |
|------|--------|------|------|
| D-M02-01 | DAPO 损失函数 | `src/llamafactory/train/grpo/loss.py` (扩展) | 代码修改 |
| D-M02-02 | 采样与过滤模块 | `src/llamafactory/train/grpo/sampling.py` | 新增代码 |
| D-M02-03 | 奖励整形模块 | `src/llamafactory/train/grpo/reward_shaping.py` | 新增代码 |
| D-M02-04 | Trainer DAPO 分支 | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M02-05 | DAPO 参数定义 | `src/llamafactory/hparams/finetuning_args.py` (扩展) | 代码修改 |
| D-M02-06 | DAPO 配置模板 | `examples/train_lora/qwen3_lora_dapo.yaml` | 配置文件 |

---

## 5. 验收标准

### 5.1 功能验收

- ✅ `grpo_loss_mode=dapo` 时可完成一轮完整训练
- ✅ 日志中可见 Dynamic Sampling 过滤行为（如 `Filtered 3/16 trivial groups`）
- ✅ Overlong Shaping 生效：长 response 的 reward 低于短 response（同等质量下）

### 5.2 代码质量验收

- ✅ 新增函数包含 docstring 和类型注解
- ✅ 非对称 clip 的数值范围在日志中可见（调试信息）

### 5.3 性能验收

- ✅ Dynamic Sampling 过滤至少 10% 的 trivial groups（在 toy 数据集上）
- ✅ 训练速度相比 GRPO 提升 ≥ 5%（因减少无效样本计算）

---

## 6. 依赖关系

### 上游依赖

- **M01 (GRPO)**: 依赖目录结构、参数定义、trainer 骨架、loss.py 基础实现

### 下游依赖

- **M03 (GSPO)**: 依赖本阶段的 loss 路由机制和 `_get_loss_kwargs` 模式
- **M04 (DCPO)**: 依赖本阶段的非对称 clip 思想和 Dynamic Sampling 基础设施

### 并行依赖

- 无

---

## 7. 详细技术规范

### 7.1 DAPO 非对称 Clip 公式

\[
L_{\text{DAPO}} = -\frac{1}{\sum_{i,j} m_{i,j}} \sum_{i,j} m_{i,j} \cdot \min\left(r_{i,j} \cdot A_i, \text{clip}(r_{i,j}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \cdot A_i\right)
\]

其中：
- \(\varepsilon_{\text{low}} = 0.2\)
- \(\varepsilon_{\text{high}} = 0.28\)（允许更大的正向更新）

### 7.2 Dynamic Sampling 过滤条件

对于 group \(g\)，若满足以下条件之一则过滤：

\[
\sum_{r \in g} \mathbb{1}[r > 0] = 0 \quad \text{或} \quad \sum_{r \in g} \mathbb{1}[r > 0] = |g|
\]

即全错或全对的 group 不参与梯度计算。

### 7.3 Overlong Penalty 计算

\[
\text{penalty} = \alpha \cdot \frac{\max(0, \text{length} - (\text{max\_len} - \text{buffer}))}{\text{buffer}}
\]

\[
r_{\text{final}} = r_{\text{original}} - \text{penalty}
\]

其中：
- \(\alpha = 1.0\)（惩罚因子）
- \(\text{buffer} = 256\)
- \(\text{max\_len} = 4096\)

---

## 8. 风险与应对

### 风险 8.1: Dynamic Sampling 过滤过多样本

**风险描述**：若数据集质量差，可能导致大量 group 被过滤，实际 batch_size 过小。

**应对策略**：
- 添加日志监控过滤率（`filtered_groups / total_groups`）
- 若过滤率 > 50%，发出警告并建议调整数据集或关闭 Dynamic Sampling

### 风险 8.2: Overlong Shaping 惩罚过重

**风险描述**：惩罚因子过大可能导致模型倾向于生成过短 response。

**应对策略**：
- 默认 `penalty_factor=1.0` 为保守值
- 在配置模板中提供注释，指导用户根据任务调整

### 风险 8.3: 非对称 Clip 导致策略漂移

**风险描述**：ε_high > ε_low 可能使策略在正向更新时步子过大。

**应对策略**：
- 配合 KL loss 使用（`grpo_kl_coef=0.001`）
- 在日志中监控 `ratio` 的分布，确保大部分 token 在 clip 范围内

---

## 9. 阶段完成 Checklist

- [ ] `loss.py` 新增 `compute_dapo_loss`（非对称 clip）
- [ ] `sampling.py` 实现 `filter_trivial_groups`
- [ ] `reward_shaping.py` 实现 `apply_overlong_penalty`
- [ ] `finetuning_args.py` 新增 `dapo_*` 参数（至少 7 个字段）
- [ ] `trainer.py` 集成 DAPO 分支（loss_fn 路由 + overlong shaping + dynamic sampling）
- [ ] `qwen3_lora_dapo.yaml` 配置模板可运行
- [ ] 日志中可见 Dynamic Sampling 过滤行为
- [ ] 完成一轮完整训练（loss 下降）
- [ ] 在 `/docs/开发进度/` 创建 `M02_完成.md`，记录变更文件与验证结果

---

> **下一步**: 完成 M02 后，进入 **M03: GSPO 算法扩展**（序列级 importance ratio + clip）。
