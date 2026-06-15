# M06: DCPO 进阶特性（可选）

> **阶段编号**: M06  
> **对应原里程碑**: M2.5  
> **创建时间**: 2026-06-10  
> **预计工期**: 4-5天（可选阶段）  
> **前置阶段**: M04 (DCPO算法扩展), M05 (RewardManager集成)

---

## 1. 阶段定位

本阶段为**可选的深度优化阶段**，在 M04 DCPO 核心实现基础上，探索以下进阶特性：

1. **DCPO + DAPO Dynamic Sampling 组合**：将 DCPO 与 Dynamic Sampling 结合，进一步提升训练效率
2. **DCPO + Overlong Shaping 组合**：支持长文本任务的奖励整形
3. **DCPO + GSPO 序列级 clip_c 思想组合**：探索序列级和 token 级 clip 的混合模式
4. **Megatron 分布式训练支持**：适配 Megatron-LM 框架的分布式训练
5. **与规则化奖励深度集成**：支持 rule-based reward 与 DCPO 的融合

本阶段目标是在数学/代码/工具调用等 benchmark 上达到或超过 DAPO 基线。

---

## 2. 阶段目标

### 2.1 业务目标

- 提供 DCPO 的高级变体，满足特定场景的性能优化需求
- 在复杂任务（多步推理、工具调用、Agent 行为）上超越 DAPO/GSPO

### 2.2 技术目标

- 实现 DCPO + Dynamic Sampling 组合（trainer 集成）
- 实现 DCPO + Overlong Shaping 组合（trainer 集成）
- 探索 DCPO + 序列级 clip_c 混合模式（`dcpo.py` 扩展）
- 实现 Megatron 分布式训练适配（可选）
- 实现 rule-based reward 与 DCPO 深度集成
- 在至少 2 个 benchmark 上验证进阶变体的有效性

---

## 3. 核心任务

### 任务 3.1: DCPO + Dynamic Sampling 组合

**任务描述**：在 trainer 中同时启用 DCPO 和 DAPO 的 Dynamic Sampling。

**技术细节**：

```python
def training_step(self, model, inputs):
    # ... (前面步骤)
    
    # 4. Advantage (DCPO SAS)
    if self.grpo_args.grpo_loss_mode == "dcpo" and self.grpo_args.dcpo_sas_enable:
        advantages = compute_smoothed_advantage(
            rewards, self.grpo_args.grpo_num_generations,
            threshold=self.grpo_args.dcpo_sas_threshold,
        )
    else:
        advantages = compute_group_relative_advantage(...)
    
    # 5. DCPO + Dynamic Sampling 组合
    if self.grpo_args.dapo_dynamic_sampling:
        valid_mask = filter_trivial_groups(
            rewards, self.grpo_args.grpo_num_generations,
            self.grpo_args.dapo_filter_metric,
        )
        advantages = advantages * valid_mask
    
    # ... (后续步骤)
```

**关键设计**：
- DCPO 的 SAS 平滑与 DAPO 的 Dynamic Sampling 正交，可组合使用
- 需确保过滤后的样本仍能满足 DCPO 的 group-relative 计算要求

---

### 任务 3.2: DCPO + Overlong Shaping 组合

**任务描述**：在 trainer 中同时启用 DCPO 和 DAPO 的 Overlong Reward Shaping。

**技术细节**：

```python
def training_step(self, model, inputs):
    # ... (Rollout)
    
    # 2. RewardManager 评分
    rewards = self._compute_rewards(prompts, response_strs, ground_truths)
    
    # 3. DCPO + Overlong Shaping 组合
    if self.grpo_args.dapo_overlong_shaping:
        lengths = mask.sum(dim=-1)
        rewards = apply_overlong_penalty(
            rewards, lengths, self.grpo_args.grpo_max_response_length,
            self.grpo_args.dapo_overlong_buffer_len,
            self.grpo_args.dapo_overlong_penalty_factor,
        )
    
    # ... (后续步骤)
```

**关键设计**：
- Overlong Shaping 在 reward 计算后、advantage 估计前应用
- DCPO 的 SAS 平滑会进一步处理经过 shaped 的 reward

---

### 任务 3.3: DCPO + 序列级 clip_c 混合模式探索

**任务描述**：在 `dcpo.py` 中探索序列级和 token 级 clip 的混合模式。

**技术细节**：

```python
def compute_hybrid_dcpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.16,
    clip_ratio_high: float = 0.20,
    dual_clip_ratio: float = 10.0,
    clip_ratio_c: float = 3.0,
    loss_agg_mode: str = "otm",
    hybrid_mode: str = "token-first",  # "token-first" or "seq-first"
) -> torch.Tensor:
    """DCPO 混合模式: 结合 token 级 DAC 和序列级 clip_c"""
    ratio = torch.exp(log_probs - ref_log_probs)
    adv = advantages.unsqueeze(-1)

    if hybrid_mode == "token-first":
        # 先做 token 级 DAC + Dual Clip
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * adv
        token_loss = -torch.min(surr1, surr2)
        
        # Dual Clip
        neg_adv_mask = (adv < 0).float()
        dual_loss = -dual_clip_ratio * adv
        token_loss = torch.max(token_loss, dual_loss) * neg_adv_mask + \
                     token_loss * (1 - neg_adv_mask)
        
        # 再对聚合后的 loss 施加序列级 clip_c
        seq_loss = _aggregate_loss(token_loss, mask, loss_agg_mode)
        return torch.clamp(seq_loss, -clip_ratio_c, clip_ratio_c)
    
    elif hybrid_mode == "seq-first":
        # 先做序列级 ratio 和 clip_c
        seq_log_ratio = ((log_probs - ref_log_probs) * mask).sum(dim=-1)
        seq_ratio = torch.exp(seq_log_ratio)
        
        surr1 = seq_ratio * advantages
        surr2 = torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * advantages
        surr2 = torch.clamp(surr2, -clip_ratio_c, clip_ratio_c)
        
        return -torch.min(surr1, surr2).mean()
```

**关键设计**：
- `token-first`：先做 token 级 DCPO loss，聚合后再施加序列级 clip_c
- `seq-first`：先做序列级 ratio 和 clip_c，再 mean
- 需通过 ablation 实验验证哪种模式更优

---

### 任务 3.4: Megatron 分布式训练支持（可选）

**任务描述**：适配 Megatron-LM 框架的分布式训练。

**技术要点**：

```python
# trainer.py 中适配 Megatron
def training_step(self, model, inputs):
    # Megatron 分布式上下文
    from megatron.core import mpu
    
    if self.grpo_args.gspo_use_megatron:
        # 使用 Megatron 的 tensor parallel + pipeline parallel
        # ... (适配 rollout 和 loss 计算)
        pass
```

**关键设计**：
- Megatron 的 tensor parallel 会影响 log_probs 的计算方式
- 需在 `mpu.get_tensor_model_parallel_world_size()` 下正确聚合 loss
- 需验证 Dynamic Sampling 在分布式环境下的正确性

---

### 任务 3.5: Rule-based Reward 与 DCPO 深度集成

**任务描述**：支持规则化奖励与 DCPO 的融合。

**技术细节**：

```python
# reward/manager.py 扩展
class RewardManager:
    def __init__(self, finetuning_args):
        # ... (已有逻辑)
        
        # 支持 rule-based reward 组合
        self.use_rule_based = finetuning_args.grpo_use_rule_based_reward
        if self.use_rule_based:
            self.rule_based_fn = load_rule_based_reward(finetuning_args)

    def __call__(self, inputs: List[RewardInput]) -> torch.Tensor:
        # 主评分函数
        scores = self._score_main(inputs)
        
        # 规则化奖励（如格式检查、步骤完整性等）
        if self.use_rule_based:
            rule_scores = self.rule_based_fn(inputs)
            # 加权组合
            scores = 0.7 * scores + 0.3 * rule_scores
        
        return scores
```

**关键设计**：
- rule-based reward 可作为主评分函数的补充（如检查数学推导步骤）
- 加权系数（0.7/0.3）可通过配置调整

---

### 任务 3.6: Ablation 实验脚本

**任务描述**：编写 ablation 脚本，验证 DCPO 三大核心技术的独立效果。

**实验设计**：

| 实验组 | DAC | SAS | OTM | Dynamic Sampling | Overlong Shaping |
|--------|-----|-----|-----|------------------|------------------|
| Baseline (DAPO) | ✗ | ✗ | ✗ | ✓ | ✓ |
| DCPO-DAC | ✓ | ✗ | ✗ | ✓ | ✓ |
| DCPO-SAS | ✗ | ✓ | ✗ | ✓ | ✓ |
| DCPO-OTM | ✗ | ✗ | ✓ | ✓ | ✓ |
| DCPO-Full | ✓ | ✓ | ✓ | ✓ | ✓ |

**脚本示例**：

```python
# scripts/run_ablation.py
import subprocess

configs = {
    "dapo": {"grpo_loss_mode": "dapo", "dapo_dynamic_sampling": True},
    "dcpo-dac": {"grpo_loss_mode": "dcpo", "dcpo_sas_enable": False, "dcpo_loss_agg_mode": "token-mean"},
    "dcpo-sas": {"grpo_loss_mode": "dcpo", "dcpo_clip_ratio_low": 0.2, "dcpo_clip_ratio_high": 0.28, "dcpo_loss_agg_mode": "token-mean"},
    "dcpo-otm": {"grpo_loss_mode": "dcpo", "dcpo_sas_enable": False, "dcpo_clip_ratio_low": 0.2, "dcpo_clip_ratio_high": 0.28},
    "dcpo-full": {"grpo_loss_mode": "dcpo"},
}

for name, config in configs.items():
    # 生成配置文件
    # 运行训练
    # 收集指标 (TCR, RUR, final loss)
    pass
```

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 类型 |
|------|--------|------|------|
| D-M06-01 | DCPO + Dynamic Sampling 集成 | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M06-02 | DCPO + Overlong Shaping 集成 | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M06-03 | DCPO 混合模式损失 | `src/llamafactory/train/grpo/dcpo.py` (扩展) | 代码修改 |
| D-M06-04 | Megatron 适配（可选） | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M06-05 | Rule-based Reward 集成 | `src/llamafactory/train/grpo/reward/manager.py` (扩展) | 代码修改 |
| D-M06-06 | Ablation 实验脚本 | `scripts/run_ablation.py` | 脚本 |
| D-M06-07 | 进阶配置模板 | `examples/train_lora/qwen3_lora_dcpo_advanced.yaml` | 配置文件 |

---

## 5. 验收标准

### 5.1 功能验收

- ✅ DCPO + Dynamic Sampling 组合可正常训练，过滤行为日志可见
- ✅ DCPO + Overlong Shaping 组合可正常训练，长 response reward 被惩罚
- ✅ Ablation 脚本可运行，输出各实验组的 TCR/RUR/final loss 对比表

### 5.2 代码质量验收

- ✅ 混合模式损失函数包含详细 docstring，注明与标准 DCPO 的差异
- ✅ Megatron 适配代码有清晰的注释，说明分布式上下文

### 5.3 性能验收

- ✅ DCPO-Full 在数学/代码 benchmark 上达到或超过 DAPO 基线
- ✅ DCPO-DAC/SAS/OTM 的 ablation 结果与论文趋势一致

---

## 6. 依赖关系

### 上游依赖

- **M04 (DCPO)**: 依赖 DAC/SAS/OTM 核心实现
- **M05 (RewardManager)**: 依赖 RewardManager 提供的评分接口

### 下游依赖

- 无（本阶段为最终可选优化阶段）

### 并行依赖

- 无

---

## 7. 详细技术规范

### 7.1 DCPO + Dynamic Sampling 组合公式

先做 DCPO 的 SAS 平滑优势：
\[
A_g^{\text{smooth}} = \tanh\left(\frac{A_g}{k}\right) \cdot k
\]

再应用 Dynamic Sampling 过滤：
\[
A_g^{\text{final}} = A_g^{\text{smooth}} \cdot \mathbb{1}[\text{group}_g \text{ is valid}]
\]

### 7.2 DCPO + Overlong Shaping 组合流程

1. RewardManager 评分：\(r_{\text{original}}\)
2. Overlong Shaping：\(r_{\text{shaped}} = r_{\text{original}} - \text{penalty}\)
3. SAS 平滑优势：\(A_g = \tanh(\text{group\_relative}(r_{\text{shaped}}) / k) \cdot k\)
4. DCPO Loss：使用 \(A_g\) 计算

### 7.3 DCPO 混合模式对比

| 模式 | 优势 | 劣势 |
|------|------|------|
| `token-first` | 保留 token 级细粒度信息 | 序列级 clip_c 可能过度抑制 |
| `seq-first` | 序列级操作更鲁棒 | 丢失 token 级细粒度信息 |

---

## 8. 风险与应对

### 风险 8.1: 组合特性过多导致配置复杂

**风险描述**：DCPO + Dynamic Sampling + Overlong Shaping + 混合模式可能导致配置参数过多。

**应对策略**：
- 提供预设配置模板（如 `qwen3_lora_dcpo_advanced.yaml`）
- 在文档中提供"推荐组合"表格

### 风险 8.2: Megatron 适配工作量大

**风险描述**：Megatron-LM 的分布式训练涉及大量底层改造。

**应对策略**：
- 本阶段标记为"可选"，优先完成其他进阶特性
- 先完成单卡验证，再逐步扩展到分布式

### 风险 8.3: Ablation 实验计算资源需求大

**风险描述**：5 组 ablation 实验需大量 GPU 时间。

**应对策略**：
- 先用小数据集（如 100 样本）做快速验证
- 仅在有显著提升的实验组上跑完整数据集

---

## 9. 阶段完成 Checklist

- [ ] `trainer.py` 集成 DCPO + Dynamic Sampling 组合
- [ ] `trainer.py` 集成 DCPO + Overlong Shaping 组合
- [ ] `dcpo.py` 实现混合模式损失函数（`compute_hybrid_dcpo_loss`）
- [ ] （可选）Megatron 分布式训练适配完成
- [ ] `reward/manager.py` 支持 rule-based reward 组合
- [ ] `scripts/run_ablation.py` Ablation 实验脚本可运行
- [ ] `qwen3_lora_dcpo_advanced.yaml` 配置模板可运行
- [ ] DCPO-Full 在至少 1 个 benchmark 上超过 DAPO 基线
- [ ] 在 `/docs/开发进度/` 创建 `M06_完成.md`，记录变更文件与验证结果

---

> **总结**: 完成 M06 后，PLAN_01 策略优化算法集成项目全部阶段已完成。建议整理最终文档，发布 Release 版本。
