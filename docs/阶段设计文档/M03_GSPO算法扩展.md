# M03: GSPO 算法扩展

> **阶段编号**: M03  
> **对应原里程碑**: M1  
> **创建时间**: 2026-06-10  
> **预计工期**: 3-4天  
> **前置阶段**: M01 (GRPO基础框架搭建)

---

## 1. 阶段定位

本阶段在 M01 的 GRPO 基础框架上，扩展 **GSPO (Group Sequence-level Policy Optimization)** 算法，核心创新为：

1. **序列级 Importance Ratio**：在整条 response 级别计算 ratio，而非 token 级别
2. **序列级 Clip**：对序列级 ratio 施加 clip，并使用额外的 `clip_ratio_c` 限制极端值
3. **适用于噪声大/长度方差大的任务**：序列级操作对 token 级噪声更鲁棒

GSPO 适合代码生成、长文本匹配等需要整体质量评估的场景。

---

## 2. 阶段目标

### 2.1 业务目标

- 支持序列级策略优化，提升代码生成/长文本任务的训练稳定性
- 提供与 GRPO/DAPO 并行的算法选择，用户可根据任务特性切换

### 2.2 技术目标

- 实现 `compute_gspo_loss`（序列级 ratio + clip_c）
- 在 `finetuning_args.py` 中添加 `gspo_*` 参数
- Trainer 中按 `grpo_loss_mode == "gspo"` 分支调用新逻辑
- 创建 `qwen3_lora_gspo.yaml` 配置模板
- 三种算法（GRPO/DAPO/GSPO）端到端集成测试通过

---

## 3. 核心任务

### 任务 3.1: 实现 GSPO 损失函数 (`loss.py`)

**任务描述**：在 `loss.py` 中新增 `compute_gspo_loss`，实现序列级 importance ratio 和 clip。

**技术细节**：

```python
def compute_gspo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.28,
    clip_ratio_c: float = 3.0,
) -> torch.Tensor:
    """GSPO: 序列级 importance ratio + clip"""
    # 序列级 log ratio: 对 token 级 log ratio 求和
    seq_log_ratio = ((log_probs - ref_log_probs) * mask).sum(dim=-1)
    seq_ratio = torch.exp(seq_log_ratio)  # [batch]

    # 序列级 PPO clip
    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * advantages
    
    # GSPO 特有: 二次 clip 限制极端值
    surr2 = torch.clamp(surr2, -clip_ratio_c, clip_ratio_c)

    loss = -torch.min(surr1, surr2)
    return loss.mean()
```

**关键差异**：
- GRPO/DAPO: token 级别 ratio 和 loss
- GSPO: 序列级别 ratio（对 token log_probs 求和后再 exp），最终返回 batch mean

---

### 任务 3.2: 扩展 `finetuning_args.py` 参数

**任务描述**：在 `finetuning_args.py` 中添加 GSPO 特有参数。

**新增参数**：

```python
@dataclass
class FinetuningArguments:
    # ... (M01/M02 已有参数)
    
    # === GSPO 特有参数 ===
    gspo_clip_ratio_c: float = 3.0         # 序列级 clip 参数（二次 clip 范围）
    gspo_use_megatron: bool = False        # 是否使用 Megatron 策略（预留）
```

**注**：GSPO 复用 `dapo_clip_ratio_low/high` 作为序列级 clip 范围（见 `_get_loss_kwargs`）。

---

### 任务 3.3: Trainer 中集成 GSPO 分支 (`trainer.py`)

**任务描述**：在 `CustomGRPOTrainer` 中扩展 loss_fn 路由和 `_get_loss_kwargs`。

**修改内容**：

```python
class CustomGRPOTrainer(Trainer):
    def __init__(self, ref_model, reward_fn, finetuning_args, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        self.reward_fn = reward_fn
        
        # 损失函数路由（扩展 GSPO）
        self.loss_fn = {
            "grpo": compute_grpo_loss,
            "dapo": compute_dapo_loss,
            "gspo": compute_gspo_loss,
        }[finetuning_args.grpo_loss_mode]

    def _get_loss_kwargs(self):
        mode = self.grpo_args.grpo_loss_mode
        if mode == "grpo":
            return {"clip_ratio": self.grpo_args.grpo_clip_ratio,
                    "loss_agg_mode": self.grpo_args.grpo_loss_agg_mode}
        elif mode == "dapo":
            return {"clip_ratio_low": self.grpo_args.dapo_clip_ratio_low,
                    "clip_ratio_high": self.grpo_args.dapo_clip_ratio_high}
        elif mode == "gspo":
            # GSPO 复用 dapo_clip_ratio_low/high，并新增 clip_ratio_c
            return {"clip_ratio_low": self.grpo_args.dapo_clip_ratio_low,
                    "clip_ratio_high": self.grpo_args.dapo_clip_ratio_high,
                    "clip_ratio_c": self.grpo_args.gspo_clip_ratio_c}
```

---

### 任务 3.4: 创建 GSPO 配置模板

**任务描述**：创建 `examples/train_lora/qwen3_lora_gspo.yaml`。

**配置内容**：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct
stage: grpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
grpo_loss_mode: gspo
grpo_num_generations: 8
grpo_max_response_length: 8192
# GSPO 复用 dapo_clip_ratio_low/high 作为序列级 clip 范围
dapo_clip_ratio_low: 0.2
dapo_clip_ratio_high: 0.28
gspo_clip_ratio_c: 3.0
# GSPO 论文: 因序列级 KL 已被重要性比率稀释, 适当提高 kl_coef 防止策略漂移
grpo_use_kl_loss: true
grpo_kl_coef: 0.1
grpo_entropy_coeff: 0.0
dataset: grpo_math_demo
template: qwen3
cutoff_len: 2048
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1
bf16: true
output_dir: saves/qwen3-4b/lora/gspo
```

**关键变化**：
- `grpo_loss_mode: gspo`
- `grpo_max_response_length: 8192`（GSPO 适合超长文本）
- `grpo_kl_coef: 0.1`（相比 GRPO 的 0.001，提高 100 倍以防止策略漂移）
- `gspo_clip_ratio_c: 3.0`

---

### 任务 3.5: 三种算法端到端集成测试

**任务描述**：编写测试脚本，验证 GRPO/DAPO/GSPO 均可正常训练。

**测试内容**：

```python
# tests/test_grpo_algorithms_e2e.py
import pytest
from llamafactory.train.grpo import run_grpo

@pytest.mark.parametrize("loss_mode", ["grpo", "dapo", "gspo"])
def test_algorithm_e2e(loss_mode):
    """测试三种算法均可完成一轮训练"""
    # 1. 加载对应配置
    config_path = f"examples/train_lora/qwen3_lora_{loss_mode}.yaml"
    
    # 2. 运行训练（仅 1 step）
    # ... (调用 run_grpo)
    
    # 3. 验证 loss 下降
    assert final_loss < initial_loss
```

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 类型 |
|------|--------|------|------|
| D-M03-01 | GSPO 损失函数 | `src/llamafactory/train/grpo/loss.py` (扩展) | 代码修改 |
| D-M03-02 | Trainer GSPO 分支 | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M03-03 | GSPO 参数定义 | `src/llamafactory/hparams/finetuning_args.py` (扩展) | 代码修改 |
| D-M03-04 | GSPO 配置模板 | `examples/train_lora/qwen3_lora_gspo.yaml` | 配置文件 |
| D-M03-05 | 三种算法集成测试 | `tests/test_grpo_algorithms_e2e.py` | 测试代码 |

---

## 5. 验收标准

### 5.1 功能验收

- ✅ `grpo_loss_mode=gspo` 时可完成一轮完整训练
- ✅ 序列级 ratio 的数值范围在合理区间（日志可见）
- ✅ 三种算法（GRPO/DAPO/GSPO）均可正常运行，切换仅需修改 `grpo_loss_mode`

### 5.2 代码质量验收

- ✅ GSPO 损失函数包含 docstring，注明与 GRPO/DAPO 的差异
- ✅ 测试脚本覆盖三种算法的端到端流程

### 5.3 性能验收

- ✅ GSPO 在代码生成任务上，相比 GRPO 的训练稳定性提升（loss 震荡更小）
- ✅ 显存占用与 GRPO 相当（无额外显存开销）

---

## 6. 依赖关系

### 上游依赖

- **M01 (GRPO)**: 依赖目录结构、参数定义、trainer 骨架、loss 路由机制

### 下游依赖

- **M04 (DCPO)**: 依赖本阶段的多算法路由模式和 `_get_loss_kwargs` 设计

### 并行依赖

- 无

---

## 7. 详细技术规范

### 7.1 GSPO 序列级 Importance Ratio

\[
r_{\text{seq}} = \exp\left(\sum_{j=1}^{T} (\log \pi_\theta(y_j|x) - \log \pi_{\text{ref}}(y_j|x)) \cdot m_j\right)
\]

与 token 级 ratio 的区别：
- Token 级：\(r_{t} = \exp(\log \pi_\theta(y_t|x) - \log \pi_{\text{ref}}(y_t|x))\)
- 序列级：先对 log ratio 求和，再 exp（等价于 token ratio 的连乘）

### 7.2 GSPO 双重 Clip

第一重 clip（PPO-style）：
\[
r_{\text{clipped}} = \text{clip}(r_{\text{seq}}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})
\]

第二重 clip（GSPO 特有）：
\[
L_{\text{final}} = \text{clip}(-\min(r_{\text{seq}} \cdot A, r_{\text{clipped}} \cdot A), -c, c)
\]

其中 \(c = 3.0\) 为 `clip_ratio_c`。

### 7.3 GSPO KL 系数调整

GSPO 论文建议：因序列级 KL 已被重要性比率稀释，需提高 `kl_coef` 防止策略漂移：

- GRPO: `kl_coef = 0.001`
- GSPO: `kl_coef = 0.1`（提高 100 倍）

---

## 8. 风险与应对

### 风险 8.1: 序列级 Ratio 数值爆炸

**风险描述**：长文本的 token ratio 连乘可能导致数值溢出。

**应对策略**：
- 在 log 空间求和（`seq_log_ratio`），最后再 exp
- 添加数值稳定性检查：`if torch.isinf(seq_ratio): warn(...)`

### 风险 8.2: KL 系数不当导致策略漂移

**风险描述**：GSPO 的 `kl_coef` 需比 GRPO 高 100 倍，用户可能忽略。

**应对策略**：
- 在配置模板中设置默认值 `grpo_kl_coef: 0.1`
- 在日志中打印 `kl_coef` 值，若检测到 `gspo` 模式但 `kl_coef < 0.01`，发出警告

### 风险 8.3: 二次 Clip 过度抑制梯度

**风险描述**：`clip_ratio_c=3.0` 可能过小，导致极端样本被过度截断。

**应对策略**：
- 在日志中监控 `surr2` 被二次 clip 的比例
- 提供配置参数供用户调整

---

## 9. 阶段完成 Checklist

- [ ] `loss.py` 新增 `compute_gspo_loss`（序列级 ratio + clip_c）
- [ ] `finetuning_args.py` 新增 `gspo_*` 参数（至少 2 个字段）
- [ ] `trainer.py` 集成 GSPO 分支（loss_fn 路由 + `_get_loss_kwargs` 扩展）
- [ ] `qwen3_lora_gspo.yaml` 配置模板可运行
- [ ] 三种算法（GRPO/DAPO/GSPO）端到端测试通过
- [ ] 完成一轮完整训练（loss 下降）
- [ ] 在 `/docs/开发进度/` 创建 `M03_完成.md`，记录变更文件与验证结果

---

> **下一步**: 完成 M03 后，进入 **M04: DCPO 算法扩展**（DAC + SAS + OTM Loss 三大创新）。
