# M04: DCPO 算法扩展（DAC + SAS + OTM Loss 三大创新）

> **阶段编号**: M04  
> **对应原里程碑**: M2  
> **创建时间**: 2026-06-10  
> **预计工期**: 6-8天  
> **前置阶段**: M01 (GRPO基础框架搭建), M02 (DAPO算法扩展)

---

## 1. 阶段定位

本阶段在 M02 的 DAPO 基础上，扩展 **DCPO (Decoupled Clip Policy Optimization)** 算法的三大核心创新：

1. **DAC (Dynamic-Adaptive Clipping)**：收紧的非对称 clip 范围（ε_low=0.16, ε_high=0.20），配合阈值调度器
2. **SAS (Smooth Advantage Standardization)**：使用 tanh 平滑替代硬截断，提升训练稳定性
3. **OTM Loss (Only Token Mean Loss)**：仅在单条 response 内求 token-mean，再 batch-mean

DCPO 是 DAPO 的进一步改进，在数学/代码等推理任务上 **TCR 下降约 30%、RUR 提升约 25%**。

---

## 2. 阶段目标

### 2.1 业务目标

- 提供最高质量的推理任务 RLHF 训练能力
- 在数学推理（GSM8K/MATH）、代码生成等 benchmark 上超越 DAPO 基线

### 2.2 技术目标

- 实现 `compute_dcpo_loss`（DAC 非对称 clip + Dual Clip + OTM Loss）
- 实现 `compute_smoothed_advantage`（SAS tanh 平滑优势）
- 在 `dcpo.py` 中实现 DAC 阈值调度 + Dual Clip + OTM 聚合工具函数
- 在 `finetuning_args.py` 中添加 `dcpo_*` 参数
- Trainer 中注册 `"dcpo": compute_dcpo_loss` 分支 + SAS/DAC 分发逻辑
- 新增 `_get_dac_clip_ratios` 方法（constant / linear_decay 调度）
- 创建 `qwen3_lora_dcpo.yaml` 配置模板
- 单元测试：`compute_dcpo_loss` 对照论文公式数值一致性
- 单元测试：`compute_smoothed_advantage` 与硬截断等价性 / 边界行为

---

## 3. 核心任务

### 任务 3.1: 实现 DCPO 损失函数 (`loss.py`)

**任务描述**：在 `loss.py` 中新增 `compute_dcpo_loss`，实现 DAC + Dual Clip + OTM Loss。

**技术细节**：

```python
def compute_dcpo_loss(
    log_probs: torch.Tensor,          # [batch, seq_len]
    ref_log_probs: torch.Tensor,      # [batch, seq_len]
    advantages: torch.Tensor,         # [batch]  (经 SAS 平滑)
    mask: torch.Tensor,               # [batch, seq_len]
    clip_ratio_low: float = 0.16,
    clip_ratio_high: float = 0.20,
    dual_clip_ratio: float = 10.0,
    loss_agg_mode: str = "otm",
) -> torch.Tensor:
    """DCPO: DAC 非对称 clip + Dual Clip + OTM Loss 聚合
    1) DAC:  ε_low=0.16, ε_high=0.20 (论文: 较 DAPO 收紧)
    2) Dual Clip: 对负优势的 token, 取 max(token_loss, -dual_clip_ratio * adv)
    3) OTM Loss: per-response token-mean → batch-mean
    """
    ratio = torch.exp(log_probs - ref_log_probs)  # [batch, seq_len]
    adv = advantages.unsqueeze(-1)                # [batch, 1]

    # --- (1) DAC: 非对称 clip ---
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * adv
    token_loss = -torch.min(surr1, surr2)

    # --- (2) Dual Clip: 防止负优势下 ratio>1 的极端更新 ---
    if dual_clip_ratio is not None and dual_clip_ratio > 0:
        neg_adv_mask = (adv < 0).float()
        dual_loss = -dual_clip_ratio * adv
        token_loss = torch.max(token_loss, dual_loss) * neg_adv_mask + \
                     token_loss * (1 - neg_adv_mask)

    return _aggregate_loss(token_loss, mask, loss_agg_mode)
```

**关键差异**：
- DAPO: ε_low=0.2, ε_high=0.28
- DCPO: ε_low=0.16, ε_high=0.20（更收紧，减少过度更新）
- Dual Clip: 对负优势 token，限制 loss 上界为 `-dual_clip_ratio * adv`

---

### 任务 3.2: 实现 SAS 平滑优势 (`advantage.py`)

**任务描述**：在 `advantage.py` 中新增 `compute_smoothed_advantage`，使用 tanh 平滑替代硬截断。

**技术细节**：

```python
def compute_smoothed_advantage(
    rewards: torch.Tensor,         # [batch]
    group_size: int,
    threshold: float = 3.0,
) -> torch.Tensor:
    """DCPO SAS (Smooth Advantage Standardization) 平滑优势标准化
    标准 GRPO/DAPO 用 hard clip 把 |adv| 截断到 [-k, k], 但硬截断在边界处
    不平滑, 易引发训练震荡. SAS 用 tanh 平滑近似:
        adv_smooth = tanh(adv / k) * k
    """
    # 先做标准 group-relative 归一化
    advantages = compute_group_relative_advantage(
        rewards, group_size, norm_by_std=True,
    )

    # SAS 平滑: tanh(adv/k) * k
    smoothed = torch.tanh(advantages / threshold) * threshold
    return smoothed
```

**关键优势**：
- 硬截断：`adv = torch.clamp(adv, -k, k)`（一阶不连续）
- SAS 平滑：`adv = tanh(adv/k) * k`（一阶连续可导，优化更稳定）

---

### 任务 3.3: 实现 `dcpo.py` 专属模块

**任务描述**：新建 `dcpo.py`，实现 DAC 阈值调度 + Dual Clip + OTM 聚合工具函数。

**技术细节**：

```python
from typing import Tuple


def get_dac_clip_ratios(
    schedule: str,
    global_step: int,
    max_steps: int,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> Tuple[float, float]:
    """DCPO DAC 阈值调度: constant 或 linear_decay
    
    Args:
        schedule: "constant" 或 "linear_decay"
        global_step: 当前训练步数
        max_steps: 最大训练步数
        clip_ratio_low: 最终 ε_low
        clip_ratio_high: 初始/最终 ε_high (不变化)
    
    Returns:
        (current_clip_low, current_clip_high)
    """
    if schedule == "constant":
        return clip_ratio_low, clip_ratio_high
    elif schedule == "linear_decay":
        # 按训练进度从 (high, high) 线性收敛到 (low, high)
        progress = min(1.0, global_step / max(1, max_steps))
        cur_low = clip_ratio_high - (clip_ratio_high - clip_ratio_low) * progress
        return cur_low, clip_ratio_high
    return clip_ratio_low, clip_ratio_high


def compute_otm_loss(token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """OTM (Only Token Mean) Loss 聚合
    per-response token-mean → batch-mean
    
    与 _aggregate_loss 中的 "seq-mean-token-mean" 数学等价,
    保留为独立函数便于论文对齐。
    """
    lengths = mask.sum(dim=-1).clamp(min=1)
    return ((token_loss * mask).sum(dim=-1) / lengths).mean()
```

---

### 任务 3.4: 扩展 `finetuning_args.py` 参数

**任务描述**：在 `finetuning_args.py` 中添加 DCPO 特有参数。

**新增参数**：

```python
@dataclass
class FinetuningArguments:
    # ... (M01/M02/M03 已有参数)
    
    # === DCPO 特有参数 (DAPO 的进一步改进) ===
    # 1) DAC (Dynamic-Adaptive Clipping)
    dcpo_clip_ratio_low: float = 0.16      # DAC 下界 ε_low (DAPO 默认 0.2, DCPO 收紧)
    dcpo_clip_ratio_high: float = 0.20     # DAC 上界 ε_high (DAPO 默认 0.28, DCPO 收紧)
    dcpo_dual_clip_ratio: float = 10.0     # Dual Clip 上界 r_max (论文 r_max=10)
    
    # 2) SAS (Smooth Advantage Standardization)
    dcpo_sas_enable: bool = True           # 是否启用 SAS
    dcpo_sas_threshold: float = 3.0        # SAS 截断阈值 k (论文用 k=3)
    
    # 3) OTM Loss
    dcpo_loss_agg_mode: Literal[
        "otm", "token-mean", "seq-mean-token-mean"
    ] = "otm"
    
    # 可选: DAC 调度器
    dcpo_clip_schedule: Literal["constant", "linear_decay"] = "constant"
```

---

### 任务 3.5: Trainer 中集成 DCPO 分支 (`trainer.py`)

**任务描述**：在 `CustomGRPOTrainer` 中扩展 loss_fn 路由、优势计算分支和 DAC 调度逻辑。

**修改内容**：

```python
class CustomGRPOTrainer(Trainer):
    def __init__(self, ref_model, reward_fn, finetuning_args, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        self.reward_fn = reward_fn
        
        # 损失函数路由（扩展 DCPO）
        self.loss_fn = {
            "grpo": compute_grpo_loss,
            "dapo": compute_dapo_loss,
            "gspo": compute_gspo_loss,
            "dcpo": compute_dcpo_loss,
        }[finetuning_args.grpo_loss_mode]

    def training_step(self, model, inputs):
        # ... (前面步骤同 M02/M03)
        
        # 4. 计算 advantage
        # DCPO: 若启用 SAS, 用 tanh 平滑优势
        if self.grpo_args.grpo_loss_mode == "dcpo" and self.grpo_args.dcpo_sas_enable:
            advantages = compute_smoothed_advantage(
                rewards, self.grpo_args.grpo_num_generations,
                threshold=self.grpo_args.dcpo_sas_threshold,
            )
        else:
            advantages = compute_group_relative_advantage(
                rewards, self.grpo_args.grpo_num_generations,
                self.grpo_args.grpo_norm_adv_by_std,
            )
        
        # ... (DAPO 过滤逻辑)
        
        # 5.5 DCPO: DAC 阈值调度
        if self.grpo_args.grpo_loss_mode == "dcpo":
            self._current_dac_clip = self._get_dac_clip_ratios()
        else:
            self._current_dac_clip = None
        
        # ... (后续步骤)

    def _get_loss_kwargs(self):
        mode = self.grpo_args.grpo_loss_mode
        # ... (GRPO/DAPO/GSPO 分支)
        elif mode == "dcpo":
            # 优先使用 DAC 调度器产出的 ε_low/ε_high
            if getattr(self, "_current_dac_clip", None) is not None:
                clip_low, clip_high = self._current_dac_clip
            else:
                clip_low = self.grpo_args.dcpo_clip_ratio_low
                clip_high = self.grpo_args.dcpo_clip_ratio_high
            return {"clip_ratio_low": clip_low,
                    "clip_ratio_high": clip_high,
                    "dual_clip_ratio": self.grpo_args.dcpo_dual_clip_ratio,
                    "loss_agg_mode": self.grpo_args.dcpo_loss_agg_mode}

    def _get_dac_clip_ratios(self):
        """DCPO DAC 阈值调度: constant 或 linear_decay"""
        from .dcpo import get_dac_clip_ratios
        return get_dac_clip_ratios(
            schedule=self.grpo_args.dcpo_clip_schedule,
            global_step=self.state.global_step,
            max_steps=self.state.max_steps,
            clip_ratio_low=self.grpo_args.dcpo_clip_ratio_low,
            clip_ratio_high=self.grpo_args.dcpo_clip_ratio_high,
        )
```

---

### 任务 3.6: 创建 DCPO 配置模板

**任务描述**：创建 `examples/train_lora/qwen3_lora_dcpo.yaml`。

**配置内容**：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct
stage: grpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

# === 算法选择 ===
grpo_loss_mode: dcpo

# === Rollout 参数 (与 DAPO 一致: G=16) ===
grpo_num_generations: 16
grpo_temperature: 1.0
grpo_top_p: 1.0
grpo_top_k: -1
grpo_max_response_length: 4096

# === DCPO 三大核心技术 ===
# 1) DAC: 收紧的非对称 clip 范围
dcpo_clip_ratio_low: 0.16
dcpo_clip_ratio_high: 0.20
dcpo_dual_clip_ratio: 10.0
dcpo_clip_schedule: constant

# 2) SAS: 平滑优势标准化
dcpo_sas_enable: true
dcpo_sas_threshold: 3.0

# 3) OTM Loss
dcpo_loss_agg_mode: otm

# === KL 损失 ===
grpo_use_kl_loss: true
grpo_kl_coef: 0.001
grpo_kl_type: kl
grpo_norm_adv_by_std: true

# === 训练超参数 ===
dataset: grpo_math_demo
template: qwen3
cutoff_len: 2048
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
num_train_epochs: 1
bf16: true
output_dir: saves/qwen3-4b/lora/dcpo
```

---

### 任务 3.7: 单元测试 - DCPO 损失函数

**任务描述**：编写 `tests/test_dcpo_loss.py`，验证 `compute_dcpo_loss` 对照论文公式数值一致性。

**测试内容**：

```python
# tests/test_dcpo_loss.py
import torch
from llamafactory.train.grpo.loss import compute_dcpo_loss


def test_dcpo_loss_basic():
    """测试 DCPO 损失函数基本功能"""
    batch_size = 16
    seq_len = 32
    
    log_probs = torch.randn(batch_size, seq_len)
    ref_log_probs = torch.randn(batch_size, seq_len)
    advantages = torch.randn(batch_size)
    mask = torch.ones(batch_size, seq_len)
    
    loss = compute_dcpo_loss(
        log_probs, ref_log_probs, advantages, mask,
        clip_ratio_low=0.16, clip_ratio_high=0.20,
        dual_clip_ratio=10.0, loss_agg_mode="otm",
    )
    
    assert loss.shape == torch.Size([])  # scalar
    assert torch.isfinite(loss)


def test_dcpo_dual_clip():
    """测试 Dual Clip 对负优势的限制"""
    # 构造负优势样本
    advantages = -torch.ones(8)
    # ... (验证 token_loss 被 dual_clip_ratio 限制)
```

---

### 任务 3.8: 单元测试 - SAS 平滑优势

**任务描述**：编写 `tests/test_sas_advantage.py`，验证 SAS 与硬截断的等价性和边界行为。

**测试内容**：

```python
# tests/test_sas_advantage.py
import torch
from llamafactory.train.grpo.advantage import compute_smoothed_advantage


def test_sas_vs_hard_clip():
    """测试 SAS 平滑与硬截断的近似等价性"""
    rewards = torch.randn(64)
    group_size = 8
    threshold = 3.0
    
    sas_adv = compute_smoothed_advantage(rewards, group_size, threshold)
    
    # 验证 |adv| <= threshold
    assert (sas_adv.abs() <= threshold + 1e-6).all()
    
    # 验证小优势时 SAS ≈ 线性
    small_adv = torch.tensor([0.1, -0.2, 0.05])
    # ... (验证 tanh 近似线性)


def test_sas_boundary():
    """测试 SAS 在边界处的平滑性"""
    # 验证 tanh 平滑在 threshold 处一阶连续
    # ... (数值导数 vs 解析导数)
```

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 类型 |
|------|--------|------|------|
| D-M04-01 | DCPO 损失函数 | `src/llamafactory/train/grpo/loss.py` (扩展) | 代码修改 |
| D-M04-02 | SAS 平滑优势 | `src/llamafactory/train/grpo/advantage.py` (扩展) | 代码修改 |
| D-M04-03 | DCPO 专属模块 | `src/llamafactory/train/grpo/dcpo.py` | 新增代码 |
| D-M04-04 | Trainer DCPO 分支 | `src/llamafactory/train/grpo/trainer.py` (扩展) | 代码修改 |
| D-M04-05 | DCPO 参数定义 | `src/llamafactory/hparams/finetuning_args.py` (扩展) | 代码修改 |
| D-M04-06 | DCPO 配置模板 | `examples/train_lora/qwen3_lora_dcpo.yaml` | 配置文件 |
| D-M04-07 | DCPO 损失单元测试 | `tests/test_dcpo_loss.py` | 测试代码 |
| D-M04-08 | SAS 优势单元测试 | `tests/test_sas_advantage.py` | 测试代码 |

---

## 5. 验收标准

### 5.1 功能验收

- ✅ `grpo_loss_mode=dcpo` 时可完成一轮完整训练，loss 正常下降
- ✅ SAS 平滑优势生效：日志中可见 `adv_smooth` 的分布比硬截断更平滑
- ✅ DAC 调度器工作正常（若使用 `linear_decay`，ε_low 随训练进度收紧）

### 5.2 代码质量验收

- ✅ 单元测试 `test_dcpo_loss.py` 和 `test_sas_advantage.py` 全部通过
- ✅ DCPO 损失函数对照论文公式 4/8，数值误差 < 1e-5

### 5.3 性能验收

- ✅ 与 DAPO 同等超参下，DCPO 的 **TCR (Token Clipping Ratio)** 更低（下降 ≥ 20%）
- ✅ 与 DAPO 同等超参下，DCPO 的 **RUR (Response Utilization Ratio)** 更高（提升 ≥ 15%）
- ✅ 算法切换仅需修改 `grpo_loss_mode`

**TCR 计算**：
\[
\text{TCR} = \frac{\#\{\text{token} \mid \text{ratio} \notin [1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}}]\}}{\#\{\text{token}\}}
\]

**RUR 计算**：
\[
\text{RUR} = \frac{1}{G} \sum_{g=1}^{G} \mathbb{1}\!\left[\exists\, r \in \text{group}_g \text{ s.t. } \text{loss contribution of } r \neq 0\right]
\]

---

## 6. 依赖关系

### 上游依赖

- **M01 (GRPO)**: 依赖目录结构、参数定义、trainer 骨架
- **M02 (DAPO)**: 依赖非对称 clip 思想、Dynamic Sampling 基础设施

### 下游依赖

- **M05 (RewardManager)**: DCPO 可与任意 reward_type 组合
- **M06 (DCPO 进阶)**: 依赖本阶段的 DAC/SAS/OTM 核心实现

### 并行依赖

- **M05 (RewardManager)** 可与本阶段并行开发

---

## 7. 详细技术规范

### 7.1 DCPO DAC 公式

\[
L_{\text{DCPO}} = -\frac{1}{G} \sum_{g=1}^{G} \frac{1}{T_g} \sum_{j=1}^{T_g} m_{g,j} \cdot \min\left(r_{g,j} \cdot A_g, \text{clip}(r_{g,j}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \cdot A_g\right)
\]

其中：
- \(\varepsilon_{\text{low}} = 0.16\)（DAPO: 0.2）
- \(\varepsilon_{\text{high}} = 0.20\)（DAPO: 0.28）

### 7.2 DCPO Dual Clip

对负优势 token（\(A_g < 0\)），取：
\[
L_{\text{token}} = \max\left(L_{\text{clipped}}, -r_{\text{max}} \cdot A_g\right)
\]

其中 \(r_{\text{max}} = 10\)，防止负优势下 ratio>1 的极端负梯度。

### 7.3 DCPO SAS 公式

标准 group-relative advantage：
\[
A_g = \frac{r_g - \mu_{\text{group}}}{\sigma_{\text{group}}}
\]

SAS 平滑：
\[
A_g^{\text{smooth}} = \tanh\left(\frac{A_g}{k}\right) \cdot k
\]

其中 \(k = 3.0\)。

### 7.4 DCPO OTM Loss

与 `seq-mean-token-mean` 数学等价：
\[
L_{\text{OTM}} = \frac{1}{G} \sum_{g=1}^{G} \left(\frac{1}{T_g} \sum_{j=1}^{T_g} L_{g,j} \cdot m_{g,j}\right)
\]

---

## 8. 风险与应对

### 风险 8.1: DAC 阈值过紧导致训练缓慢

**风险描述**：ε_low=0.16, ε_high=0.20 相比 DAPO 更收紧，可能导致训练初期收敛慢。

**应对策略**：
- 提供 `linear_decay` 调度器：从 (0.20, 0.20) 线性收敛到 (0.16, 0.20)
- 在日志中监控 TCR，若 TCR < 5% 说明 clip 过紧，建议调整

### 风险 8.2: SAS 平滑导致梯度消失

**风险描述**：tanh 在 |adv| ≫ k 时饱和，梯度趋近于 0。

**应对策略**：
- 默认 `threshold=3.0`，覆盖 99.7% 的正态分布样本
- 在日志中监控 `|adv| > threshold` 的比例，若 > 5% 建议提高 threshold

### 风险 8.3: Dual Clip 参数敏感

**风险描述**：`dual_clip_ratio=10.0` 可能不适用于所有任务。

**应对策略**：
- 提供配置参数供用户调整
- 在日志中打印 Dual Clip 生效的 token 比例

---

## 9. 阶段完成 Checklist

- [ ] `loss.py` 新增 `compute_dcpo_loss`（DAC + Dual Clip + OTM）
- [ ] `advantage.py` 新增 `compute_smoothed_advantage`（SAS tanh 平滑）
- [ ] `dcpo.py` 实现 `get_dac_clip_ratios` + `compute_otm_loss`
- [ ] `finetuning_args.py` 新增 `dcpo_*` 参数（至少 7 个字段）
- [ ] `trainer.py` 集成 DCPO 分支（loss_fn 路由 + SAS/DAC 分发 + `_get_dac_clip_ratios`）
- [ ] `qwen3_lora_dcpo.yaml` 配置模板可运行
- [ ] `tests/test_dcpo_loss.py` 单元测试通过
- [ ] `tests/test_sas_advantage.py` 单元测试通过
- [ ] 与 DAPO 对比：TCR 下降 ≥ 20%，RUR 提升 ≥ 15%
- [ ] 在 `/docs/开发进度/` 创建 `M04_完成.md`，记录变更文件与验证结果

---

> **下一步**: 完成 M04 后，进入 **M05: RewardManager 集成**（4 种评分函数 + LLM-as-Judge）。
