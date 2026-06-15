# M01: GRPO基础框架搭建

> **阶段编号**: M01  
> **对应原里程碑**: M0  
> **创建时间**: 2026-06-10  
> **预计工期**: 5-7天

---

## 1. 阶段定位

本阶段是整个策略优化算法集成的**基础框架搭建阶段**，实现标准 GRPO 算法的核心可运行版本。作为后续 DAPO/GSPO/DCPO 扩展的基础设施基座，本阶段必须完成：

- 训练目录结构与模块骨架
- GRPO 损失函数与优势估计
- Stage 注册与参数定义
- 端到端训练流程编排
- 基础配置模板

**关键设计决策**：四种算法共享同一 stage `grpo`，通过 `grpo_loss_mode` 参数切换算法变体，确保后续扩展无需修改路由逻辑。

---

## 2. 阶段目标

### 2.1 业务目标

- 提供可用的 GRPO RLHF 训练能力，支持 Qwen3 等主流模型的策略优化
- 建立可扩展的算法框架，为后续 DAPO/GSPO/DCPO 集成奠定基础

### 2.2 技术目标

- 实现 `compute_grpo_loss`（对称 clip + group-relative advantage）
- 实现 `compute_group_relative_advantage`（group 内归一化）
- 在 `finetuning_args.py` 中完成 GRPO 相关参数定义
- 在 `tuner.py` 中注册 `grpo` stage 路由
- 完成端到端训练流程（rollout → reward → advantage → loss → update）
- 提供可运行的配置模板 `qwen3_lora_grpo.yaml`

---

## 3. 核心任务

### 任务 3.1: 创建 `train/grpo/` 目录结构

**任务描述**：建立 GRPO 训练模块的目录骨架，包含核心文件与导出接口。

**文件清单**：
```
src/llamafactory/train/grpo/
├── __init__.py          # export run_grpo
├── workflow.py          # 训练流程编排
├── trainer.py           # CustomGRPOTrainer 主类
├── loss.py             # GRPO 损失函数
└── advantage.py        # 优势函数估计
```

**产出**：
- `__init__.py` 导出 `run_grpo` 函数
- 各文件包含基础骨架（函数签名 + docstring）

---

### 任务 3.2: 实现 GRPO 损失函数 (`loss.py`)

**任务描述**：实现标准 GRPO 的 token-level clipped policy loss。

**技术细节**：

```python
def compute_grpo_loss(
    log_probs: torch.Tensor,          # [batch, seq_len]
    ref_log_probs: torch.Tensor,      # [batch, seq_len]
    advantages: torch.Tensor,         # [batch]
    mask: torch.Tensor,               # [batch, seq_len]
    clip_ratio: float = 0.2,
    loss_agg_mode: str = "seq-mean-token-mean",
) -> torch.Tensor:
    """标准 GRPO token-level clipped policy loss"""
    ratio = torch.exp(log_probs - ref_log_probs)  # [batch, seq_len]
    adv = advantages.unsqueeze(-1)  # [batch, 1]

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    token_loss = -torch.min(surr1, surr2)

    return _aggregate_loss(token_loss, mask, loss_agg_mode)
```

**聚合模式** (`_aggregate_loss`)：
- `token-mean`: 全局 token-mean
- `seq-mean-token-sum`: 先 seq 内求和，再 batch mean
- `seq-mean-token-mean`: 先 seq 内 token-mean，再 batch mean

---

### 任务 3.3: 实现优势函数估计 (`advantage.py`)

**任务描述**：实现 group-relative advantage 计算。

**技术细节**：

```python
def compute_group_relative_advantage(
    rewards: torch.Tensor,       # [batch]
    group_size: int,
    norm_by_std: bool = True,
) -> torch.Tensor:
    """Group-relative advantage: 在每个 group 内归一化"""
    num_groups = rewards.shape[0] // group_size
    rewards = rewards.view(num_groups, group_size)

    mean = rewards.mean(dim=-1, keepdim=True)
    if norm_by_std:
        std = rewards.std(dim=-1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards - mean) / std
    else:
        advantages = rewards - mean

    return advantages.view(-1)
```

---

### 任务 3.4: 扩展 `finetuning_args.py` 参数定义

**任务描述**：在 `src/llamafactory/hparams/finetuning_args.py` 中添加 GRPO 相关参数。

**核心参数**：

```python
@dataclass
class FinetuningArguments:
    # === GRPO 共享参数 ===
    grpo_loss_mode: Literal["grpo", "dapo", "gspo", "dcpo"] = "grpo"
    grpo_num_generations: int = 8
    grpo_temperature: float = 1.0
    grpo_top_p: float = 1.0
    grpo_top_k: int = -1
    grpo_max_response_length: int = 2048
    grpo_ppo_epochs: int = 1
    grpo_mini_batch_size: int = 8
    grpo_grad_clip: float = 1.0
    grpo_use_kl_loss: bool = True
    grpo_kl_coef: float = 0.001
    grpo_kl_type: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "kl"
    grpo_entropy_coeff: float = 0.0
    grpo_norm_adv_by_std: bool = True

    # === GRPO 特有参数 ===
    grpo_clip_ratio: float = 0.2
    grpo_loss_agg_mode: Literal[
        "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
    ] = "seq-mean-token-mean"
```

---

### 任务 3.5: 注册 `grpo` stage 路由 (`tuner.py`)

**任务描述**：在 `tuner.py` 的 stage 路由中新增 `grpo` 分支。

**代码修改**：

```python
from .grpo import run_grpo

# 在 stage 路由中新增:
elif finetuning_args.stage == "grpo":
    run_grpo(model_args, data_args, training_args, finetuning_args, generating_args)
```

同时扩展 stage 枚举：
```python
stage: Literal["pt", "sft", "rm", "dpo", "ppo", "kto", "grpo"] = "sft"
```

---

### 任务 3.6: 实现 Trainer 主类 (`trainer.py`)

**任务描述**：实现 `CustomGRPOTrainer`，完成单步训练流程。

**核心流程**：

```python
class CustomGRPOTrainer(Trainer):
    def __init__(self, ref_model, reward_fn, finetuning_args, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        self.reward_fn = reward_fn

    def training_step(self, model, inputs):
        """单步训练: rollout → reward → advantage → loss"""
        prompts = inputs["input_ids"]
        
        # 1. Rollout 生成 n 个 response
        responses, log_probs, mask = self._rollout(model, prompts)
        
        # 2. Reward 计算
        rewards = self._compute_rewards(prompts, responses)
        
        # 3. Advantage 计算
        advantages = compute_group_relative_advantage(
            rewards, self.grpo_args.grpo_num_generations,
            self.grpo_args.grpo_norm_adv_by_std,
        )
        
        # 4. Ref log probs
        with torch.no_grad():
            ref_log_probs = self._get_log_probs(self.ref_model, responses)
        
        # 5. Policy loss
        loss = compute_grpo_loss(
            log_probs, ref_log_probs, advantages, mask,
            clip_ratio=self.grpo_args.grpo_clip_ratio,
            loss_agg_mode=self.grpo_args.grpo_loss_agg_mode,
        )
        
        # 6. KL loss
        if self.grpo_args.grpo_use_kl_loss:
            kl = self._compute_kl(log_probs, ref_log_probs, mask)
            loss = loss + self.grpo_args.grpo_kl_coef * kl
        
        return loss
```

**注**：`_rollout`、`_compute_rewards`、`_get_log_probs`、`_compute_kl` 等辅助方法可先用占位实现，后续迭代完善。

---

### 任务 3.7: 实现 Workflow 编排 (`workflow.py`)

**任务描述**：实现 `run_grpo` 函数，完成端到端训练流程。

**核心流程**：

```python
def run_grpo(model_args, data_args, training_args, finetuning_args, generating_args):
    """GRPO 统一训练入口"""
    tokenizer = load_tokenizer(model_args)
    dataset = load_dataset(data_args, stage="grpo")
    model = load_model(model_args, training_args)
    ref_model = create_ref_model(model_args, finetuning_args)
    reward_fn = create_reward_fn(finetuning_args)  # 临时 reward 函数

    trainer = CustomGRPOTrainer(
        model=model, ref_model=ref_model,
        reward_fn=reward_fn,
        finetuning_args=finetuning_args, args=training_args,
        train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
```

---

### 任务 3.8: 创建 GRPO 配置模板

**任务描述**：创建 `examples/train_lora/qwen3_lora_grpo.yaml` 配置模板。

**配置内容**：

```yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct
stage: grpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
grpo_loss_mode: grpo
grpo_num_generations: 8
grpo_temperature: 1.0
grpo_max_response_length: 2048
grpo_clip_ratio: 0.2
grpo_loss_agg_mode: seq-mean-token-mean
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
output_dir: saves/qwen3-4b/lora/grpo
```

---

## 4. 交付物清单

| 编号 | 交付物 | 路径 | 类型 |
|------|--------|------|------|
| D-M01-01 | GRPO 模块目录 | `src/llamafactory/train/grpo/` | 目录 |
| D-M01-02 | 损失函数实现 | `src/llamafactory/train/grpo/loss.py` | 代码 |
| D-M01-03 | 优势函数实现 | `src/llamafactory/train/grpo/advantage.py` | 代码 |
| D-M01-04 | Trainer 主类 | `src/llamafactory/train/grpo/trainer.py` | 代码 |
| D-M01-05 | Workflow 编排 | `src/llamafactory/train/grpo/workflow.py` | 代码 |
| D-M01-06 | 模块导出 | `src/llamafactory/train/grpo/__init__.py` | 代码 |
| D-M01-07 | finetuning_args 扩展 | `src/llamafactory/hparams/finetuning_args.py` | 代码修改 |
| D-M01-08 | stage 注册 | `src/llamafactory/train/tuner.py` | 代码修改 |
| D-M01-09 | GRPO 配置模板 | `examples/train_lora/qwen3_lora_grpo.yaml` | 配置文件 |

---

## 5. 验收标准

### 5.1 功能验收

- ✅ `grpo_loss_mode=grpo` 时可完成一轮完整训练（不报错）
- ✅ Loss 在训练过程中正常下降（至少 10 个 step）
- ✅ 配置模板可直接运行（`llamafactory-cli train examples/train_lora/qwen3_lora_grpo.yaml`）

### 5.2 代码质量验收

- ✅ 所有新增文件通过 linter 检查
- ✅ 核心函数包含 docstring 和类型注解
- ✅ 无硬编码路径或 magic number

### 5.3 性能验收

- ✅ 在单卡 V100 上，batch_size=2, grpo_num_generations=8 时可正常训练
- ✅ 显存占用不超过 40GB（Qwen3-4B + LoRA）

---

## 6. 依赖关系

### 上游依赖

- **无**：本阶段为基础框架搭建，不依赖其他阶段

### 下游依赖

- **M02 (DAPO)**: 依赖本阶段的目录结构、参数定义、trainer 骨架
- **M03 (GSPO)**: 依赖本阶段的 loss.py 和 advantage.py 基础实现
- **M04 (RewardManager)**: 依赖本阶段的 workflow 接口定义（`reward_fn` 参数）

### 并行依赖

- **M04 (RewardManager)** 可与本阶段并行开发，但需约定接口契约：
  - `reward_fn(prompts, responses, ground_truths) -> torch.Tensor[batch]`

---

## 7. 详细技术规范

### 7.1 GRPO 损失公式

标准 GRPO 使用对称 clip 的 PPO-style 损失：

\[
L_{\text{GRPO}} = -\frac{1}{\sum_{i,j} m_{i,j}} \sum_{i=1}^{B} \sum_{j=1}^{T_i} m_{i,j} \cdot \min\left(r_{i,j} \cdot A_i, \text{clip}(r_{i,j}, 1-\epsilon, 1+\epsilon) \cdot A_i\right)
\]

其中：
- \(r_{i,j} = \exp(\log \pi_\theta(y_{i,j}|x) - \log \pi_{\text{ref}}(y_{i,j}|x))\)
- \(A_i\) 为 group-relative advantage
- \(m_{i,j}\) 为 mask（仅对 response token 为 1）

### 7.2 Group-Relative Advantage

在每个 group（同一 prompt 的 n 个 response）内做归一化：

\[
A_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}}}
\]

其中 \(\mu_{\text{group}}\) 和 \(\sigma_{\text{group}}\) 为 group 内 reward 的均值和标准差。

### 7.3 Rollout 接口约定

```python
def _rollout(self, model, prompts) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    返回:
        responses: [batch * num_generations, seq_len]
        log_probs: [batch * num_generations, seq_len]
        mask: [batch * num_generations, seq_len]
    """
    pass
```

### 7.4 KL 损失计算

```python
def _compute_kl(self, log_probs, ref_log_probs, mask, kl_type="kl"):
    """根据 kl_type 计算不同形式的 KL 散度"""
    if kl_type == "kl":
        return (torch.exp(log_probs - ref_log_probs) * (log_probs - ref_log_probs) * mask).sum() / mask.sum()
    # 其他 kl_type 在后续阶段实现
```

---

## 8. 风险与应对

### 风险 8.1: Rollout 实现复杂度高

**风险描述**：vLLM/HF 采样器的集成可能涉及大量代码改造。

**应对策略**：
- M01 阶段先用 HF `model.generate()` 占位
- 后续阶段再优化为 vLLM 异步采样

### 风险 8.2: Reward 函数未就绪

**风险描述**：M04 (RewardManager) 可能未同步完成。

**应对策略**：
- M01 使用简单的 random reward 或 rule-based reward 占位
- 确保 reward 接口可替换（函数签名约定）

### 风险 8.3: 显存溢出

**风险描述**：同时加载 policy model + ref model 可能导致 OOM。

**应对策略**：
- 使用 `offload` 策略将 ref model 放在 CPU
- 减小 `grpo_num_generations` 和 `per_device_train_batch_size`

---

## 9. 阶段完成 Checklist

- [ ] `train/grpo/` 目录结构创建完成
- [ ] `loss.py` 实现 `compute_grpo_loss` + `_aggregate_loss`
- [ ] `advantage.py` 实现 `compute_group_relative_advantage`
- [ ] `trainer.py` 实现 `CustomGRPOTrainer` 基础骨架
- [ ] `workflow.py` 实现 `run_grpo` 端到端流程
- [ ] `finetuning_args.py` 新增 GRPO 参数（至少 15 个字段）
- [ ] `tuner.py` 注册 `grpo` stage 路由
- [ ] `qwen3_lora_grpo.yaml` 配置模板可运行
- [ ] 完成一轮完整训练（loss 下降）
- [ ] 代码通过 linter 检查
- [ ] 在 `/docs/开发进度/` 创建 `M01_完成.md`，记录变更文件与验证结果

---

> **下一步**: 完成 M01 后，进入 **M02: DAPO 扩展**（非对称 clip + Dynamic Sampling + Overlong Shaping）。
