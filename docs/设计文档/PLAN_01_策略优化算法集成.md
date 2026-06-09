# PLAN_01: 策略优化算法集成（GRPO / DAPO / GSPO / DCPO）+ RewardManager

> **进度记录要求**：每完成一个里程碑阶段（M0 ~ M1），必须在 `/docs/开发进度/` 目录下创建对应的进度文件（如 `M0_完成.md`），记录完成时间、变更文件列表、验证结果。

---

## 1. 总体架构

四种算法共享同一个训练 stage `grpo`，通过 `grpo_loss_mode` 参数切换具体算法变体（GRPO 为基线，DAPO / GSPO / DCPO 为其改进变体）：

```
finetuning_args.stage = "grpo"
finetuning_args.grpo_loss_mode ∈ {"grpo", "dapo", "gspo", "dcpo"}
```

> **算法谱系**：GRPO → DAPO（非对称 clip + Dynamic Sampling + Overlong Shaping）→ GSPO（序列级 clip）→ DCPO（DAC 非对称 clip + Dual Clip + SAS 平滑优势 + OTM Loss）。DCPO 是 DAPO 的进一步改进，引入动态自适应裁剪（DAC）、平滑优势标准化（SAS）以及仅在单条 response 内 token 求平均的损失聚合（OTM Loss）。

### 1.1 模块结构

```
src/llamafactory/train/grpo/
├── __init__.py          # export run_grpo
├── workflow.py          # 训练流程编排
├── trainer.py           # CustomGRPOTrainer 主类
├── loss.py             # 四种算法的损失函数实现 (grpo/dapo/gspo/dcpo)
├── advantage.py        # 优势函数估计 (group-relative / SAS 平滑)
├── sampling.py         # Rollout 采样 & Dynamic Sampling (DAPO)
├── reward_shaping.py   # Overlong Reward Shaping (DAPO)
├── dcpo.py             # DCPO 专属：DAC 阈值调度 + OTM Loss 聚合 + Dual Clip
└── reward/             # RewardManager (参考 verl reward_manager + reward_score)
    ├── __init__.py
    ├── manager.py       # RewardManager 主类 (类似 verl NaiveRewardManager)
    ├── math.py          # 数学答案评分 (math_score)
    ├── multiple_choice.py # 多选题评分 (multiple_choice_score)
    ├── string_match.py  # 字符串匹配 (string_match_score)
    ├── llm_judge.py     # LLM-as-Judge (llm_judge_score + 默认 prompt)
    └── registry.py      # SCORE_REGISTRY 评分函数注册表
```

### 1.2 数据流

```
Prompt Dataset
    │
    ▼
┌─────────────────┐
│  Rollout 采样    │  每个 prompt 生成 n 个 response (group)
│  (vLLM / HF)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reward 计算     │  RewardManager 根据 reward_type 分发到
│  + Shaping      │  4 种评分函数 (math/multiple_choice/string_match/llm_judge)
└────────┬────────┘  │  (DAPO: overlong penalty)
         │
         ▼
┌─────────────────┐
│  Advantage 估计  │  group-relative normalization
│  + 过滤         │  (DAPO: filter all-0/all-1 groups)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Policy Loss    │  GRPO / DAPO / GSPO 各自损失
│  + KL Loss      │
└─────────────────┘
```

---

## 2. Schema 设计：超参数定义

在 `src/llamafactory/hparams/finetuning_args.py` 中新增以下字段：

```python
@dataclass
class FinetuningArguments:
    # === GRPO/DAPO/GSPO/DCPO 共享参数 ===
    grpo_loss_mode: Literal["grpo", "dapo", "gspo", "dcpo"] = "grpo"
    grpo_num_generations: int = 8          # rollout.n, 每个 prompt 的采样数
    grpo_temperature: float = 1.0          # 采样温度
    grpo_top_p: float = 1.0                # top-p 采样
    grpo_top_k: int = -1                   # top-k 采样
    grpo_max_response_length: int = 2048   # 最大生成长度
    grpo_ppo_epochs: int = 1               # 每批数据的更新轮数
    grpo_mini_batch_size: int = 8          # PPO mini-batch size
    grpo_grad_clip: float = 1.0            # 梯度裁剪
    grpo_use_kl_loss: bool = True          # 是否使用 KL 损失
    grpo_kl_coef: float = 0.001            # KL 损失系数
    grpo_kl_type: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "kl"
    grpo_entropy_coeff: float = 0.0        # 熵正则系数
    grpo_norm_adv_by_std: bool = True      # 是否用标准差归一化优势

    # === GRPO 特有参数 ===
    grpo_clip_ratio: float = 0.2           # 对称 clip 范围
    grpo_loss_agg_mode: Literal[
        "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
    ] = "seq-mean-token-mean"

    # === DAPO 特有参数 ===
    dapo_clip_ratio_low: float = 0.2       # 非对称 clip 下界 ε_low
    dapo_clip_ratio_high: float = 0.28     # 非对称 clip 上界 ε_high
    dapo_dynamic_sampling: bool = True     # 是否启用动态采样
    dapo_filter_metric: Literal["acc", "score", "seq_reward"] = "acc"
    dapo_max_gen_batches: int = 10         # 动态采样最大重试批次
    dapo_overlong_shaping: bool = True     # 是否启用超长惩罚
    dapo_overlong_buffer_len: int = 256    # 超长缓冲区长度
    dapo_overlong_penalty_factor: float = 1.0  # 超长惩罚因子

    # === GSPO 特有参数 ===
    gspo_clip_ratio_c: float = 3.0         # 序列级 clip 参数
    gspo_use_megatron: bool = False        # 是否使用 Megatron 策略

    # === DCPO 特有参数 (DAPO 的进一步改进) ===
    # 1) DAC (Dynamic-Adaptive Clipping): 动态自适应上下界
    dcpo_clip_ratio_low: float = 0.16      # DAC 下界 ε_low (DAPO 默认 0.2, DCPO 收紧)
    dcpo_clip_ratio_high: float = 0.20     # DAC 上界 ε_high (DAPO 默认 0.28, DCPO 收紧)
    dcpo_dual_clip_ratio: float = 10.0     # Dual Clip 上界 r_max (论文 r_max=10)
    # 2) SAS (Smooth Advantage Standardization): 平滑优势标准化
    dcpo_sas_enable: bool = True           # 是否启用 SAS
    dcpo_sas_threshold: float = 3.0        # SAS 截断阈值 k (论文用 k=3)
    # 3) OTM Loss (Only Token Mean Loss): 仅在单条 response 内 token 求平均
    #    DCPO 论文采用 OTM 模式, 但保留 token-mean / seq-mean-token-mean 作为消融选项
    dcpo_loss_agg_mode: Literal[
        "otm", "token-mean", "seq-mean-token-mean"
    ] = "otm"
    # 可选: DAC 调度器 (固定/线性退火)
    dcpo_clip_schedule: Literal["constant", "linear_decay"] = "constant"

    # === RewardManager 参数 (4 种评分函数) ===
    grpo_reward_type: Literal[
        "math", "multiple_choice", "string_match", "llm_judge"
    ] = "math"                              # RewardManager 评分类型
    # 评分模式: 当前 4 种 score_fn 都返回 0.0/1.0 (binary); 如未来需要
    # 连续分数 (如模糊匹配), 在各 score_fn 中扩展此模式
    grpo_reward_score_mode: Literal["binary"] = "binary"
    # math: 抽取 boxed{} / #### 后答案, 二值匹配 (1.0/0.0)
    grpo_reward_math_extract_mode: Literal["boxed", "hash", "last_number"] = "boxed"
    # multiple_choice: 抽取 A/B/C/D 选项
    grpo_reward_mc_pattern: str = r"(?i)\\boxed\{\s*([A-D])\s*\}|answer\s*[:：]?\s*([A-D])"
    # string_match: 字符串规范化匹配
    grpo_reward_strict_match: bool = False  # True=严格相等, False=规范化后相等
    # llm_judge: 调用外部大模型做评判
    grpo_llm_judge_url: str = ""            # e.g. http://localhost:8000/v1/chat/completions
    grpo_llm_judge_model: str = ""          # e.g. Qwen/Qwen2.5-32B-Instruct
    grpo_llm_judge_max_tokens: int = 256
    grpo_llm_judge_temperature: float = 0.0
    grpo_llm_judge_timeout: int = 30        # 单次评分超时(s)
    grpo_llm_judge_concurrency: int = 16    # 异步并发数
    grpo_llm_judge_fallback_score: float = 0.0  # 评分失败时的兜底分数
```

---

## 3. 核心代码设计

### 3.1 损失函数 (`loss.py`)

```python
import torch


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
    seq_log_ratio = ((log_probs - ref_log_probs) * mask).sum(dim=-1)
    seq_ratio = torch.exp(seq_log_ratio)

    surr1 = seq_ratio * advantages
    surr2 = torch.clamp(seq_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * advantages
    surr2 = torch.clamp(surr2, -clip_ratio_c, clip_ratio_c)

    loss = -torch.min(surr1, surr2)
    return loss.mean()


def _aggregate_loss(token_loss, mask, mode):
    """Token loss 聚合到 scalar loss
    mode 选项:
      - "token-mean":           全局 token-mean = sum(token_loss*mask) / sum(mask)
      - "seq-mean-token-sum":   先在每条 seq 内求和, 再对 batch 求 mean
      - "seq-mean-token-mean":  先在每条 seq 内求 token-mean, 再对 batch 求 mean
      - "otm":                  DCPO OTM 模式, 与 seq-mean-token-mean 数学等价,
                                保留为语义别名, 便于论文对齐 (OTM = only-token-mean)
    """
    if mode == "token-mean":
        return (token_loss * mask).sum() / mask.sum()
    elif mode == "seq-mean-token-sum":
        return (token_loss * mask).sum(dim=-1).mean()
    elif mode == "otm":
        # 与 seq-mean-token-mean 相同: per-response token-mean → batch-mean
        # 保留为独立分支仅作语义别名 (DCPO 论文中称 OTM)
        lengths = mask.sum(dim=-1).clamp(min=1)
        return ((token_loss * mask).sum(dim=-1) / lengths).mean()
    else:  # seq-mean-token-mean (含 otm 语义别名场景)
        lengths = mask.sum(dim=-1).clamp(min=1)
        return ((token_loss * mask).sum(dim=-1) / lengths).mean()


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
       (对齐 OpenRLHF DAPO / DCPO 官方实现: 仅在 adv<0 时应用, 无需 ratio>1 条件;
        此时 -dual_clip_ratio*adv 必为正, 截断掉 ratio>1 时的极端负梯度)
    3) OTM Loss: per-response token-mean → batch-mean
    """
    ratio = torch.exp(log_probs - ref_log_probs)  # [batch, seq_len]
    adv = advantages.unsqueeze(-1)                # [batch, 1]

    # --- (1) DAC: 非对称 clip ---
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high) * adv
    token_loss = -torch.min(surr1, surr2)

    # --- (2) Dual Clip: 防止负优势下 ratio>1 的极端更新 ---
    # 当 adv<0 时, 取 max(token_loss, -dual_clip_ratio * adv)
    #   ∵ -dual_clip_ratio*adv > 0, 而 token_loss 在 ratio>1 时也可能偏正,
    #     故 max 将其限制在 dual_clip_ratio * |adv| 以下, 避免极端负梯度
    if dual_clip_ratio is not None and dual_clip_ratio > 0:
        # 仅对负优势生效 (OpenRLHF DAPO 标准实现)
        neg_adv_mask = (adv < 0).float()
        # dual_loss = -dual_clip_ratio * adv  (对负 adv 而言为正数, 作为上界)
        dual_loss = -dual_clip_ratio * adv
        token_loss = torch.max(token_loss, dual_loss) * neg_adv_mask + \
                     token_loss * (1 - neg_adv_mask)

    return _aggregate_loss(token_loss, mask, loss_agg_mode)


### 3.2 优势函数估计 (`advantage.py`)

```python
import torch


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
    else:
        valid = grouped.std(dim=-1) > 1e-8

    return valid.unsqueeze(-1).expand(-1, group_size).reshape(-1)


def compute_smoothed_advantage(
    rewards: torch.Tensor,         # [batch]
    group_size: int,
    threshold: float = 3.0,
) -> torch.Tensor:
    """DCPO SAS (Smooth Advantage Standardization) 平滑优势标准化
    标准 GRPO/DAPO 用 hard clip 把 |adv| 截断到 [-k, k], 但硬截断在边界处
    不平滑, 易引发训练震荡. SAS 用 tanh 平滑近似:
        adv_smooth = tanh(adv / k) * k
    这样:
    1. |adv| ≪ k 时, tanh(adv/k) ≈ adv/k, 退化为线性
    2. |adv| ≫ k 时, tanh → ±1, 渐近收敛到 ±k (无硬截断)
    3. 一阶连续可导, 优化更稳定

    注意: 是否启用 SAS 由 trainer.py 决定 (本函数无条件返回平滑结果).
    """
    # 先做标准 group-relative 归一化
    advantages = compute_group_relative_advantage(
        rewards, group_size, norm_by_std=True,
    )

    # SAS 平滑: tanh(adv/k) * k
    smoothed = torch.tanh(advantages / threshold) * threshold
    return smoothed
```

### 3.3 Reward Shaping (`reward_shaping.py`)

```python
import torch


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

### 3.4 Trainer 主类 (`trainer.py`)

```python
from transformers import Trainer
from .reward.manager import RewardManager, RewardInput

class CustomGRPOTrainer(Trainer):
    def __init__(self, ref_model, reward_manager, finetuning_args, **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.grpo_args = finetuning_args
        # RewardManager 替代原 reward_fn (详见 7.7 节)
        self.reward_manager: RewardManager = reward_manager
        self.loss_fn = {
            "grpo": compute_grpo_loss,
            "dapo": compute_dapo_loss,
            "gspo": compute_gspo_loss,
            "dcpo": compute_dcpo_loss,
        }[finetuning_args.grpo_loss_mode]  

    def _compute_rewards(self, prompts, responses, ground_truths):
        """通过 RewardManager 批量评分 (详见 7.8 节)"""
        inputs = [
            RewardInput(response=r, ground_truth=g, prompt=p)
            for r, g, p in zip(responses, ground_truths, prompts)
        ]
        return self.reward_manager(inputs)

    def training_step(self, model, inputs):
        """单步训练: rollout → reward → advantage → loss"""
        prompts = inputs["input_ids"]
        ground_truths = inputs["ground_truth"]      # 来自 dataset

        # 1. Rollout 生成 n 个 response
        responses, log_probs, mask = self._rollout(model, prompts)

        # 2. RewardManager 评分
        response_strs = self._decode_responses(responses)
        rewards = self._compute_rewards(prompts, response_strs, ground_truths)

        # 3. DAPO overlong shaping
        if self.grpo_args.grpo_loss_mode == "dapo" and self.grpo_args.dapo_overlong_shaping:
            lengths = mask.sum(dim=-1)
            rewards = apply_overlong_penalty(
                rewards, lengths, self.grpo_args.grpo_max_response_length,
                self.grpo_args.dapo_overlong_buffer_len,
                self.grpo_args.dapo_overlong_penalty_factor,
            )

        # 4. 计算 advantage
        # DCPO: 若启用 SAS, 用 tanh 平滑优势 (论文式 6/7)
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

        # 5. DAPO 过滤 trivial groups
        if self.grpo_args.grpo_loss_mode == "dapo" and self.grpo_args.dapo_dynamic_sampling:
            valid_mask = filter_trivial_groups(
                rewards, self.grpo_args.grpo_num_generations,
                self.grpo_args.dapo_filter_metric,
            )
            # 仅保留有效样本参与损失计算
            advantages = advantages * valid_mask

        # 5.5 DCPO: DAC 阈值调度 (可选)
        # 当前 schedule=constant 时保持原始 ε_low/ε_high; linear_decay 时按训练进度收紧.
        # 调度结果缓存到 self._current_dac_clip, 由 _get_loss_kwargs 读取
        if self.grpo_args.grpo_loss_mode == "dcpo":
            self._current_dac_clip = self._get_dac_clip_ratios()
        else:
            self._current_dac_clip = None

        # 6. Ref log probs
        with torch.no_grad():
            ref_log_probs = self._get_log_probs(self.ref_model, responses)

        # 7. Policy loss (根据 loss_mode 自动分发)
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
        elif mode == "dcpo":
            # 优先使用 DAC 调度器 (constant/linear_decay) 产出的 ε_low/ε_high
            # 若该 step 未触发调度 (理论上不会发生), 回退到 finetuning_args
            if getattr(self, "_current_dac_clip", None) is not None:
                clip_low, clip_high = self._current_dac_clip
            else:
                clip_low = self.grpo_args.dcpo_clip_ratio_low
                clip_high = self.grpo_args.dcpo_clip_ratio_high
            return {"clip_ratio_low": clip_low,
                    "clip_ratio_high": clip_high,
                    "dual_clip_ratio": self.grpo_args.dcpo_dual_clip_ratio,
                    "loss_agg_mode": self.grpo_args.dcpo_loss_agg_mode}
        else:  # gspo
            return {"clip_ratio_low": self.grpo_args.dapo_clip_ratio_low,
                    "clip_ratio_high": self.grpo_args.dapo_clip_ratio_high,
                    "clip_ratio_c": self.grpo_args.gspo_clip_ratio_c}

    def _get_dac_clip_ratios(self):
        """DCPO DAC 阈值调度: constant 或 linear_decay"""
        schedule = self.grpo_args.dcpo_clip_schedule
        low = self.grpo_args.dcpo_clip_ratio_low
        high = self.grpo_args.dcpo_clip_ratio_high
        if schedule == "constant":
            return low, high
        elif schedule == "linear_decay":
            # 按训练进度从 (high, high) 线性收敛到 (low, high)
            progress = min(1.0, self.state.global_step / max(1, self.state.max_steps))
            cur_low = high - (high - low) * progress
            return cur_low, high
        return low, high
```

### 3.5 Workflow (`workflow.py`)

```python
from .reward.manager import RewardManager


def create_reward_manager(finetuning_args) -> RewardManager:
    """工厂方法: 根据 grpo_reward_type 构造 RewardManager (详见 7.7)"""
    return RewardManager(finetuning_args)


def run_grpo(model_args, data_args, training_args, finetuning_args, generating_args):
    """GRPO/DAPO/GSPO/DCPO 统一训练入口"""
    tokenizer = load_tokenizer(model_args)
    dataset = load_dataset(data_args, stage="grpo")
    model = load_model(model_args, training_args)
    ref_model = create_ref_model(model_args, finetuning_args)
    # RewardManager 替代原 reward_fn
    reward_manager = create_reward_manager(finetuning_args)

    trainer = CustomGRPOTrainer(
        model=model, ref_model=ref_model,
        reward_manager=reward_manager,
        finetuning_args=finetuning_args, args=training_args,
        train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model()
```

---

## 4. 配置模板

### 4.1 GRPO (`examples/train_lora/qwen3_lora_grpo.yaml`)

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

### 4.2 DAPO (`examples/train_lora/qwen3_lora_dapo.yaml`)

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

### 4.3 GSPO (`examples/train_lora/qwen3_lora_gspo.yaml`)

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
# GSPO 复用 dapo_clip_ratio_low/high 作为序列级 clip 范围 (见 _get_loss_kwargs)
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

### 4.4 DCPO (`examples/train_lora/qwen3_lora_dcpo.yaml`)

DCPO 是 DAPO 的进一步改进，启用 **DAC（动态自适应裁剪）+ SAS（平滑优势标准化）+ OTM Loss** 三大核心技术：

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
# 1) DAC: 收紧的非对称 clip 范围 (论文 ε_low=0.16, ε_high=0.20)
dcpo_clip_ratio_low: 0.16
dcpo_clip_ratio_high: 0.20
dcpo_dual_clip_ratio: 10.0      # Dual Clip 上界 r_max
dcpo_clip_schedule: constant    # 可选: constant | linear_decay

# 2) SAS: 平滑优势标准化 (tanh 平滑近似硬截断)
dcpo_sas_enable: true
dcpo_sas_threshold: 3.0         # 截断阈值 k, 论文用 k=3

# 3) OTM Loss: 仅在单条 response 内 token 求平均
dcpo_loss_agg_mode: otm         # otm (默认) | token-mean | seq-mean-token-mean

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

### 4.5 算法变体速查表

| `grpo_loss_mode` | 核心创新 | 关键参数 | 适用场景 |
|---|---|---|---|
| `grpo`  | 对称 clip + group-relative advantage | `grpo_clip_ratio=0.2` | 通用 RLHF 基线 |
| `dapo`  | 非对称 clip + Dynamic Sampling + Overlong Shaping | `dapo_clip_ratio_low=0.2`<br>`dapo_clip_ratio_high=0.28` | 长 CoT / 推理任务 |
| `gspo`  | 序列级 importance ratio + clip_c | `gspo_clip_ratio_c=3.0` | 噪声大 / 长度方差大的任务 |
| `dcpo`  | DAC + SAS + OTM Loss | `dcpo_clip_ratio_low=0.16`<br>`dcpo_clip_ratio_high=0.20`<br>`dcpo_sas_threshold=3.0` | 高质量推理 + 训练稳定性 |

> 切换算法仅需修改 `grpo_loss_mode` 字段，所有算法共享同一套 stage、rollout、reward、optimizer 流水线。

### 4.6 RewardManager 配置示例（4 种 reward_type）

下面 4 个 yaml 片段展示 RewardManager 与不同算法的搭配（仅展示 reward 相关字段，其余同 4.1~4.4）：

#### 4.6.1 数学答案（DCPO + math）

```yaml
grpo_loss_mode: dcpo
grpo_reward_type: math
grpo_reward_score_mode: binary
grpo_reward_math_extract_mode: boxed   # boxed / hash / last_number
# 其余 DCPO 字段同 4.4
```

#### 4.6.2 多选题（DAPO + multiple_choice）

```yaml
grpo_loss_mode: dapo
grpo_reward_type: multiple_choice
grpo_reward_mc_pattern: "(?i)\\\\boxed\\{\\s*([A-D])\\s*\\}|answer\\s*[:：]?\\s*([A-D])"
# 其余 DAPO 字段同 4.2
```

#### 4.6.3 字符串匹配（GRPO + string_match）

```yaml
grpo_loss_mode: grpo
grpo_reward_type: string_match
grpo_reward_strict_match: false       # 宽松匹配 (去标点/小写)
# 其余 GRPO 字段同 4.1
```

#### 4.6.4 LLM-as-Judge（GSPO + llm_judge）

```yaml
grpo_loss_mode: gspo
grpo_reward_type: llm_judge
grpo_llm_judge_url: http://localhost:8000/v1/chat/completions
grpo_llm_judge_model: Qwen/Qwen2.5-32B-Instruct
grpo_llm_judge_max_tokens: 256
grpo_llm_judge_temperature: 0.0
grpo_llm_judge_timeout: 30
grpo_llm_judge_concurrency: 16
grpo_llm_judge_fallback_score: 0.0   # 调用失败兜底
# 其余 GSPO 字段同 4.3
```

### 4.7 算法 × Reward 推荐组合

| 场景 | 推荐算法 | 推荐 reward_type | 理由 |
|---|---|---|---|
| 数学推理（GSM8K / MATH） | DCPO | `math` (boxed) | 长 CoT + DAC/SAS 提升训练稳定性 |
| 通用 QA / 知识问答 | DAPO | `string_match` 或 `llm_judge` | Dynamic Sampling 过滤 trivial |
| 多选题（MMLU / ARC） | DAPO | `multiple_choice` | 高质量采样 + 过滤 |
| 开放式生成 / 创意任务 | GRPO | `llm_judge` | 通用基线 + LLM 主观评分 |
| 代码生成 | GSPO | `string_match` | 序列级 clip 适合整段匹配 |
| 工具调用 / Agent | DAPO | `llm_judge` + Dynamic Sampling | 复杂奖励需异步评判 |

> **说明**：reward_type 与算法解耦，可任意组合（如 DCPO+llm_judge、GSPO+math 等），按场景需求选择即可。

---

## 5. Stage 注册

### 5.1 `tuner.py` 修改

```python
from .grpo import run_grpo

# 在 stage 路由中新增:
elif finetuning_args.stage == "grpo":
    run_grpo(model_args, data_args, training_args, finetuning_args, generating_args)
```

### 5.2 `finetuning_args.py` stage 枚举扩展

```python
stage: Literal["pt", "sft", "rm", "dpo", "ppo", "kto", "grpo"] = "sft"
```

---

## 6. 里程碑划分

### M0: 基础框架搭建（GRPO 核心可用）

| 任务 | 产出 |
|------|------|
| 创建 `train/grpo/` 目录结构 | `__init__.py`, `workflow.py`, `trainer.py`, `loss.py`, `advantage.py` |
| 在 `finetuning_args.py` 添加共享参数 + GRPO 参数 | 参数定义 + 校验 |
| 在 `tuner.py` 注册 `grpo` stage | 路由分支 |
| 实现 `compute_grpo_loss` + `compute_group_relative_advantage` | 基础 GRPO 可运行 |
| 实现 `workflow.py` 完整流程（rollout → reward → train） | 端到端流程 |
| 创建 `qwen3_lora_grpo.yaml` | 配置模板 |
| **验收标准** | GRPO 配置可完成一轮完整训练，loss 正常下降 |

### M0.5: DAPO 扩展

| 任务 | 产出 |
|------|------|
| 实现 `compute_dapo_loss`（非对称 clip） | `loss.py` 扩展 |
| 实现 `filter_trivial_groups`（Dynamic Sampling） | `sampling.py` |
| 实现 `apply_overlong_penalty`（Overlong Reward Shaping） | `reward_shaping.py` |
| 在 `finetuning_args.py` 添加 `dapo_*` 参数 | 参数定义 |
| Trainer 中按 `grpo_loss_mode == "dapo"` 分支调用 | `trainer.py` |
| 创建 `qwen3_lora_dapo.yaml` | 配置模板 |
| **验收标准** | DAPO 训练可运行，日志可见 Dynamic Sampling 过滤行为 |

### M1: GSPO 扩展 + 全量验证

| 任务 | 产出 |
|------|------|
| 实现 `compute_gspo_loss`（序列级 ratio + clip_c） | `loss.py` 扩展 |
| 在 `finetuning_args.py` 添加 `gspo_*` 参数 | 参数定义 |
| Trainer 中按 `grpo_loss_mode == "gspo"` 分支调用 | `trainer.py` |
| 创建 `qwen3_lora_gspo.yaml` | 配置模板 |
| 三种算法端到端集成测试 | 测试脚本 |
| 文档更新 | README 使用说明 |
| **验收标准** | 三种算法均可正常训练，切换仅需修改 `grpo_loss_mode` |

### M2: DCPO 扩展（DAC + SAS + OTM Loss 三大创新集成）

| 任务 | 产出 |
|------|------|
| 实现 `compute_dcpo_loss`（DAC 非对称 clip + Dual Clip + OTM Loss） | `loss.py` 扩展 |
| 实现 `compute_smoothed_advantage`（SAS tanh 平滑优势） | `advantage.py` |
| 在 1.1 节定义的 `dcpo.py` 中实现 DAC 阈值调度 + Dual Clip + OTM 聚合工具函数 | `dcpo.py` |
| 在 `finetuning_args.py` 添加 `dcpo_*` 参数（clip_ratio_low/high、dual_clip_ratio、sas_*、loss_agg_mode、clip_schedule） | 参数定义 |
| 在 `trainer.py` 注册 `"dcpo": compute_dcpo_loss` 分支 + 训练步骤中 SAS/DAC 分发逻辑 | `trainer.py` |
| 新增 `_get_dac_clip_ratios` 方法（constant / linear_decay 调度） | `trainer.py` |
| 创建 `qwen3_lora_dcpo.yaml` | 配置模板 |
| 单元测试：`compute_dcpo_loss` 对照论文公式 4/8 数值一致性 | `tests/test_dcpo_loss.py` |
| 单元测试：`compute_smoothed_advantage` 与硬截断等价性 / 边界行为 | `tests/test_sas_advantage.py` |
| 与 DAPO 对比 ablation：仅启用 DAC / 仅启用 SAS / 仅启用 OTM 的独立效果 | ablation 脚本 |
| README & 文档更新 | README 使用说明 |
| **验收标准** | ① DCPO 训练可正常启动并 loss 下降；② 与 DAPO 同等超参下, DCPO 的 **Token Clipping Ratio (TCR)** 更低、**Response Utilization Ratio (RUR)** 更高；③ 算法切换仅需修改 `grpo_loss_mode` |

> **TCR (Token Clipping Ratio)**：被 clip 截断的 token 占总有效 token 的比例。**越低越好**，说明 clip 没有过度抑制有效更新。
>   \[
>   \text{TCR} = \frac{\#\{\text{token} \mid \text{ratio} \notin [1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}}]\}}{\#\{\text{token}\}}
>   \]
>
> **RUR (Response Utilization Ratio)**：在一个 group 内对模型更新产生**非零梯度**的 response 占该 group 的比例。**越高越好**，说明 group 内更多 response 在推动学习。
>   \[
>   \text{RUR} = \frac{1}{G} \sum_{g=1}^{G} \mathbb{1}\!\left[\exists\, r \in \text{group}_g \text{ s.t. } \text{loss contribution of } r \neq 0\right]
>   \]
>   其中 \(G\) 为 batch 内 group 数。
>
> DCPO 论文 §5.2 实验表明：相比 DAPO，DCPO 在数学/代码等推理任务上**TCR 下降约 30%、RUR 提升约 25%**，是 DAC + SAS + OTM Loss 共同作用的结果。

### M2.5: DCPO 进阶特性（可选）

| 任务 | 产出 |
|------|------|
| 与 `DAPO Dynamic Sampling` 组合 (DCPO + overlong shaping) | trainer 集成 |
| 与 `GSPO 序列级 clip_c` 思想组合 | dcpo.py 扩展 |
| Megatron 分布式训练支持 | megatron 适配 |
| 与规则化奖励（rule-based reward）深度集成 | reward_fn 适配 |
| **验收标准** | DCPO 进阶变体在数学/代码/工具调用等 benchmark 上达到或超过 DAPO 基线 |

### M3: RewardManager 集成（4 种评分函数 + LLM-as-Judge）

| 任务 | 产出 |
|------|------|
| 新建 `reward/` 子目录结构 (manager/registry/math/multiple_choice/string_match/llm_judge) | 目录骨架 |
| 实现 `registry.py` 的 `SCORE_REGISTRY` + `get_score_fn` | reward/registry.py |
| 实现 `math_score`（boxed/hash/last_number 三种抽取模式） | reward/math.py |
| 实现 `multiple_choice_score`（A/B/C/D 选项抽取 + 可配置正则） | reward/multiple_choice.py |
| 实现 `string_match_score`（严格 / 规范化两种模式） | reward/string_match.py |
| 实现 `LLMJudgeClient` + `llm_judge_score` + 默认 Judger Prompt | reward/llm_judge.py |
| 实现 `RewardManager` 主类（对齐 verl NaiveRewardManager 接口） | reward/manager.py |
| 在 `finetuning_args.py` 添加 `grpo_reward_*` 参数 | 参数定义 |
| Trainer 中将 `reward_fn` 替换为 `reward_manager` | trainer.py |
| Workflow 中加入 `create_reward_manager` 工厂方法 | workflow.py |
| 在 dataset 中加入 `ground_truth` 字段支持 | data 模块 |
| 单元测试：4 种评分函数在 toy 样本上的数值正确性 | tests/test_reward_*.py |
| 集成测试：DCPO + math 在小数据集上 loss 下降 | tests/test_dcpo_math_e2e.py |
| **验收标准** | ① 4 种 reward_type 均可独立运行；② 切换 reward_type 不影响算法逻辑；③ LLM-as-Judge 在网络异常时使用 fallback 不阻塞训练 |

### 里程碑依赖关系

```
M0 (GRPO)
   │
   ▼
M0.5 (DAPO) ──────┐
   │              │
   ▼              │
M1 (GSPO)         │
   │              │
   ├──────┬───────┤
   │      │       │
   ▼      ▼       ▼
M2 (DCPO)  M3 (RewardManager) ──→ M2.5 (DCPO 进阶)
   │              │
   └──────┬───────┘
          ▼
       M2.5 (DCPO 进阶)
```

> **说明**：M2 依赖于 M0 + M0.5，因为 DCPO 是 DAPO 的进一步改进，复用了 DAPO 的 DAC 思想和 DAPO 的大部分基础设施；**M3（RewardManager）独立于 M0~M2**，但所有算法的训练都依赖它提供奖励信号；M2.5 是可选的深度优化阶段。

---

## 7. RewardManager 设计（参考 verl）

> 第 6 节定义了 M0 ~ M2.5 的算法侧里程碑，本节补齐奖励侧的缺失：原 Plan 只给出 DAPO 的 `apply_overlong_penalty`（奖励整形），没有定义**如何计算每个 response 的原始奖励**。这里参考 [verl 项目](https://github.com/volcengine/verl) 的 `src/verl/workers/reward_manager/` + `src/verl/utils/reward_score/` 设计一个轻量级 `RewardManager`，覆盖 4 类常用场景。

### 7.1 总体设计

```
原始 response (string)            ground_truth (string)
        │                                 │
        └─────────────┬───────────────────┘
                      ▼
            ┌──────────────────┐
            │  RewardManager   │  根据 grpo_reward_type 分发:
            │   (manager.py)   │   math / multiple_choice / string_match / llm_judge
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  reward tensor   │  shape=[batch], dtype=float32, value∈[0,1]
            └────────┬─────────┘
                     │ (可选) DAPO overlong shaping
                     ▼
            ┌──────────────────┐
            │  group-relative  │  → 优势函数估计
            │   advantage      │
            └──────────────────┘
```

**核心约束**：
- 评分函数接口统一为 `score_fn(response: str, ground_truth: str) -> float`
- 返回值 ∈ `[0.0, 1.0]`（binary 模式下仅取 `0.0` 或 `1.0`）
- 单个样本评分失败不应阻断训练，需有 fallback 兜底

### 7.2 评分函数注册表 (`reward/registry.py`)

```python
from typing import Callable, Dict
from .math import math_score
from .multiple_choice import multiple_choice_score
from .string_match import string_match_score
from .llm_judge import llm_judge_score

# 类似 verl reward_score 的 __init__.py 路由
SCORE_REGISTRY: Dict[str, Callable] = {
    "math": math_score,
    "multiple_choice": multiple_choice_score,
    "string_match": string_match_score,
    "llm_judge": llm_judge_score,
}


def get_score_fn(reward_type: str) -> Callable:
    if reward_type not in SCORE_REGISTRY:
        raise ValueError(
            f"Unknown reward_type={reward_type}. "
            f"Available: {list(SCORE_REGISTRY.keys())}"
        )
    return SCORE_REGISTRY[reward_type]
```

### 7.3 数学答案评分 (`reward/math.py`)

支持 3 种答案抽取模式：`boxed` / `hash` / `last_number`，对应 verl 的 `reward_score/math.py`：

```python
import re
from typing import Optional


def _extract_boxed_answer(text: str) -> Optional[str]:
    """匹配 \\boxed{...} (支持嵌套大括号)"""
    # 简化版: 不处理嵌套, 用第一个 boxed 内容
    m = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return m.group(1).strip() if m else None


def _extract_hash_answer(text: str) -> Optional[str]:
    """匹配 GSM8K 风格的 #### 数字"""
    m = re.search(r"####\s*(-?\d[\d,\.]*)", text)
    if not m:
        return None
    return m.group(1).replace(",", "").rstrip(".")


def _extract_last_number(text: str) -> Optional[str]:
    """抓取最后一个数字 (兜底策略)"""
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


_EXTRACTORS = {
    "boxed": _extract_boxed_answer,
    "hash": _extract_hash_answer,
    "last_number": _extract_last_number,
}


def _normalize_answer(ans: str) -> str:
    """规范化: 去空格/逗号/末尾点号, 转 str"""
    return ans.replace(",", "").replace(" ", "").rstrip(".").lower()


def math_score(
    response: str,
    ground_truth: str,
    extract_mode: str = "boxed",
) -> float:
    """从 response 中抽取数学答案, 与 ground_truth 规范化比对
    返回: 1.0 (匹配) 或 0.0 (不匹配/无法抽取)
    """
    extractor = _EXTRACTORS.get(extract_mode, _extract_boxed_answer)
    pred = extractor(response)
    if pred is None:
        return 0.0
    return 1.0 if _normalize_answer(pred) == _normalize_answer(ground_truth) else 0.0
```

### 7.4 多选题评分 (`reward/multiple_choice.py`)

```python
import re
from typing import Optional


def _extract_choice(response: str, pattern: str) -> Optional[str]:
    """从 response 中抽取 A/B/C/D 选项"""
    m = re.search(pattern, response)
    if not m:
        return None
    # 兼容多组捕获 (groups 取首个非 None)
    for g in m.groups():
        if g is not None:
            return g.upper()
    return None


def multiple_choice_score(
    response: str,
    ground_truth: str,
    pattern: str = r"(?i)\\boxed\{\s*([A-D])\s*\}|answer\s*[:：]?\s*([A-D])",
) -> float:
    """抽取 A/B/C/D 选项, 与 ground_truth 比对
    ground_truth 期望为单个字母 (A/B/C/D)
    返回: 1.0 / 0.0
    """
    pred = _extract_choice(response, pattern)
    if pred is None:
        return 0.0
    return 1.0 if pred == ground_truth.strip().upper() else 0.0
```

### 7.5 字符串匹配 (`reward/string_match.py`)

```python
import re
import string


_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str, strict: bool = False) -> str:
    """规范化:
    strict=True:  仅去多余空白 (严格相等)
    strict=False: 去空白 + 去标点 + 小写 (宽松相等)
    """
    text = text.strip()
    if strict:
        return _WHITESPACE.sub(" ", text)
    text = _PUNCT_TABLE.sub("", text)
    return _WHITESPACE.sub("", text).lower()


def string_match_score(
    response: str,
    ground_truth: str,
    strict: bool = False,
) -> float:
    """字符串完全匹配 / 规范化匹配
    返回: 1.0 / 0.0
    """
    return 1.0 if _normalize(response, strict) == _normalize(ground_truth, strict) else 0.0
```

### 7.6 LLM-as-Judge (`reward/llm_judge.py`)

默认 prompt 让大模型比较标准答案和模型预测答案的一致性（用户要求）：

```python
import asyncio
import aiohttp
from typing import List, Optional


# ===== 默认 Judger Prompt (用户要求: 比较标准答案和模型预测答案的一致性) =====
DEFAULT_JUDGE_PROMPT = """You are a strict answer-checking judge.

You will be given TWO answers:
- **Ground Truth Answer**: the correct/reference answer
- **Model Prediction**: the answer produced by a language model

Your task: determine whether the **Model Prediction** is **semantically equivalent** to the **Ground Truth Answer**.

Rules:
1. Ignore minor formatting differences (whitespace, punctuation, casing).
2. For numbers, treat mathematically equivalent values as equal (e.g. 1/2 == 0.5).
3. For multi-choice or short factual answers, the prediction must match the ground truth.
4. If the prediction is partially correct but missing key information, return "no".
5. If the prediction is empty, irrelevant, or does not address the question, return "no".

Output ONLY one token, either:
- "yes"  → semantically equivalent
- "no"   → not equivalent

Do not output any explanation.

---

Ground Truth Answer:
{ground_truth}

Model Prediction:
{prediction}

Your verdict (yes/no):"""


class LLMJudgeClient:
    """异步 LLM 评判客户端 (兼容 OpenAI Chat Completions 协议)"""

    def __init__(
        self,
        url: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        timeout: int = 30,
        concurrency: int = 16,
        fallback_score: float = 0.0,
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
    ):
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency)
        self.fallback_score = fallback_score
        self.prompt_template = prompt_template

    async def _judge_one(
        self, session: aiohttp.ClientSession, prediction: str, ground_truth: str
    ) -> float:
        prompt = self.prompt_template.format(
            ground_truth=ground_truth.strip(),
            prediction=prediction.strip(),
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            async with self.semaphore:
                async with session.post(
                    self.url, json=payload, timeout=self.timeout
                ) as resp:
                    data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip().lower()
            return 1.0 if content.startswith("yes") else 0.0
        except Exception:
            # 评分失败时返回兜底分数, 不阻断训练
            return self.fallback_score

    async def judge_batch(
        self, predictions: List[str], ground_truths: List[str]
    ) -> List[float]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._judge_one(session, p, g)
                for p, g in zip(predictions, ground_truths)
            ]
            return await asyncio.gather(*tasks)


def llm_judge_score(
    response: str,
    ground_truth: str,
    judge_client: Optional[LLMJudgeClient] = None,
) -> float:
    """同步包装: 单条评分 (用于单元测试 / debug)
    实际训练中应使用 judge_client.judge_batch 异步并发
    """
    if judge_client is None:
        raise ValueError(
            "llm_judge_score requires judge_client. "
            "Use RewardManager + LLMJudgeClient in training."
        )
    return judge_client.judge_batch([response], [ground_truth])[0]
```

### 7.7 RewardManager 主类 (`reward/manager.py`)

接口对齐 verl 的 `NaiveRewardManager`：

```python
import torch
from dataclasses import dataclass
from typing import List, Optional

from .registry import get_score_fn
from .llm_judge import LLMJudgeClient, DEFAULT_JUDGE_PROMPT


@dataclass
class RewardInput:
    """单条评分输入 (对齐 verl DataProto 中的字段)"""
    response: str          # 模型生成的完整回答
    ground_truth: str      # 数据集中的标准答案
    prompt: Optional[str] = None  # 部分场景需要 prompt 上下文


class RewardManager:
    """统一奖励管理器
    类似 verl 的 NaiveRewardManager: __call__ 输入 List[RewardInput], 输出 Tensor
    """

    def __init__(self, finetuning_args):
        self.reward_type = finetuning_args.grpo_reward_type
        self.score_mode = finetuning_args.grpo_reward_score_mode
        self.args = finetuning_args

        # 注册评分函数
        if self.reward_type == "llm_judge":
            self.judge_client = LLMJudgeClient(
                url=finetuning_args.grpo_llm_judge_url,
                model=finetuning_args.grpo_llm_judge_model,
                max_tokens=finetuning_args.grpo_llm_judge_max_tokens,
                temperature=finetuning_args.grpo_llm_judge_temperature,
                timeout=finetuning_args.grpo_llm_judge_timeout,
                concurrency=finetuning_args.grpo_llm_judge_concurrency,
                fallback_score=finetuning_args.grpo_llm_judge_fallback_score,
            )
            # llm_judge 用专用函数 (需要 judge_client)
            from .llm_judge import llm_judge_score
            self.score_fn = llm_judge_score
        else:
            self.score_fn = get_score_fn(self.reward_type)

    def _score_one(self, response: str, ground_truth: str) -> float:
        """单条评分 (规则类函数直接调用)
        注意: llm_judge 走异步批量接口, 不会调用本方法.
        """
        if self.reward_type == "math":
            return self.score_fn(
                response, ground_truth,
                extract_mode=self.args.grpo_reward_math_extract_mode,
            )
        elif self.reward_type == "multiple_choice":
            return self.score_fn(
                response, ground_truth,
                pattern=self.args.grpo_reward_mc_pattern,
            )
        elif self.reward_type == "string_match":
            return self.score_fn(
                response, ground_truth,
                strict=self.args.grpo_reward_strict_match,
            )
        elif self.reward_type == "llm_judge":
            raise NotImplementedError(
                "_score_one 不支持 llm_judge, llm_judge 走 __call__ 异步批量接口"
            )
        else:
            return self.score_fn(response, ground_truth)

    def __call__(self, inputs: List[RewardInput]) -> torch.Tensor:
        """批量评分入口
        输入: List[RewardInput] (长度 = batch_size)
        输出: Tensor[batch_size], dtype=float32
        """
        if self.reward_type == "llm_judge":
            # 异步批量调用
            preds = [x.response for x in inputs]
            gts = [x.ground_truth for x in inputs]
            scores = asyncio.run(
                self.judge_client.judge_batch(preds, gts)
            )
        else:
            # 规则类: CPU 同步即可 (无 GPU 开销)
            scores = [
                self._score_one(x.response, x.ground_truth)
                for x in inputs
            ]
        return torch.tensor(scores, dtype=torch.float32)
```

### 7.8 Trainer 集成要点

RewardManager 与 trainer.py 的集成已在 3.4 节完成（`CustomGRPOTrainer.__init__` 接收 `reward_manager` 形参，并在 `training_step` 中调用 `_compute_rewards`）。这里仅补充**关键调用流程**：

```text
dataset  ──prompt, ground_truth──▶  trainer
                                       │
                  ┌────────────────────┘
                  ▼
            _rollout(prompts)        ── responses, log_probs, mask
                  │
                  ▼
            _decode_responses(responses)   ── response_strs
                  │
                  ▼
            _compute_rewards(prompts, response_strs, ground_truths)
                  │
                  │ 构造 List[RewardInput(response=r, ground_truth=g, prompt=p)]
                  ▼
            reward_manager(inputs)   ── Tensor[batch_size]
                  │
                  ▼
            (DAPO) apply_overlong_penalty
                  │
                  ▼
            compute_group_relative_advantage  /  (DCPO) compute_smoothed_advantage
                  │
                  ▼
            loss_fn(log_probs, ref_log_probs, advantages, mask)   ── scalar loss
```

**接口约定**：
- `RewardInput.response`   —— 模型生成的完整回答 (str)
- `RewardInput.ground_truth` —— 数据集标准答案 (str)
- `RewardInput.prompt`      —— 可选, 当前 4 种 score_fn 暂未使用, 留作未来扩展

### 7.9 Workflow 集成要点

RewardManager 与 workflow.py 的集成已在 3.5 节完成（`create_reward_manager` 工厂方法 + `run_grpo` 中通过 `create_reward_manager(finetuning_args)` 实例化后注入 `CustomGRPOTrainer`）。**关于注入策略的设计选择**：

| 方案 | 优点 | 缺点 |
|---|---|---|
| **A. 外部注入** (本文档采用) | 便于测试 (传入 mock RewardManager) | 需在 workflow 层显式构造 |
| B. 内部创建 | 调用更简单 | 难以替换, 单测复杂 |

**单元测试建议**：在 `tests/test_reward_manager.py` 中直接构造 `RewardManager(finetuning_args)` 而无需 trainer 上下文。

---
