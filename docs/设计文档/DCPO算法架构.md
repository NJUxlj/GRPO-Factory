# DCPO 算法架构

本文档系统性地梳理 DCPO（Dynamic Clipping Policy Optimization，动态裁剪策略优化）算法的整体架构、核心原理、关键模块以及与相关算法的对比。DCPO 由百度（Baichuan Inc.）的 Shihui Yang 等人在 2025 年 9 月提出（arXiv:2509.02333），代码开源在 [lime-RL/DCPO](https://github.com/lime-RL/DCPO/)，基于 verl（HybridFlow）框架实现。

## 一、背景与动机

### 1.1 RLVR 与 GRPO 系列算法的发展

RLVR（Reinforcement Learning from Verifiable Rewards，可验证奖励的强化学习）作为提升大语言模型（LLM）推理能力的关键技术，近年来备受关注。其核心思路是使用基于规则的最终结果奖励（如数学答案正确性）来优化 LLM。GRPO（DeepSeek-AI, 2024）系列算法是该方向的代表工作。

### 1.2 GRPO/DAPO 的两个核心局限

虽然 GRPO 系列算法（GRPO、DAPO、GSPO 等）在很多任务上表现优异，但近期研究发现两个关键局限：

1. **Token 级裁剪缺陷（Token Clipping Deficit）**：GRPO 与 DAPO 对所有 token 使用固定裁剪边界 `[1-ε, 1+ε]`（GRPO 对称）或 `[1-ε_low, 1+ε_high]`（DAPO 非对称）。这种固定边界不能适应 token 概率分布——对低概率（高熵）token 而言，绝对探索空间被严重压缩，限制了稀有但信息量大的 token 的学习信号传递。
2. **响应级效率问题（Response-Level Inefficiency）**：GRPO 在同一 prompt 的所有 response 奖励完全相同时（如全部正确或全部错误），标准化后优势全为 0，导致这些 response 被丢弃，超过一半的生成样本无法参与参数更新。DAPO 通过 Dynamic Sampling（重复采样）缓解这一问题，但带来严重的采样效率损失（生成 3~5 倍数据才达到相同训练步数）。

### 1.3 DCPO 的解决思路

DCPO 通过两大核心创新同时解决上述两个问题：

| 创新 | 解决问题 | 体现层面 |
|------|---------|----------|
| **Dynamic-Adaptive Clipping (DAC)** | Token Clipping Deficit | Token 级 |
| **Smooth Advantage Standardization (SAS)** | Response-Level Inefficiency | Response 级 |

此外，DCPO 还提出了 **OTM Loss**（Only Token Mean Loss），更合理地处理不同长度的 response。

## 二、算法核心创新概述

```
┌─────────────────────────────────────────────────────────┐
│                       DCPO 总览                          │
├─────────────────────────────────────────────────────────┤
│  输入：prompt q，组大小 G，旧策略 π_old                  │
│                                                          │
│  Step 1: 采样 G 个 response，计算规则化奖励 R_j          │
│  Step 2: SAC/SAS 计算平滑优势 Â_j                       │
│  Step 3: DAC 计算动态裁剪边界                           │
│  Step 4: OTM Loss 计算策略梯度                          │
│  Step 5: 更新参数 θ                                     │
└─────────────────────────────────────────────────────────┘
```

## 三、核心模块详解

### 3.1 Dynamic-Adaptive Clipping (DAC) - 动态自适应裁剪

#### 3.1.1 问题根源分析

从 importance sampling 方差控制的角度看，固定裁剪 `|r(x) - 1| ≤ ε` 仅控制概率比率 `r(x) = p(x)/q(x)` 与 1 的距离。然而：

\[
\operatorname{Var}_{x \sim q}\!\left[f(x)\,\frac{p(x)}{q(x)}\right] - \operatorname{Var}_{x \sim p}\!\left[f(x)\right] = \mathbb{E}_{x \sim p}\!\left[f(x)^2\left(\frac{p(x)}{q(x)} - 1\right)\right]
\]

当 `q(x) → 0`（低概率 token）时，`(p(x)/q(x) - 1)` 可能极大，但 fixed clipping 限制其绝对值，导致这些高熵 token 的学习信号被过度裁剪。

#### 3.1.2 DAC 的核心约束

DCPO 不直接约束 `r(x)`，而是在概率空间中施加约束：

\[
\bigl|(r(x) - 1)\,p(x)\bigr| \le \epsilon
\]

将 `p(x) = r(x) q(x)` 代入，并对 `r(x)` 求解一元二次不等式，得到**闭式动态裁剪边界**：

**下界**：
\[
r(x) \ge \tfrac12 + \tfrac12 \sqrt{\max\!\left(1 - \tfrac{4 \epsilon_{\text{low}}}{q(x)},\, 0\right)}
\]

**上界**：
\[
r(x) \le \tfrac12 + \tfrac12 \sqrt{1 + \tfrac{4 \epsilon_{\text{high}}}{q(x)}}
\]

#### 3.1.3 与 GRPO 边界对齐

DCPO 要求边界在两个特殊点 `(q, r)` 上与 GRPO 的固定边界重合：
- 高概率点：`(q, r) = (1/(1+ε_grpo), 1+ε_grpo)`
- 中概率点：`(q, r) = (1, 1-ε_grpo)`

代入上述两个方程并取 `ε_grpo = 0.2`，可解得：
- `ε_low = 0.16`
- `ε_high = 0.20`

DCPO 取这两个值，使得：
- 在高频 token 区域（`q(x) ≥ 1/(1+ε_high)` ≈ 0.83），DAC 边界与 GRPO 固定边界重合，保证稳定；
- 在低频 token 区域（`q(x) < 1/(1+ε_high)`），DAC 上界随 `q(x)` 减小而迅速扩展（∝ 1/√q），为稀有 token 提供更大的探索空间。

#### 3.1.4 Dual Clip 上限

为防止在极低概率（`q(x) → 0`）时 DAC 上界过大引发梯度爆炸，DCPO 借鉴 PPO dual-clip 思想，引入硬上限：

\[
r_{\max} = 10
\]

无论正负优势，`r(x)` 都限制在 `[0, 10]` 区间内。

#### 3.1.5 DAC 行为可视化分析

DCPO 论文 Figure 4 中展示了 DAC 与 fixed clipping 在不同 `q(x)` 区间的差异：

| 区间 | 固定裁剪宽度 | DAC 裁剪宽度 | 含义 |
|------|-------------|-------------|------|
| `q(x) ∈ [0, 0.1]` | 常数 0.4 | 随 `q(x)` 减小而扩大 | 低概率 token 获得更大探索空间 |
| `q(x) ∈ [0.1, 1]` | 常数 0.4 | 与 fixed 重合（高频区） | 高概率 token 保持稳定裁剪 |

**结论**：DAC 通过为低概率 token 提供与概率成反比的扩展探索空间，自然恢复了对稀有但高信息量 token 的学习能力。

### 3.2 Smooth Advantage Standardization (SAS) - 平滑优势标准化

#### 3.2.1 GRPO/DAPO 的优势计算及其问题

GRPO/DAPO 中第 `i` 步采样的优势计算公式：

\[
\hat{A}^{i,t}_{j} = \frac{R^i_j - \mu^i}{\sigma^i}
\]

其中 `μ^i, σ^i` 仅基于当前步同 prompt 的 `G` 个 response 计算。这一设计存在两个问题：

1. **零优势问题**：当 `G` 个 response 奖励全部相同时（如全对或全错），`σ = 0` → `Â = 0` → 该 prompt 的所有 response 均不参与参数更新。
2. **波动问题**：高熵采样会导致 `R^i_j` 高度倾斜，每步 `μ^i, σ^i` 波动大，标准化的优势可能反复正负翻转，破坏训练稳定性。

#### 3.2.2 累计标准化（A_total）

DCPO 提出：同一 prompt 的所有 response 在整个训练过程中的奖励分布，可视为来自同一个全局分布。定义累计标准化：

\[
\hat{A}^{i}_{\text{total}, j} = \frac{R^i_j - \mu^i_{\text{total}}}{\sigma^i_{\text{total}}}
\]

其中 `μ^i_total, σ^i_total` 基于截至第 `i` 步累计的 `G·i` 个 response 计算。

#### 3.2.3 双标准化平滑公式

为缓解两者各自的波动，DCPO 引入加权平滑：

\[
\hat{SA}^i_{\text{new}, j} = \frac{i-1}{i}\,\hat{A}^i_{\text{new}, j} + \frac{1}{i}\,\hat{A}^i_{\text{total}, j}
\]

\[
\hat{SA}^i_{\text{total}, j} = \frac{1}{i}\,\hat{A}^i_{\text{new}, j} + \frac{i-1}{i}\,\hat{A}^i_{\text{total}, j}
\]

其中 `1/i, (i-1)/i` 为随步数 `i` 变化的权重，前者更偏向累计分布，后者更偏向当前分布。

#### 3.2.4 最终优势：取绝对值较小者

为同时保留两者的稳定信息，DCPO 最终优势为：

\[
\hat{A}^i_j = \begin{cases} \hat{SA}^i_{\text{new}, j}, & \text{if } |\hat{SA}^i_{\text{new}, j}| < |\hat{SA}^i_{\text{total}, j}| \\ \hat{SA}^i_{\text{total}, j}, & \text{otherwise} \end{cases}
\]

#### 3.2.5 SAS 的额外效益：零优势不再等于零梯度

即使在当前步所有 response 优势为 0 的情况下，由于 `(i-1)/i < 1`，累计统计仍能提供非零信号（具体贡献为 `(1/i)·A^i_total`）。这意味着：

> **一旦 prompt 参与模型优化，其后续所有 response 都能参与参数更新，即使当前优势为 0。**

这是 DCPO 的核心"零梯度问题"解决方案：不再需要 DAPO 那种昂贵的 Dynamic Sampling 重复采样机制。

#### 3.2.6 累计统计的增量计算

为减少内存开销，DCPO 采用增量方式计算 `μ^i_total, σ^i_total`（论文公式 (29)(30)）：

\[
\mu^i_{\text{total}} = \frac{1}{i}\bigl(\mu^i_{\text{new}} + (i-1)\,\mu^i_{\text{old}}\bigr)
\]

\[
\sigma^{i\,2}_{\text{total}} = \frac{1}{i}\bigl(\sigma^{i\,2}_{\text{new}} + (i-1)\,\sigma^{i\,2}_{\text{old}} + \tfrac{i-1}{i}\,(\mu^i_{\text{old}} - \mu^i_{\text{new}})^2\bigr)
\]

只需存储当前步与历史步的统计量，无需保留所有历史 response 的奖励记录。

### 3.3 Only Token Mean Loss (OTM Loss)

#### 3.3.1 三种损失聚合模式对比

DCPO 论文明确对比了三种损失聚合方式：

**SLM（Sequence-Level Mean，GRPO 原始）**：
\[
\mathcal{T}_{\text{GRPO}} = \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\bigl(r_{i,t}\hat{A}_{i,t},\, \text{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}\bigr)
\]

缺点：将 advantage 除以 `G`，削弱 advantage 信号的相对关系。

**TLM（Token-Level Mean，DAPO 原始）**：
\[
\mathcal{T}_{\text{DAPO}} = \frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\min\bigl(r_{i,t}\hat{A}_{i,t},\, \text{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}})\hat{A}_{i,t}\bigr)
\]

缺点：长 response 在分母中权重更大，可能让低 advantage 的长 response 反超高 advantage 的短 response。

**OTM（Only Token Mean，DCPO 原创）**：
\[
\mathcal{T}_{\text{DCPO}} = \sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\bigl(r_{i,t}\hat{A}_{i,t},\, \text{clip}\bigl(r_{i,t}, 1-\varepsilon_{\text{low}}(q), 1+\varepsilon_{\text{high}}(q)\bigr)\hat{A}_{i,t}\bigr)
\]

特点：**只在单条 response 内对 token 求平均，不再跨 batch 求平均**。这保留了 advantage 的相对关系，同时让每条 response 内每个 token 获得相等权重。

#### 3.3.2 OTM 与 SLM/TLM 的数学差异举例

DCPO 论文示例：

| Response | Advantage | 长度 | SLM 权重 | TLM 权重 | OTM 权重 |
|----------|-----------|------|----------|----------|----------|
| A | 1.0 | 500 | 1/(G·500) | 500/(500+1500)=0.25 | 1/500 |
| B | 0.5 | 1500 | 1/(G·1500) | 1500/(500+1500)=0.75 | 1/1500 |

- **TLM 总贡献**：A = 1·0.25=0.25；B = 0.5·0.75=0.375。**B 反超 A**，不合理。
- **OTM 总贡献**：A = 1·(1/500)·500 = 1.0（按 token 等权）；B = 0.5·(1/1500)·1500 = 0.5。A 仍占主导，**保持 relative 关系**。

#### 3.3.3 DCPO 的整体损失函数

将 DAC 边界代入 OTM 损失结构，最终 DCPO 损失为：

\[
\mathcal{T}_{\text{DCPO}}(\theta) = \sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\min\!\left(r_{i,t}(\theta)\hat{A}_{i,t},\,\text{clip}\!\left(r_{i,t}(\theta),\, 1-\varepsilon_{\text{low}}(q),\, 1+\varepsilon_{\text{high}}(q)\right)\hat{A}_{i,t}\right)
\]

其中 `clip` 边界使用 DAC 动态计算的下/上界（公式 (4)），并以 `r_max = 10` 为硬上限。

## 四、整体算法流程

DCPO 单步训练的整体流程：

```
┌──────────────────────────────────────────────────────────┐
│                    DCPO 训练循环                          │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │ 1. Rollout 阶段                       │
        │   - 对每个 prompt 采样 G 个 response │
        │   - 记录 π_old 下的 token 概率 q(x) │
        │   - 记录响应文本 o_i                 │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │ 2. Reward 阶段                        │
        │   - 规则化打分: R ∈ {+1, 0, -1}      │
        │   - 计算当前步统计 μ_new, σ_new      │
        │   - 更新累计统计 μ_total, σ_total    │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │ 3. 优势计算阶段 (SAS)                 │
        │   - 计算 A_new = (R - μ_new)/σ_new  │
        │   - 计算 A_total = (R - μ_total)/σ_total │
        │   - 加权平滑: SA_new, SA_total       │
        │   - 取绝对值较小者: A_final          │
        └──────────────────────────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────────┐
        │ 4. Actor 训练阶段                     │
        │   - 计算 π_θ 下的新概率 p(x)         │
        │   - 计算 r(x) = p(x) / q(x)          │
        │   - 应用 DAC 动态裁剪                │
        │   - 应用 Dual Clip (≤ 10)            │
        │   - 计算 OTM Loss                    │
        │   - 反向传播更新 θ                   │
        └──────────────────────────────────────┘
                          │
                          ▼
                   进入下一轮训练
```

### 算法伪代码（简化版）

```python
# 论文公式综合，对应一个 prompt 的处理
def dcpo_step(prompt, old_probs_q, responses, rewards_R):
    G = len(responses)

    # Step 1: 更新累计统计
    mu_new = rewards_R.mean()
    sigma_new = rewards_R.std()
    mu_total = (mu_new + (i-1) * mu_old) / i
    sigma_total = compute_total_std(sigma_new, sigma_old, mu_new, mu_old, i)
    update_running_stats(mu_total, sigma_total)

    # Step 2: 计算两种优势
    A_new = (rewards_R - mu_new) / (sigma_new + eps)
    A_total = (rewards_R - mu_total) / (sigma_total + eps)

    # Step 3: SAS 平滑
    SA_new = (i-1)/i * A_new + 1/i * A_total
    SA_total = 1/i * A_new + (i-1)/i * A_total

    # Step 4: 取绝对值较小者
    A_final = torch.where(SA_new.abs() < SA_total.abs(), SA_new, SA_total)

    # Step 5: DAC 动态裁剪边界
    eps_low = 0.16
    eps_high = 0.20
    r_max = 10.0
    lower_bound = 0.5 + 0.5 * torch.sqrt(torch.clamp(1 - 4*eps_low/old_probs_q, min=0))
    upper_bound = 0.5 + 0.5 * torch.sqrt(1 + 4*eps_high/old_probs_q)
    lower_bound = torch.clamp(lower_bound, min=0)        # Dual Clip
    upper_bound = torch.clamp(upper_bound, max=r_max)    # Dual Clip

    # Step 6: 计算 ratio 与 clip
    new_probs_p = compute_new_probs(responses)
    ratio = new_probs_p / old_probs_q
    clipped_ratio = torch.clamp(ratio, lower_bound, upper_bound)

    # Step 7: OTM Loss（只对单条 response 内 token 求平均）
    pg_losses1 = -A_final * ratio
    pg_losses2 = -A_final * clipped_ratio
    pg_loss_per_token = torch.maximum(pg_losses1, pg_losses2)
    # 按 token 平均，只在 response 内
    response_loss = (pg_loss_per_token * mask).sum(dim=-1) / mask.sum(dim=-1)
    total_loss = response_loss.sum()  # 不再除以 G
```

## 五、DCPO 关键优势指标分析

### 5.1 Token Clipping Ratio（TCR）

定义：
\[
\text{TCR} = \frac{\text{被裁剪的 token 数}}{\text{总 token 数}}
\]

| 算法 | TCR 趋势 | 量级 |
|------|---------|------|
| GRPO | 模型相关：1.5B/3B 上升；7B/14B 下降 | 高 |
| DAPO | 持续上升 | 高（与 GRPO 同量级） |
| **DCPO** | **稳定低水平** | **比 GRPO/DAPO 低一个数量级** |

### 5.2 Response Utilization Ratio（RUR）

定义：
\[
\text{RUR} = \frac{\text{非零优势 response 数}}{\text{总 response 数}} \times 100\%
\]

DCPO 论文实验结果（Qwen2.5 系列 4 个模型）：

| 模型 | GRPO RUR | DCPO RUR | 提升 |
|------|---------|----------|------|
| Qwen2.5-Math-1.5B-Instruct | 45.6% | 67.1% | +21.5% |
| Qwen2.5-3B | 48.3% | 74.3% | +26.0% |
| Qwen2.5-Math-7B | 37.4% | 73.2% | +35.8% |
| Qwen2.5-14B | 43.9% | 72.4% | +28.5% |
| **平均** | **43.8%** | **71.8%** | **+28.0%（绝对） / +64%（相对）** |

> 关键结论：DCPO 让 **超过 70% 的生成 response 都能参与模型更新**，极大提升了样本利用率。

## 六、DCPO 与相关算法的完整对比

### 6.1 算法演进

```
GRPO (2024, DeepSeek)
    ↓ 局限：固定对称裁剪、零优势问题、序列级均值损失
DAPO (2025, ByteDance)
    ↓ 改进：Clip-Higher、Dynamic Sampling、TLM Loss
GSPO (2025, Qwen)
    ↓ 改进：序列级裁剪，解决 MoE 稳定性
DCPO (2025, Baichuan)
    ↑ 综合改进：DAC + SAS + OTM
```

### 6.2 关键技术对比

| 维度 | GRPO | DAPO | GSPO | **DCPO** |
|------|------|------|------|----------|
| 裁剪方式 | 固定对称 | 固定非对称 | 序列级固定 | **动态自适应** |
| 裁剪参数 | `ε = 0.2` | `ε_low=0.2, ε_high=0.28` | `ε_low=3e-4, ε_high=4e-4` | `ε_low=0.16, ε_high=0.20` |
| 上限保护 | 无 | 无 | 无 | **`r_max=10` (Dual Clip)** |
| 优势标准化 | 当前步 | 当前步 | 当前步 | **当前步 + 累计步 平滑** |
| 零优势处理 | 丢弃 | Dynamic Sampling | 丢弃 | **SAS 累计权重保留** |
| 损失聚合 | SLM | TLM | 序列级 | **OTM** |
| KL 约束 | 奖励函数 | 损失函数 | 损失函数 | 损失函数 |
| 训练效率 | 基准 | 较慢 | 较慢 | **比 DAPO 快约 1 倍** |

### 6.3 性能基准对比

DCPO 论文 Table 1（基于 4 个模型 × 4 个数学推理基准）：

| 模型 | 指标 | GRPO | DAPO | GSPO | **DCPO** |
|------|------|------|------|------|----------|
| Qwen2.5-Math-7B | AIME24 Avg@32 | 32.1 | 31.6 | 34.9 | **38.8** |
| Qwen2.5-14B | AIME25 Avg@32 | 10.5 | 15.3 | 9.9 | **19.0** |

DCPO 在所有 4 个模型、4 个基准（MATH500、AMC23、AIME24、AIME25）上稳定优于或持平于 GRPO/DAPO/GSPO。

## 七、DCPO 的代码实现要点

DCPO 的开源代码基于 [verl](https://github.com/volcengine/verl) 框架，主要修改集中在以下模块：

### 7.1 核心代码模块

| 模块 | 文件位置 | 功能 |
|------|----------|------|
| Dynamic Clip 计算 | `verl/trainer/ppo/core_algos.py` | 实现 DAC 闭式解与 Dual Clip |
| 优势标准化 | `verl/trainer/ppo/core_algos.py` | 实现 SAS（new + total 平滑） |
| 损失聚合 | `verl/trainer/ppo/core_algos.py` | 实现 OTM Loss |
| 算法注册 | `verl/trainer/ppo/ray_trainer.py` | 注册 `adv_estimator = "dcpo"` |
| 奖励函数 | 用户自定义 | 实现三值规则化奖励 |

### 7.2 关键实现细节

1. **`algorithm.adv_estimator` 的扩展**：需要在 verl 的核心算法文件中注册 `"dcpo"` 估计器，复用 GRPO 的训练流程，但替换优势计算函数。
2. **DAC 的向量化解法**：在 `core_algos.py` 中对 `r(x)` 和 `q(x)` 进行向量化运算：
   ```python
   low = 0.5 + 0.5 * torch.sqrt(torch.clamp(1 - 4 * eps_low / old_probs, min=0))
   high = 0.5 + 0.5 * torch.sqrt(1 + 4 * eps_high / old_probs)
   low = torch.clamp(low, max=high)  # 保证 low <= high
   clipped_ratio = torch.clamp(ratio, low, high)
   clipped_ratio = torch.clamp(clipped_ratio, min=0, max=r_max)  # Dual Clip
   ```
3. **SAS 的状态维护**：在 `RayTrainer` 中维护 `(mu_old, sigma_old, step_count)` 状态；每步结束后增量更新累计统计。
4. **OTM 损失的实现**：在原有 `loss_agg_mode` 中新增 `"otm"` 选项：
   ```python
   if loss_agg_mode == "otm":
       seq_loss = (loss_mat * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)  # token-mean
       loss = seq_loss.sum()  # 仅对 response 求和，不再除以 G
   ```
5. **训练基础设施**：保留 DAPO 的所有基础设施（如 hybrid engine、vLLM rollout、FSDP/Megatron 后端等）。

## 八、DCPO 的局限与展望

### 8.1 当前局限

1. **累计统计的初始化**：训练初期的累计统计量基于极少量数据，可能不稳定，但 `1/i` 权重使得早期影响有限。
2. **`r_max` 的硬上限敏感性**：`r_max = 10` 是基于经验选取，未在论文中详细讨论该超参数的敏感性。
3. **OTM 不再除以 G**：损失量级与 GRPO/TLM 不同，可能影响学习率设置，需要对应调整。
4. **实验规模有限**：仅在 4 个数学推理模型上验证，尚未在通用对话、代码等任务上广泛测试。

### 8.2 后续可能的研究方向

- **跨任务泛化**：将 DAC + SAS 思想迁移到代码生成、工具调用等 RLVR 任务；
- **自适应 `r_max`**：根据 token 概率分布动态调整 `r_max`，避免人工调参；
- **与 sequence-level 优化的结合**：DAC 与 GSPO 等序列级方法的融合；
- **更高效的累计统计**：用 EMA 或滑动窗口替代累计统计，进一步压缩存储。

## 九、参考文献

1. **DCPO 论文**：Shihui Yang et al., "DCPO: Dynamic Clipping Policy Optimization", arXiv:2509.02333, 2025.
2. **GRPO 论文**：DeepSeek-AI, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", arXiv:2402.03300, 2024.
3. **DAPO 论文**：Qiying Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", arXiv:2503.14476, 2025.
4. **GSPO 论文**：Chujie Zheng et al., "Group Sequence Policy Optimization", arXiv:2507.18071, 2025.
5. **verl 框架**：Guangming Sheng et al., "HybridFlow: A Flexible and Efficient RLHF Framework", arXiv:2409.19256, 2024.
6. **DCPO 代码库**：[https://github.com/lime-RL/DCPO/](https://github.com/lime-RL/DCPO/)

## 十、附录：核心公式速查表

| 编号 | 公式 | 含义 |
|------|------|------|
| (1) | \(\operatorname{Var}_{x \sim q}[f \cdot p/q] - \operatorname{Var}_{x \sim p}[f] = \mathbb{E}_{x \sim p}[f^2(p/q - 1)]\) | IS 方差膨胀 |
| (3) | \(|(r(x)-1)\,p(x)| \le \epsilon\) | DAC 约束 |
| (4) | \(\tfrac12 + \tfrac12\sqrt{\max(1-4\epsilon_{\text{low}}/q, 0)} \le r(x) \le \tfrac12 + \tfrac12\sqrt{1+4\epsilon_{\text{high}}/q}\) | DAC 闭式边界 |
| (5) | \(\hat{A}^i_{\text{total}, j} = (R^i_j - \mu^i_{\text{total}})/\sigma^i_{\text{total}}\) | 累计标准化优势 |
| (6) | \(\hat{SA}^i_{\text{new}} = \tfrac{i-1}{i}\hat{A}_{\text{new}} + \tfrac{1}{i}\hat{A}_{\text{total}}\) | SAS 平滑公式 |
| (7) | \(\hat{A}^i_j = \arg\min(|\hat{SA}_{\text{new}}|, |\hat{SA}_{\text{total}}|)\) | 最终优势 |
| (8) | \(\mathcal{T}_{\text{DCPO}} = \sum_i \tfrac{1}{|o_i|}\sum_t \min(r_{i,t}\hat{A}_{i,t}, \text{clip}(r_{i,t})\hat{A}_{i,t})\) | DCPO 损失（OTM） |
| (15) | \(R_j = +1/0/-1\) （规则化三值奖励） | 奖励函数 |

---

> **文档维护**：本文档应与 `DCPO所需参数.md` 配合阅读；前者为参数速查表，后者为算法架构与原理详解。
