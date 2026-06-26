# GRPO-Factory

[![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=social)](https://github.com/hiyouga/LLaMA-Factory/stargazers)
[![GitHub workflow](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml/badge.svg)](https://github.com/hiyouga/LLaMA-Factory/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/llamafactory)](https://pypi.org/project/llamafactory/)

> 基于 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 的 **Group Relative Policy Optimization** 强化学习微调框架。
> 一键实现 GRPO / DAPO / GSPO / DCPO 四种 RLHF 算法的大模型训练。

---

## 目录

- [核心特性](#核心特性)
- [支持的算法](#支持的算法)
- [Reward 评分系统](#reward-评分系统)
- [快速开始](#快速开始)
- [配置参数详解](#配置参数详解)
- [安装](#安装)
- [硬件需求](#硬件需求)
- [引用](#引用)
- [协议](#协议)

---

## 核心特性

- **四种主流 GRPO 变体算法**：GRPO、DAPO、GSPO、DCPO，一套框架全部覆盖。
- **模块化 Reward 评分**：内置数学验证、选择题匹配、字符串匹配、LLM-as-Judge 四种评分器，支持自定义扩展。
- **灵活的 Advantage 计算**：Group-relative normalization 与 SAS (Smoothed Advantage) 平滑。
- **多种 Loss 聚合模式**：token-mean、seq-mean-token-sum、seq-mean-token-mean、OTM。
- **完全兼容 LLaMA-Factory**：沿用其模型加载、数据处理、LoRA/全参微调、多 GPU 分布式训练能力。

---

## 支持的算法

| 算法 | 论文 | 关键特性 |
|------|------|---------|
| **GRPO** | [DeepSeekMath](https://arxiv.org/abs/2402.03300) | 对称 clip + group-relative advantage |
| **DAPO** | [DAPO](https://arxiv.org/abs/2503.14476) | 非对称 clip + 动态采样 + Overlong Shaping |
| **GSPO** | [GSPO](https://arxiv.org/abs/2502.00862) | 序列级重要性比率 + clip_c |
| **DCPO** | [DCPO](https://arxiv.org/abs/2502.20934) | DAC 动态 clip + SAS 平滑 + Dual Clip + OTM |

所有算法共享统一的 `grpo` 训练阶段，通过 `grpo_loss_mode` 参数一键切换。

---

## Reward 评分系统

| 评分器 | 参数值 | 适用场景 |
|--------|--------|---------|
| 数学验证 | `math` | GSM8K / MATH 等数学推理任务，支持 `\boxed{}` 提取与哈希匹配 |
| 选择题 | `multiple_choice` | 选择题格式 `A/B/C/D`，支持中英文、大小写不敏感 |
| 字符串匹配 | `string_match` | 精确匹配 / 宽松匹配（忽略空格、标点、大小写） |
| LLM 裁判 | `llm_judge` | 调用外部 LLM API 进行开放式答案评判 |

---

## 快速开始

### 1. 训练（LoRA 微调 GRPO）

```bash
llamafactory-cli train examples/train_lora/qwen3_lora_grpo.yaml
```

### 2. 切换到不同算法

```bash
# DAPO（非对称 clip + 动态采样）
llamafactory-cli train examples/train_lora/qwen3_lora_dapo.yaml

# GSPO（序列级重要性比率）
llamafactory-cli train examples/train_lora/qwen3_lora_gspo.yaml

# DCPO（DAC + SAS + Dual Clip）
llamafactory-cli train examples/train_lora/qwen3_lora_dcpo.yaml
```

### 3. 使用不同 Reward 类型

```bash
# 选择题评分
llamafactory-cli train examples/train_lora/qwen3_lora_grpo.yaml \
    grpo_reward_type=multiple_choice \
    dataset=grpo_aime_demo

# LLM-as-Judge 评分
llamafactory-cli train examples/train_lora/qwen3_lora_dcpo.yaml \
    grpo_reward_type=llm_judge \
    grpo_llm_judge_url=http://localhost:8000/v1 \
    grpo_llm_judge_model=Qwen3-4B-Instruct \
    grpo_llm_judge_api_key=EMPTY
```

### 4. 多 GPU 分布式训练

```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train examples/train_lora/qwen3_lora_dcpo_advanced.yaml
```

---

## 配置参数详解

### 算法核心参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `stage` | 训练阶段，必须为 `grpo` | `grpo` |
| `grpo_loss_mode` | 算法模式 | `grpo` / `dapo` / `gspo` / `dcpo` |
| `grpo_num_generations` | 每个 prompt 的采样数 | 整数，典型值 4~16 |
| `grpo_temperature` | 采样温度 | 浮点数，默认 1.0 |
| `grpo_clip_ratio` | PPO 裁剪范围 | 浮点数，默认 0.2 |
| `grpo_max_response_length` | 生成最大长度 | 整数，默认 2048 |

### Advantage 计算参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `grpo_norm_adv_by_std` | 是否对 advantage 标准差归一化 | `true` / `false` |
| `grpo_smoothed_advantage` | 是否启用 SAS 平滑 | `true` / `false` |
| `grpo_sas_temperature` | SAS 平滑温度 | 浮点数，默认 1.0 |

### KL 散度控制参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `grpo_use_kl_loss` | 是否加入 KL 惩罚 | `true` / `false` |
| `grpo_kl_coef` | KL 系数 | 浮点数，默认 0.001 |
| `grpo_kl_type` | KL 类型 | `kl` / `abs` / `mse` / `low_var_kl` |

### Loss 聚合参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `grpo_loss_agg_mode` | Loss 聚合方式 | `token-mean` / `seq-mean-token-sum` / `seq-mean-token-mean` / `otm` |

### Reward 参数

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `grpo_reward_type` | 评分器类型 | `math` / `multiple_choice` / `string_match` / `llm_judge` |
| `grpo_reward_math_extract_mode` | 数学答案提取模式 | `boxed` / `hash` / `last_number` |
| `grpo_reward_string_strict` | 字符串是否严格匹配 | `true` / `false` |
| `grpo_llm_judge_url` | LLM 裁判 API 地址，支持 `/v1` 或完整 `/v1/chat/completions` | 如 `http://localhost:8000/v1` |
| `grpo_llm_judge_model` | LLM 裁判模型名 | 如 `Qwen3-4B-Instruct` |
| `grpo_llm_judge_api_key` | LLM 裁判 API Key，本地无鉴权服务可留空 | 如 `EMPTY` 或真实 token |

### DAPO 专属参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `grpo_clip_ratio_low` | 下界 clip（非对称） | 需显式设置 |
| `grpo_clip_ratio_high` | 上界 clip（非对称） | 需显式设置 |
| `grpo_overlong_buffer_len` | 超长惩罚缓冲区长度 | 需显式设置 |
| `grpo_overlong_penalty_factor` | 超长惩罚系数 | 需显式设置 |

### DCPO 专属参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `grpo_use_dac` | 使用动态 clip（DAC） | `false` |
| `grpo_dac_lambda` | DAC 衰减因子 | 需显式设置 |
| `grpo_dac_max_clip` | DAC 最大 clip 值 | 需显式设置 |
| `grpo_dac_min_clip` | DAC 最小 clip 值 | 需显式设置 |
| `grpo_dcpo_dual_clip` | 启用 Dual Clip | `false` |
| `grpo_dcpo_dual_clip_c` | Dual Clip 下界 | 需显式设置 |
| `grpo_dcpo_alpha` | DCPO 混合权重 | 需显式设置 |

---

## 安装

### 依赖

| 必需项 | 推荐版本 |
|--------|---------|
| Python | >= 3.11 |
| PyTorch | 2.6.0+ |
| transformers | 4.50.0+ |
| datasets | 3.2.0+ |
| accelerate | 1.2.1+ |
| peft | 0.15.1+ |

### 从源码安装

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git GRPO-Factory
cd GRPO-Factory
pip install -e .
```

### CUDA 13.0 / RTX 5090 用户

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

---

## 硬件需求

GRPO 训练需要额外的显存用于 rollout 生成（vLLM 推理），以下为 LoRA + GRPO 的典型配置（bf16）：

| 模型大小 | 最低显存 | 推荐显存 |
|---------|---------|---------|
| 0.5B-2B | 8 GB | 16 GB |
| 4B-8B | 24 GB | 32 GB |
| 14B | 40 GB | 80 GB |
| 32B | 80 GB | 160 GB |

> 通过设置 `vllm_gpu_memory_utilization` 可调节 vLLM 推理与训练之间的显存分配比例。

---

## 引用

如果本项目对您有帮助，请引用 LLaMA-Factory：

```bibtex
@inproceedings{zheng2024llamafactory,
  title={LlamaFactory: Unified Efficient Fine-Tuning of 100+ Language Models},
  author={Yaowei Zheng and Richong Zhang and Junhao Zhang and Yanhan Ye and Zheyan Luo and Zhangchi Feng and Yongqiang Ma},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)},
  address={Bangkok, Thailand},
  publisher={Association for Computational Linguistics},
  year={2024},
  url={http://arxiv.org/abs/2403.13372}
}
```

## 协议

本仓库代码依照 [Apache-2.0](LICENSE) 协议开源。使用模型权重时请遵循对应的模型协议。
