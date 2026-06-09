# DCPO 训练参数文档

本文档定义了 verl 训练框架中 DCPO（Dynamic Clipping Policy Optimization）算法所需的训练参数、配置和超参数。DCPO 由百度（Baichuan Inc.）提出，是对 GRPO 与 DAPO 的进一步改进，通过引入动态自适应裁剪（Dynamic-Adaptive Clipping, DAC）和平滑优势标准化（Smooth Advantage Standardization, SAS），解决了零梯度问题并提升了样本利用率。

## 一、数据相关参数

| 参数名称 | 参数含义 |
|---------|---------|
| `data.train_files` | 训练集文件路径（本地路径或HDFS路径），格式为parquet。DCPO论文使用 DAPO-Math-17K 与 MATH 数据集 level 3-5 的合并，约 25k 数学问题 |
| `data.val_files` | 验证集文件路径 |
| `data.prompt_key` | prompt在数据集中的字段名，默认为'prompt' |
| `data.max_prompt_length` | 最大prompt长度，DCPO论文中设为 1024 |
| `data.max_response_length` | 最大response长度，DCPO论文中设为 3072（远小于 DAPO 原始的 20k） |
| `data.train_batch_size` | 一次训练迭代采样的全局批量大小（prompt数量），DCPO论文中设为 512 |
| `data.val_batch_size` | 一次验证迭代采样的批量大小 |
| `data.return_raw_input_ids` | 是否返回原始input_ids（不添加chat template） |
| `data.return_raw_chat` | 是否返回原始chat数据 |
| `data.truncation` | 当input_ids或prompt长度超过max_prompt_length时是否截断，默认为'error' |
| `data.filter_overlong_prompts` | 是否过滤过长的prompts |

## 二、Actor/Rollout/Reference 模型参数

### 2.1 通用配置

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.hybrid_engine` | 是否使用混合引擎，DCPO 基于 verl 框架，仅支持混合引擎 |
| `actor_rollout_ref.model.path` | Huggingface模型路径（本地或HDFS）。DCPO论文测试了 Qwen2.5-Math-1.5B-Instruct、Qwen2.5-3B（base）、Qwen2.5-Math-7B（math base）、Qwen2.5-14B（base） |
| `actor_rollout_ref.model.external_lib` | 需要额外导入的Python包 |
| `actor_rollout_ref.model.override_config` | 用于覆盖模型原始配置 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | 是否启用梯度检查点以节省显存 |

### 2.2 Actor 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.strategy` | 训练策略，可选fsdp或megatron |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 样本分割的子批量大小，用于更新，DCPO论文中设为 32 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 单次前向传播的每GPU微批量大小 |
| `actor_rollout_ref.actor.use_dynamic_bsz` | 是否在运行时自动调整批量大小 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | 每GPU最大token数量 |
| `actor_rollout_ref.actor.grad_clip` | Actor更新的梯度裁剪阈值 |
| **`actor_rollout_ref.actor.clip_ratio_low`** | **DCPO的关键参数，对应论文中的 ε_low，控制动态裁剪下界，DCPO论文中设为 0.16** |
| **`actor_rollout_ref.actor.clip_ratio_high`** | **DCPO的关键参数，对应论文中的 ε_high，控制动态裁剪上界，DCPO论文中设为 0.20** |
| **`actor_rollout_ref.actor.dual_clip`** | **DCPO关键参数，最大概率比率上限（Dual Clip，借鉴 PPO 思想），DCPO论文中设为 10.0，防止重要性权重过大** |
| `actor_rollout_ref.actor.entropy_coeff` | 计算PPO损失时熵的权重系数 |
| **`actor_rollout_ref.actor.use_kl_loss`** | **是否使用KL损失，DCPO与DAPO一致设为True（不通过奖励函数加KL惩罚）** |
| `actor_rollout_ref.actor.kl_loss_coef` | KL损失的系数，默认为 0.001 |
| `actor_rollout_ref.actor.kl_loss_type` | KL散度的计算方式 |
| **`actor_rollout_ref.actor.loss_agg_mode`** | **DCPO的关键参数，损失聚合模式，DCPO论文使用自定义的 "otm"（Only Token Mean）模式——只在单条 response 内对 token 求平均，不再跨 batch 求平均** |
| `actor_rollout_ref.actor.ppo_epochs` | 在一组采样数据上进行更新的epoch数量 |
| `actor_rollout_ref.actor.shuffle` | 当有多个epoch时是否打乱数据 |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | 序列并行大小 |
| `actor_rollout_ref.actor.calculate_entropy` | 是否计算熵 |
| `actor_rollout_ref.actor.fsdp_config` | FSDP配置 |

### 2.3 Actor 优化器参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.optim.lr` | 学习率 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 学习率预热步数比例 |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | 最小学习率比例 |
| `actor_rollout_ref.actor.optim.warmup_style` | 预热风格，可选constant或cosine |
| `actor_rollout_ref.actor.optim.total_training_steps` | 总训练步数，DCPO论文中设为 400 |

### 2.4 Reference 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | 计算ref_log_prob时每GPU的批量大小 |
| `actor_rollout_ref.ref.log_prob_use_dynamic_bsz` | 是否使用动态批量大小 |
| `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu` | 计算ref_log_prob时每GPU最大token数量 |
| `actor_rollout_ref.ref.fsdp_config` | Reference模型的FSDP配置，对于大于7B的模型建议开启offload |

### 2.5 Rollout 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.rollout.name` | Rollout引擎名称，可选vllm或hf |
| `actor_rollout_ref.rollout.temperature` | 采样温度参数，DCPO论文中设为 1.0 |
| `actor_rollout_ref.rollout.top_k` | top_k采样参数，DCPO论文中设为 -1（vllm 关闭 top_k） |
| `actor_rollout_ref.rollout.top_p` | top_p采样参数，DCPO论文中设为 1.0 |
| `actor_rollout_ref.rollout.response_length` | 生成的最大token数，DCPO论文中设为 3072 |
| `actor_rollout_ref.rollout.dtype` | Rollout模型参数类型，应与FSDP后端一致 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM分配的GPU显存比例 |
| `actor_rollout_ref.rollout.ignore_eos` | 是否忽略EOS token继续生成 |
| `actor_rollout_ref.rollout.enforce_eager` | 是否禁用CUDAGraph |
| `actor_rollout_ref.rollout.free_cache_engine` | 是否在rollout生成后offload KVCache |
| `actor_rollout_ref.rollout.load_format` | Rollout初始化方式 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | Rollout的TP大小 |
| `actor_rollout_ref.rollout.do_sample` | 是否采样 |
| **`actor_rollout_ref.rollout.n`** | **响应数量（采样次数），DCPO必须设置为大于1的值以进行 group 采样，DCPO论文中设为 16** |

## 三、算法参数（DCPO 核心技术）

### 3.1 基础算法参数

| 参数名称 | 参数含义 |
|---------|---------|
| **`algorithm.adv_estimator`** | **优势函数估计方法，DCPO必须设置为 dcpo 或 grpo（取决于实现）** |
| `algorithm.gamma` | 折扣因子 |
| `algorithm.lam` | GAE估计器中偏差和方差的权衡参数 |
| `algorithm.kl_penalty` | KL惩罚的计算方式 |
| `algorithm.kl_ctrl.type` | KL控制器类型 |
| `algorithm.kl_ctrl.kl_coef` | KL系数 |
| `algorithm.use_kl_in_reward` | 是否在奖励中使用KL惩罚，DCPO中设为False（直接使用KL损失） |
| `algorithm.norm_adv_by_std_in_grpo` | 是否在GRPO中用标准差归一化优势 |

### 3.2 Smooth Advantage Standardization（SAS）参数

DCPO 的核心创新之一是 SAS，其通过维护累计奖励统计量，结合当前步标准化与累计标准化进行加权平滑：

| 参数名称 | 参数含义 |
|---------|---------|
| **`algorithm.sas.enable`** | **DCPO的关键参数，是否启用平滑优势标准化（SAS）机制，设为True** |
| **`algorithm.sas.smoothing_factor`** | **SAS 平滑权重（论文公式 (6)），控制 new 与 total 标准化的加权比例，随训练步数 i 自动变化（推荐使用论文中的 i/i-1 比例）** |
| **`algorithm.sas.use_cumulative_std`** | **是否在计算优势时同时使用累计奖励分布统计量 μ_total^i 和 σ_total^i** |
| **`algorithm.sas.min_advantage_threshold`** | **用于最终优势选择的阈值，DCPO 选择两者中绝对值更小的优势以降低训练波动** |

### 3.3 Dynamic Sampling（可选）

DCPO 论文实现中未显式启用 DAPO 中的 Dynamic Sampling，而是通过 SAS 隐式解决全 0/全 1 优势问题。如下参数可选启用：

| 参数名称 | 参数含义 |
|---------|---------|
| `algorithm.filter_groups.enable` | 是否启用过滤组功能（DCPO 中默认 False） |
| `algorithm.filter_groups.metric` | 用于过滤的评估指标 |
| `algorithm.filter_groups.max_num_gen_batches` | 最大生成批次数 |

## 四、训练器参数

| 参数名称 | 参数含义 |
|---------|---------|
| `trainer.total_epochs` | 训练总epoch数 |
| `trainer.project_name` | WandB项目名称 |
| `trainer.experiment_name` | WandB实验名称 |
| `trainer.logger` | 日志方式，支持console和wandb |
| `trainer.nnodes` | 训练使用的节点数，DCPO论文使用 4 个节点 × 8 个 H20 |
| `trainer.n_gpus_per_node` | 每节点GPU数，DCPO论文为 8 |
| `trainer.save_freq` | 保存checkpoint的频率（按迭代次数） |
| `trainer.test_freq` | 验证的频率（按迭代次数） |
| `trainer.critic_warmup` | 实际策略学习前训练critic的迭代次数 |
| `trainer.default_local_dir` | 本地checkpoint保存路径 |
| `trainer.default_hdfs_dir` | HDFS checkpoint路径 |
| `trainer.val_before_train` | 是否在训练前运行验证 |

## 五、奖励模型参数（DCPO 通常使用规则化奖励）

| 参数名称 | 参数含义 |
|---------|---------|
| `reward_model.enable` | 是否启用奖励模型，DCPO 中通常设为 False（使用规则化奖励函数） |
| `reward_model.model.input_tokenizer` | 输入tokenizer |
| `reward_model.model.path` | 奖励模型路径 |
| `reward_model.micro_batch_size_per_gpu` | 每GPU微批量大小 |
| `reward_model.max_length` | 最大长度 |
| `reward_model.reward_manager` | 奖励管理器 |

### 5.1 DCPO 的规则化奖励设计

DCPO 论文采用基于规则的三值奖励：

| 奖励值 | 触发条件 |
|-------|----------|
| **+1** | 格式正确 且 答案正确 |
| **0**  | 格式正确 但 答案错误 |
| **-1** | 格式错误 |

通过 `custom_reward_function.path` 与 `custom_reward_function.name` 指定用户自定义奖励函数。

## 六、Megatron 策略相关参数（可选）

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | Pipeline并行大小 |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | 张量并行大小 |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | 上下文并行大小 |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert并行大小（MoE模型） |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | 是否使用分布式checkpoint |
| `actor_rollout_ref.actor.megatron.param_offload` | 参数offload |
| `actor_rollout_ref.actor.megatron.grad_offload` | 梯度offload |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | 优化器offload |

## 七、DCPO 三大核心技术及对应参数

| 技术名称 | 技术说明 | 关键配置参数 |
|---------|---------|-------------|
| **Dynamic-Adaptive Clipping (DAC)** | 用概率相关的动态裁剪边界替代固定裁剪，为低概率 token 提供更大的探索空间，保留高概率 token 的稳定裁剪 | `clip_ratio_low: 0.16`，`clip_ratio_high: 0.20`，`dual_clip: 10.0` |
| **Smooth Advantage Standardization (SAS)** | 同时使用当前步与累计步的奖励分布做加权平滑，缓解零优势问题，提升响应利用率（RUR） | `algorithm.sas.enable: True`，`algorithm.sas.smoothing_factor` |
| **Only Token Mean Loss (OTM)** | 在单条 response 内对 token 求均值，不再跨 batch 求均值，避免长序列主导 | `loss_agg_mode: "otm"` |

## 八、DCPO 与 GRPO/DAPO 对比

| 维度 | GRPO | DAPO | DCPO |
|------|------|------|------|
| 裁剪方式 | 固定对称 `clip_ratio` | 解耦固定非对称 `clip_ratio_low/high` | 动态自适应，与 q(x) 相关的非对称边界 |
| 裁剪边界 | `[1-ε, 1+ε]` | `[1-ε_low, 1+ε_high]` | 依 `q(x)` 计算，见公式 (4) |
| 优势标准化 | 当前步标准化 | 当前步标准化 + Dynamic Sampling | 当前步 + 累计步加权平滑（SAS） |
| 损失聚合 | SLM（seq-mean-token-mean） | TLM（token-mean） | OTM（只对单条 response 内 token 求平均） |
| 最大比率上限 | 无 | 无 | `dual_clip = 10` |
| KL 约束 | 通过奖励函数 | 通过损失函数 | 通过损失函数 |
| 训练效率 | 基准 | 慢（动态采样开销） | 比 DAPO 快约 1 倍 |

## 九、DCPO 特殊说明

1. **DAC 公式**：动态裁剪下界为 `0.5 + 0.5 * sqrt(max(1 - 4*ε_low/q(x), 0))`，上界为 `0.5 + 0.5 * sqrt(1 + 4*ε_high/q(x))`，其中 `q(x)` 为旧策略下该 token 的概率。
2. **裁剪边界与 GRPO 对齐**：通过将 `(q(x), r(x)) = (1/(1+ε), 1+ε)` 与 `(q(x), r(x)) = (1, 1-ε)` 代入公式，DCPO 论文取 ε=0.2 反推出 `ε_low=0.16`、`ε_high=0.20`，在高频 token 区域与 GRPO 的固定裁剪重合。
3. **dual_clip**：借鉴 PPO dual-clip 思路，将正负优势的 `r(x)` 都限制在 `[0, 10]`，防止自适应边界在极低概率区域过大引起梯度爆炸。
4. **OTM 损失**：DCPO 论文公式 (8) 仅对单条 response 内的 token 求平均，跨 response 不再求平均。
5. **SAS 平滑**：公式 (6) 使用 `(i-1)/i` 与 `1/i` 的权重将 `A_new` 与 `A_total` 加权；公式 (7) 取两者绝对值的较小者作为最终优势，缓解训练波动。
6. **优势类型**：`algorithm.adv_estimator` 在 DCPO 实现中可设为 `dcpo`（若 verl 已扩展）或 `grpo`（使用 DCPO 自定义的核心算法替换）。
7. **基于 verl 框架**：DCPO 在 verl 框架基础上实现，保留 DAPO 的大多数基础设施（如 hybrid engine、vLLM rollout 等）。
8. **硬件规模**：DCPO 论文实验在 32 × H20 GPU 上完成，单次训练 400 步。
