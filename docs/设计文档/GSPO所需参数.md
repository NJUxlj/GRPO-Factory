# GSPO 训练参数文档

本文档定义了 verl 训练框架中 GSPO（Group Sequence Policy Optimization）算法所需的训练参数、配置和超参数。GSPO 是千问（Qwen）团队发表的一篇工作，采用序列级别的优化而非token级别。

## 一、数据相关参数

| 参数名称 | 参数含义 |
|---------|---------|
| `data.train_files` | 训练集文件路径（本地路径或HDFS路径），格式为parquet |
| `data.val_files` | 验证集文件路径 |
| `data.prompt_key` | prompt在数据集中的字段名，默认为'prompt' |
| `data.max_prompt_length` | 最大prompt长度，所有prompt会被左填充到此长度 |
| `data.max_response_length` | 最大response长度，RL算法生成时最长到此长度 |
| `data.train_batch_size` | 一次训练迭代采样的全局批量大小（prompt数量） |
| `data.val_batch_size` | 一次验证迭代采样的批量大小 |
| `data.enable_thinking` | 是否启用思维链生成 |
| `data.filter_overlong_prompts` | 是否过滤过长的prompts |
| `data.truncation` | 截断方式，如'left' |

## 二、Actor/Rollout/Reference 模型参数

### 2.1 通用配置

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.hybrid_engine` | 是否使用混合引擎 |
| `actor_rollout_ref.model.path` | Huggingface模型路径（本地或HDFS） |
| `actor_rollout_ref.model.external_lib` | 需要额外导入的Python包 |
| `actor_rollout_ref.model.override_config` | 用于覆盖模型原始配置 |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | 是否启用梯度检查点以节省显存 |

### 2.2 Actor 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.strategy` | 训练策略，可选fsdp或megatron |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 样本分割的子批量大小，用于更新，是所有worker的全局大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 单次前向传播的每GPU微批量大小 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | 每GPU最大token数量 |
| `actor_rollout_ref.actor.grad_clip` | Actor更新的梯度裁剪阈值 |
| `actor_rollout_ref.actor.clip_ratio_low` | clip下界参数 |
| `actor_rollout_ref.actor.clip_ratio_high` | clip上界参数 |
| `actor_rollout_ref.actor.clip_ratio_c` | **GSPO的关键参数，额外的clip参数，用于控制GSPO序列级别的clip范围，通常设为3.0** |
| `actor_rollout_ref.actor.entropy_coeff` | 熵系数权重，**GSPO中通常设为0** |
| `actor_rollout_ref.actor.use_kl_loss` | 是否使用KL损失，GSPO中应设为True |
| `actor_rollout_ref.actor.kl_loss_coef` | KL损失系数，GSPO中通常设为0.1 |
| `actor_rollout_ref.actor.kl_loss_type` | KL散度计算方式 |
| **`actor_rollout_ref.actor.policy_loss.loss_mode`** | **GSPO的关键参数，设为"gspo"以启用GSPO序列级别策略损失** |
| `actor_rollout_ref.actor.calculate_entropy` | 是否计算熵 |
| `actor_rollout_ref.actor.ppo_epochs` | 更新epoch数 |
| `actor_rollout_ref.actor.shuffle` | 是否打乱数据 |
| `actor_rollout_ref.actor.fsdp_config` | FSDP配置 |

### 2.3 Actor 优化器参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.optim.lr` | 学习率 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 学习率预热步数比例 |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | 最小学习率比例 |
| `actor_rollout_ref.actor.optim.warmup_style` | 预热风格 |
| `actor_rollout_ref.actor.optim.total_training_steps` | 总训练步数 |

### 2.4 Reference 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | 计算ref_log_prob时每GPU的批量大小 |
| `actor_rollout_ref.ref.fsdp_config` | Reference模型的FSDP配置 |

### 2.5 Rollout 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.rollout.name` | Rollout引擎名称（vllm/hf） |
| `actor_rollout_ref.rollout.temperature` | 采样温度参数 |
| `actor_rollout_ref.rollout.top_k` | top_k采样参数 |
| `actor_rollout_ref.rollout.top_p` | top_p采样参数 |
| `actor_rollout_ref.rollout.response_length` | 生成的最大token数 |
| `actor_rollout_ref.rollout.dtype` | Rollout模型参数类型 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM分配的GPU显存比例 |
| `actor_rollout_ref.rollout.ignore_eos` | 是否忽略EOS token |
| `actor_rollout_ref.rollout.enforce_eager` | 是否禁用CUDAGraph |
| `actor_rollout_ref.rollout.free_cache_engine` | 是否offload KVCache |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | Rollout的TP大小 |
| `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | 重新计算log_prob时每GPU的微批量大小 |
| `actor_rollout_ref.rollout.n` | 响应数量（采样次数），GSPO中通常设为8 |
| `actor_rollout_ref.rollout.val_kwargs` | 验证相关参数（do_sample, temperature, top_p, n等） |

## 三、算法参数

| 参数名称 | 参数含义 |
|---------|---------|
| `algorithm.adv_estimator` | 优势函数估计方法，设为grpo |
| `algorithm.gamma` | 折扣因子 |
| `algorithm.lam` | GAE估计器参数 |
| `algorithm.kl_penalty` | KL惩罚计算方式 |
| `algorithm.kl_ctrl.type` | KL控制器类型 |
| `algorithm.kl_ctrl.kl_coef` | KL系数，GSPO中通常设为0.0（不通过奖励施加KL约束） |
| `algorithm.use_kl_in_reward` | 是否在奖励中使用KL惩罚，**GSPO中设为False** |
| `algorithm.norm_adv_by_std_in_grpo` | 是否在GRPO中用标准差归一化优势 |

## 四、训练器参数

| 参数名称 | 参数含义 |
|---------|---------|
| `trainer.total_epochs` | 训练总epoch数 |
| `trainer.project_name` | WandB项目名称 |
| `trainer.experiment_name` | WandB实验名称 |
| `trainer.logger` | 日志方式 |
| `trainer.nnodes` | 训练节点数 |
| `trainer.n_gpus_per_node` | 每节点GPU数 |
| `trainer.save_freq` | 保存checkpoint频率 |
| `trainer.test_freq` | 验证频率 |
| `trainer.default_local_dir` | 本地checkpoint路径 |
| `trainer.default_hdfs_dir` | HDFS checkpoint路径 |

## 五、Megatron 策略相关参数（GSPO常用）

GSPO通常在MoE模型上使用megatron策略进行大规模训练：

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | Pipeline并行大小，如1 |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | 张量并行大小，如2 |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | 上下文并行大小，如2 |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert并行大小（MoE模型），如8 |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | 是否使用分布式checkpoint |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | 分布式checkpoint路径 |
| `actor_rollout_ref.actor.megatron.param_offload` | 参数offload，大规模训练时设为True |
| `actor_rollout_ref.actor.megatron.grad_offload` | 梯度offload，大规模训练时设为True |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | 优化器offload，大规模训练时设为True |

## 六、奖励模型参数（可选）

| 参数名称 | 参数含义 |
|---------|---------|
| `reward_model.enable` | 是否启用奖励模型 |
| `reward_model.model.input_tokenizer` | 输入tokenizer |
| `reward_model.model.path` | 奖励模型路径 |
| `reward_model.micro_batch_size_per_gpu` | 每GPU微批量大小 |
| `reward_model.max_length` | 最大长度 |
| `reward_model.reward_manager` | 奖励管理器 |

## 七、GSPO 核心特性及对应参数

| 特性 | 说明 | 关键配置参数 |
|------|------|-------------|
| **Sequence-Level Optimization** | 在序列级别（而非token级别）计算importance ratio、分配奖励和优化 | `policy_loss.loss_mode: "gspo"` |
| **Sequence-Level Importance Ratio** | 使用序列似然推导importance ratio，理论上更严谨 | 内置于GSPO损失计算中 |
| **MoE Training Stability** | 天然解决MoE模型RL训练的稳定性问题 | 无需额外的稳定策略 |
| **Normalized Rewards** | 对多个response的奖励进行归一化计算advantages | 内置于GSPO算法中 |

## 八、GSPO 特殊说明

1. **损失模式**：必须设置 `actor_rollout_ref.actor.policy_loss.loss_mode = "gspo"` 以启用GSPO
2. **熵系数**：GSPO通常设置 `entropy_coeff = 0`，不显式添加熵正则化
3. **KL约束**：GSPO通过KL损失而非KL奖励施加约束，设置 `use_kl_loss = True`，`kl_coef = 0.0`
4. **clip_ratio_c**：GSPO特有的clip参数，用于序列级别的clip范围控制
5. **大规模训练**：GSPO在MoE模型（如Qwen3-30B-A3B）上表现出色，通常配合megatron策略使用
6. **序列长度**：GSPO可以处理更长的序列，max_response_length可以设为较大值（如8192）
7. **稳定性**：GSPO解决了MoE模型在RL训练中的不稳定问题，无需额外的稳定技巧
