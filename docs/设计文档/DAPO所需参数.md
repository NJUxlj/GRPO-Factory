# DAPO 训练参数文档

本文档定义了 verl 训练框架中 DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）算法所需的训练参数、配置和超参数。DAPO 是字节跳动发表的一篇工作，基于 GRPO 进行了多项改进。

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
| `data.gen_batch_size` | DAPO中用于动态采样的生成批量大小，通常大于train_batch_size |
| `data.enable_thinking` | 是否启用思维链生成 |
| `data.filter_overlong_prompts` | 是否过滤过长的prompts |
| `data.truncation` | 当input_ids或prompt长度超过max_prompt_length时是否截断，默认为'left' |

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
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 样本分割的子批量大小，用于更新 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 单次前向传播的每GPU微批量大小 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | 每GPU最大token数量 |
| `actor_rollout_ref.actor.grad_clip` | Actor更新的梯度裁剪阈值 |
| **`actor_rollout_ref.actor.clip_ratio_low`** | **DAPO的关键参数，对应论文中的ε_low，控制clip下界，通常设为0.2** |
| **`actor_rollout_ref.actor.clip_ratio_high`** | **DAPO的关键参数，对应论文中的ε_high，控制clip上界，通常设为0.28，用于Clip-Higher技术** |
| `actor_rollout_ref.actor.clip_ratio_c` | 额外的clip参数，用于某些变体 |
| `actor_rollout_ref.actor.entropy_coeff` | 熵系数权重 |
| `actor_rollout_ref.actor.use_kl_loss` | 是否使用KL损失，DAPO中应设为True |
| `actor_rollout_ref.actor.kl_loss_coef` | KL损失系数 |
| `actor_rollout_ref.actor.kl_loss_type` | KL散度计算方式 |
| **`actor_rollout_ref.actor.loss_agg_mode`** | **DAPO的关键参数，损失聚合模式，设为"token-mean"实现Token-level Loss** |
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
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM分配的GPU显存比例，如0.80 |
| `actor_rollout_ref.rollout.ignore_eos` | 是否忽略EOS token |
| `actor_rollout_ref.rollout.enforce_eager` | 是否禁用CUDAGraph |
| `actor_rollout_ref.rollout.free_cache_engine` | 是否offload KVCache |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | Rollout的TP大小 |
| `actor_rollout_ref.rollout.n` | 响应数量（采样次数） |
| `actor_rollout_ref.rollout.val_kwargs` | 验证相关参数（do_sample, temperature, top_p, n等） |

## 三、算法参数（DAPO核心技术）

### 3.1 基础算法参数

| 参数名称 | 参数含义 |
|---------|---------|
| `algorithm.adv_estimator` | 优势函数估计方法，设为grpo |
| `algorithm.gamma` | 折扣因子 |
| `algorithm.lam` | GAE估计器参数 |
| `algorithm.kl_penalty` | KL惩罚计算方式 |
| `algorithm.kl_ctrl.type` | KL控制器类型 |
| `algorithm.kl_ctrl.kl_coef` | KL系数 |
| `algorithm.use_kl_in_reward` | 是否在奖励中使用KL惩罚，DAPO中通常设为False |
| `algorithm.norm_adv_by_std_in_grpo` | 是否在GRPO中用标准差归一化优势 |

### 3.2 Dynamic Sampling（动态采样）参数

| 参数名称 | 参数含义 |
|---------|---------|
| **`algorithm.filter_groups.enable`** | **DAPO的关键参数，是否启用过滤组功能，设为True实现Dynamic Sampling** |
| **`algorithm.filter_groups.metric`** | **用于过滤的评估指标，可选acc、score、seq_reward、seq_final_reward等** |
| **`algorithm.filter_groups.max_num_gen_batches`** | **最大生成批次数，限制重复采样次数，非正值表示无上限** |

### 3.3 Overlong Reward Shaping 参数

| 参数名称 | 参数含义 |
|---------|---------|
| **`reward_model.overlong_buffer.enable`** | **是否启用超长奖励整形，设为True penalize过长输出** |
| **`reward_model.overlong_buffer.len`** | **缓冲区长度，当输出长度超过max_response_length - len时开始惩罚** |
| **`reward_model.overlong_buffer.penalty_factor`** | **惩罚因子，控制最大惩罚幅度** |

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

## 五、Megatron 策略相关参数（可选）

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | Pipeline并行大小 |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | 张量并行大小 |
| `actor_rollout_ref.actor.megatron.context_parallel_size` | 上下文并行大小 |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert并行大小（MoE模型） |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | 是否使用分布式checkpoint |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | 分布式checkpoint路径 |
| `actor_rollout_ref.actor.megatron.param_offload` | 参数offload |
| `actor_rollout_ref.actor.megatron.grad_offload` | 梯度offload |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | 优化器offload |

## 六、DAPO 四大核心技术及对应参数

| 技术名称 | 技术说明 | 关键配置参数 |
|---------|---------|-------------|
| **Clip-Higher** | 通过解耦clip范围（ε_low < ε_high），提升探索能力，避免熵坍缩 | `clip_ratio_low: 0.2`，`clip_ratio_high: 0.28` |
| **Dynamic Sampling** | 过滤掉全1或全0的group，提高训练效率和稳定性 | `filter_groups.enable: True`，`filter_groups.metric: acc`，`filter_groups.max_num_gen_batches: 10` |
| **Token-level Loss** | 在token级别而非sequence级别计算损失，更精细的优化 | `loss_agg_mode: "token-mean"` |
| **Overlong Reward Shaping** | 对接近长度限制但未超过的输出进行渐进式惩罚 | `overlong_buffer.enable: True`，`overlong_buffer.penalty_factor: 1.0` |

## 七、DAPO 特殊说明

1. **Clip-Higher**：DAPO将clip范围解耦为不对称的上下界，允许更容易地提升"exploration" token的概率
2. **Dynamic Sampling**：会自动重复采样直到有足够的合格group或达到最大生成次数
3. **损失聚合**：原始DAPO论文使用token-level loss，这是与其他算法的关键区别
4. **KL约束**：DAPO不通过在奖励函数中加入KL惩罚，而是直接将KL损失加入总损失
