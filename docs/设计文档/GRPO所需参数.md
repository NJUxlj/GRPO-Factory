# GRPO 训练参数文档

本文档定义了 verl 训练框架中 GRPO（Group Relative Policy Optimization）算法所需的训练参数、配置和超参数。

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
| `data.return_raw_input_ids` | 是否返回原始input_ids（不添加chat template），用于reward model的chat template与policy不同的场景 |
| `data.return_raw_chat` | 是否返回原始chat数据 |
| `data.truncation` | 当input_ids或prompt长度超过max_prompt_length时是否截断，默认为'error' |

## 二、Actor/Rollout/Reference 模型参数

### 2.1 通用配置

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.hybrid_engine` | 是否使用混合引擎，目前仅支持混合引擎 |
| `actor_rollout_ref.model.path` | Huggingface模型路径（本地或HDFS） |
| `actor_rollout_ref.model.external_lib` | 需要额外导入的Python包，用于注册模型或tokenizer到Huggingface系统 |
| `actor_rollout_ref.model.override_config` | 用于覆盖模型原始配置，主要是dropout |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | 是否为actor启用梯度检查点以节省显存 |

### 2.2 Actor 模型参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.strategy` | 训练策略，可选fsdp或megatron |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 样本分割的子批量大小，用于PPO/GRPO更新，是所有worker的全局大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 单次前向传播的每GPU微批量大小，类似梯度累积，用于显存与速度的权衡 |
| `actor_rollout_ref.actor.use_dynamic_bsz` | 是否在运行时自动调整批量大小 |
| `actor_rollout_ref.actor.ppo_max_token_len_per_gpu` | 每GPU最大token数量，计算方式为 n * max_prompt_length + max_response_length |
| `actor_rollout_ref.actor.grad_clip` | Actor更新的梯度裁剪阈值 |
| `actor_rollout_ref.actor.clip_ratio` | GRPO的clip范围，默认为0.2 |
| `actor_rollout_ref.actor.entropy_coeff` | 计算PPO损失时熵的权重系数 |
| `actor_rollout_ref.actor.use_kl_loss` | 是否使用KL损失，**GRPO必须设置为True** |
| `actor_rollout_ref.actor.kl_loss_coef` | KL损失的系数，默认为0.001 |
| `actor_rollout_ref.actor.kl_loss_type` | KL散度的计算方式，支持kl(k1)、abs、mse(k2)、low_var_kl(k3)和full |
| `actor_rollout_ref.actor.loss_agg_mode` | 损失聚合模式，可选"token-mean"、"seq-mean-token-sum"、"seq-mean-token-mean"，原始GRPO论文使用"seq-mean-token-mean" |
| `actor_rollout_ref.actor.ppo_epochs` | 在一组采样数据上进行GRPO更新的epoch数量 |
| `actor_rollout_ref.actor.shuffle` | 当有多个epoch时是否打乱数据 |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | 序列并行大小 |
| `actor_rollout_ref.actor.calculate_entropy` | 是否计算熵 |
| `actor_rollout_ref.actor.fsdp_config` | FSDP配置，包括wrap_policy、param_offload、grad_offload、optimizer_offload等 |

### 2.3 Actor 优化器参数

| 参数名称 | 参数含义 |
|---------|---------|
| `actor_rollout_ref.actor.optim.lr` | 学习率 |
| `actor_rollout_ref.actor.optim.lr_warmup_steps_ratio` | 学习率预热步数比例 |
| `actor_rollout_ref.actor.optim.min_lr_ratio` | 最小学习率比例（仅用于cosine预热） |
| `actor_rollout_ref.actor.optim.warmup_style` | 预热风格，可选constant或cosine |
| `actor_rollout_ref.actor.optim.total_training_steps` | 总训练步数 |

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
| `actor_rollout_ref.rollout.name` | Rollout引擎名称，可选hf或vllm |
| `actor_rollout_ref.rollout.temperature` | 采样温度参数 |
| `actor_rollout_ref.rollout.top_k` | top_k采样参数，0为hf rollout，-1为vllm rollout |
| `actor_rollout_ref.rollout.top_p` | top_p采样参数 |
| `actor_rollout_ref.rollout.response_length` | 生成的最大token数 |
| `actor_rollout_ref.rollout.dtype` | Rollout模型参数类型，应与FSDP后端一致 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | vLLM分配的GPU显存比例 |
| `actor_rollout_ref.rollout.ignore_eos` | 是否忽略EOS token继续生成 |
| `actor_rollout_ref.rollout.enforce_eager` | 是否禁用CUDAGraph，设为True时禁用 |
| `actor_rollout_ref.rollout.free_cache_engine` | 是否在rollout生成后offload KVCache，默认为True |
| `actor_rollout_ref.rollout.load_format` | Rollout初始化方式，可选dummy_hf、dummy_megatron、dummy_dtensor等 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | Rollout的TP（张量并行）大小，仅对vLLM有效 |
| `actor_rollout_ref.rollout.do_sample` | 是否采样，设为False时为贪心采样 |
| `actor_rollout_ref.rollout.n` | **响应数量（采样次数），GRPO必须设置为大于1的值以进行group采样** |

## 三、算法参数

| 参数名称 | 参数含义 |
|---------|---------|
| `algorithm.adv_estimator` | 优势函数估计方法，支持gae、grpo、reinforce_plus_plus，**GRPO必须设置为grpo** |
| `algorithm.gamma` | 折扣因子 |
| `algorithm.lam` | GAE估计器中偏差和方差的权衡参数 |
| `algorithm.kl_penalty` | KL惩罚的计算方式，支持kl、abs、mse、full |
| `algorithm.kl_ctrl.type` | KL控制器类型，如fixed |
| `algorithm.kl_ctrl.kl_coef` | KL系数 |

## 四、训练器参数

| 参数名称 | 参数含义 |
|---------|---------|
| `trainer.total_epochs` | 训练总epoch数 |
| `trainer.project_name` | WandB项目名称 |
| `trainer.experiment_name` | WandB实验名称 |
| `trainer.logger` | 日志方式，支持console和wandb |
| `trainer.nnodes` | 训练使用的节点数 |
| `trainer.n_gpus_per_node` | 每节点GPU数 |
| `trainer.save_freq` | 保存checkpoint的频率（按迭代次数） |
| `trainer.test_freq` | 验证的频率（按迭代次数） |
| `trainer.critic_warmup` | 实际策略学习前训练critic的迭代次数 |
| `trainer.default_local_dir` | 本地checkpoint保存路径 |

## 五、奖励模型参数（可选）

| 参数名称 | 参数含义 |
|---------|---------|
| `reward_model.enable` | 是否启用奖励模型，GRPO通常设为False（使用用户定义的奖励函数） |
| `reward_model.model.input_tokenizer` | 输入tokenizer，如果奖励模型的chat template与policy不同需要设置 |
| `reward_model.model.path` | 奖励模型路径 |
| `reward_model.micro_batch_size_per_gpu` | 每GPU微批量大小 |
| `reward_model.max_length` | 最大长度 |
| `reward_model.reward_manager` | 奖励管理器，默认为naive |

## 六、GRPO 特殊说明

1. **必须设置 `actor_rollout_ref.rollout.n > 1`**：GRPO的核心机制是对每个prompt生成多个response组成group
2. **必须设置 `algorithm.adv_estimator = grpo`**：指定使用GRPO优势估计
3. **必须设置 `actor_rollout_ref.actor.use_kl_loss = True`**：GRPO不通过奖励函数加入KL惩罚，而是直接将KL散度加入损失
4. **`data.train_batch_size * actor_rollout_ref.rollout.n`** 是总的response/trajectory数量
