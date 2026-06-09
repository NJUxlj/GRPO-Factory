## 背景

项目基于 verl 训练框架，需支持多种 Group-based RL 算法。GRPO 是基础组相对策略优化；DAPO 在其上引入 Clip-Higher、Dynamic Sampling、Token-level Loss、Overlong Reward Shaping 四项改进；GSPO 采用序列级优化，适合 MoE 大模型训练。

## 需求
- 加入 GRPO 训练方式
- 加入 DAPO 训练方式
- 加入 GSPO 训练方式

## 目标

1. 实现统一算法配置入口，通过 `algorithm.adv_estimator` 和 `policy_loss.loss_mode` 切换三种算法
2. GRPO：支持 group 采样（`rollout.n > 1`）、KL 损失约束、多种 `loss_agg_mode`
3. DAPO：支持非对称 clip（`clip_ratio_low/high`）、动态采样过滤（`filter_groups`）、token-mean 损失、overlong reward shaping
4. GSPO：支持序列级策略损失（`loss_mode: "gspo"`）、`clip_ratio_c`、Megatron 并行策略（TP/CP/EP）
5. 三种算法共享数据加载、Rollout、Reference、Trainer 基础模块，仅在损失计算和采样策略上分化
6. 提供对应 YAML 配置模板，用户可一键切换算法
