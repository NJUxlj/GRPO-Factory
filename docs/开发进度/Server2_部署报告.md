# GRPO-Factory Server2 部署报告

> **部署日期**: 2026-06-11
> **目标服务器**: Server2 (AutoDL, 2×RTX 5090)
> **SSH**: `ssh -p 16531 root@connect.bjb2.seetacloud.com`

---

## 服务器环境

| 项目 | 详情 |
|------|------|
| GPU | 2× NVIDIA GeForce RTX 5090 (32GB each) |
| Driver | 580.76.05 |
| CUDA Driver | 13.0 |
| RAM | 754 GB |
| Disk | 200 GB (124 GB available) |
| OS | Linux (AutoDL container) |

## Conda 环境

| 项目 | 详情 |
|------|------|
| 环境名 | `grpo` |
| Python | 3.12.13 |
| PyTorch | 2.7.1+cu128 |
| Transformers | 5.2.0 |
| Conda base | `/root/miniconda` → `/root/autodl-tmp/miniconda3` |

## 项目路径

| 项目 | 路径 |
|------|------|
| 代码目录 | `/root/autodl-tmp/GRPO-Factory/` |
| 模型目录 | `/root/autodl-tmp/models/` |
| 环境脚本 | `/root/autodl-tmp/grpo_env.sh` |

## 激活环境

```bash
source /root/autodl-tmp/grpo_env.sh
# 或手动：
export PATH=/root/miniconda/bin:$PATH
source /root/miniconda/etc/profile.d/conda.sh
conda activate grpo
```

## 验证结果

- ✅ PyTorch 2.7.1+cu128 安装成功，CUDA 可用
- ✅ 2× RTX 5090 GPU 全部识别
- ✅ CUDA matmul 双卡测试通过
- ✅ GRPO/DAPO/GSPO/DCPO 四种 loss 函数全部可导入
- ✅ RewardManager（math/multiple_choice/string_match/llm_judge）全部可导入
- ✅ **46/46 单元测试全部通过**

## 已部署脚本

| 脚本 | 路径 |
|------|------|
| 环境探测 | `/root/autodl-tmp/probe.sh` |
| 环境部署 | `/root/autodl-tmp/deploy_grpo.sh` |
| PyTorch 安装 | `/root/autodl-tmp/fix_torch.sh` |
| 依赖安装 | `/root/autodl-tmp/install_grpo_deps.sh` |

## NCCL 配置

```bash
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
```
