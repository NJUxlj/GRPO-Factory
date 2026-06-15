#!/bin/bash
set -e
export PATH=/root/miniconda/bin:$PATH
source /root/miniconda/etc/profile.d/conda.sh
conda activate grpo

echo "=== Installing GRPO-Factory dependencies ==="
cd /root/autodl-tmp/GRPO-Factory

# Install core dependencies (skip torch since already installed)
pip install --no-deps transformers datasets accelerate peft trl
pip install gradio matplotlib tyro einops numpy pandas scipy
pip install sentencepiece tiktoken modelscope hf-transfer safetensors
pip install av fire omegaconf packaging protobuf pyyaml pydantic
pip install uvicorn fastapi sse-starlette aiohttp

# Install GRPO-Factory in dev mode
pip install -e .

echo ""
echo "=== Configuring NCCL ==="
# Find NCCL from PyTorch's bundled nvidia packages
NCCL_PATH=$(python -c "
import os, glob
# Try nvidia-nccl-cu12 first
try:
    import nvidia.nccl
    p = os.path.dirname(nvidia.nccl.__file__)
    lib = os.path.join(p, 'lib')
    if os.path.isdir(lib):
        print(lib, end='')
except:
    pass
")
echo "NCCL lib path: $NCCL_PATH"

# Set env vars
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
[ -n "$NCCL_PATH" ] && export LD_LIBRARY_PATH="${NCCL_PATH}:${LD_LIBRARY_PATH}"

echo ""
echo "=== Verifying Installation ==="
python -c "
import torch
import transformers
import llamafactory
print(f'PyTorch:   {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'LLaMA-Factory: installed')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.0f} GB)')
"

# Test GRPO imports
python -c "
from llamafactory.train.grpo.loss import compute_grpo_loss, compute_dapo_loss, compute_gspo_loss, compute_dcpo_loss
from llamafactory.train.grpo.advantage import compute_group_relative_advantage, compute_smoothed_advantage
from llamafactory.train.grpo.reward.manager import RewardManager
print('All GRPO modules imported successfully!')
"

# Quick CUDA smoke test
python -c "
import torch
a = torch.randn(1000, 1000, device='cuda:0')
b = torch.randn(1000, 1000, device='cuda:0')
c = torch.mm(a, b)
print(f'CUDA matmul test: {c.sum().item():.2f} (OK)')
"

echo ""
echo "=== INSTALLATION COMPLETE ==="

# Save env setup
cat > /root/autodl-tmp/grpo_env.sh << 'ENVEOF'
#!/bin/bash
export PATH=/root/miniconda/bin:$PATH
source /root/miniconda/etc/profile.d/conda.sh
conda activate grpo
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
ENVEOF
chmod +x /root/autodl-tmp/grpo_env.sh
echo "Env script saved to /root/autodl-tmp/grpo_env.sh"
