#!/bin/bash
set -e
export PATH=/root/miniconda/bin:$PATH
source /root/miniconda/etc/profile.d/conda.sh
conda activate grpo
echo "Python: $(python --version)"

# Install PyTorch 2.7.1 + cu128 (compatible with CUDA 13.0 driver)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
"

echo "Torch install complete."
