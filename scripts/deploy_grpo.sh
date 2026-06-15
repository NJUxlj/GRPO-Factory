#!/bin/bash
# GRPO-Factory Deployment Script for Server2 (AutoDL, 2x RTX 5090, CUDA 13.0)
set -e

LOG_FILE="/root/autodl-tmp/deploy_grpo.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "GRPO-Factory Deployment Started: $(date)"
echo "============================================"

# --- Step 1: Init conda ---
echo "[Step 1/5] Initializing conda..."
export CONDA_BASE="/root/miniconda"
export PATH="$CONDA_BASE/bin:$PATH"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true

# Ensure envs dir exists
mkdir -p /root/autodl-tmp/envs

# --- Step 2: Create grpo conda env ---
echo "[Step 2/5] Creating 'grpo' conda environment (Python 3.12)..."
if conda env list | grep -q "^grpo "; then
    echo "  grpo env already exists, skipping creation."
else
    conda create -n grpo python=3.12 -y
fi

conda activate grpo
echo "  Python: $(python --version)"

# --- Step 3: Install PyTorch (cu13 for CUDA 13.0) ---
echo "[Step 3/5] Installing PyTorch (cu13 wheels)..."
# Check if torch already installed
if python -c "import torch; print(torch.__version__)" 2>/dev/null; then
    echo "  PyTorch already installed: $(python -c 'import torch; print(torch.__version__)')"
else
    # Use PyTorch nightly cu13 wheels
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu131
    echo "  PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
fi

# Verify CUDA
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# --- Step 4: Configure NCCL ---
echo "[Step 4/5] Configuring NCCL environment..."
# Find nvidia-nccl-cu13 path
NCCL_LIB=$(python -c "import os; import nvidia.nccl; print(os.path.dirname(nvidia.nccl.__file__))" 2>/dev/null || echo "")
if [ -n "$NCCL_LIB" ]; then
    echo "  NCCL lib dir: $NCCL_LIB"
    export LD_LIBRARY_PATH="${NCCL_LIB}/lib:${LD_LIBRARY_PATH}"
else
    echo "  WARNING: nvidia-nccl-cu13 not found, trying pip install..."
    pip install nvidia-nccl-cu13 2>/dev/null || echo "  Could not install nvidia-nccl-cu13, using system NCCL"
fi

# Set NCCL env vars
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

# --- Step 5: Check GRPO-Factory code exists ---
echo "[Step 5/5] Verifying GRPO-Factory code..."
if [ -f "/root/autodl-tmp/GRPO-Factory/pyproject.toml" ]; then
    echo "  GRPO-Factory code exists, installing..."
    cd /root/autodl-tmp/GRPO-Factory
    pip install -e ".[all]" 2>/dev/null || pip install -e .
    echo "  GRPO-Factory installed."
else
    echo "  WARNING: /root/autodl-tmp/GRPO-Factory not found!"
    echo "  Please rsync the code from local first."
fi

# Save env vars for future sessions
cat > /root/autodl-tmp/grpo_env.sh << 'EOF'
#!/bin/bash
export CONDA_BASE="/root/miniconda"
export PATH="$CONDA_BASE/bin:$PATH"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate grpo
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
# NCCL lib path (adjust if needed)
NCCL_LIB=$(python -c "import os; import nvidia.nccl; print(os.path.dirname(nvidia.nccl.__file__))" 2>/dev/null || echo "")
[ -n "$NCCL_LIB" ] && export LD_LIBRARY_PATH="${NCCL_LIB}/lib:${LD_LIBRARY_PATH}"
EOF
chmod +x /root/autodl-tmp/grpo_env.sh

echo ""
echo "============================================"
echo "Deployment Complete: $(date)"
echo "To activate: source /root/autodl-tmp/grpo_env.sh"
echo "============================================"
