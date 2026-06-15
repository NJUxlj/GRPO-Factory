#!/bin/bash
# Probe script for server2 environment
echo "=== HOST ==="
hostname

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi FAILED"

echo "=== CUDA Driver ==="
nvidia-smi 2>/dev/null | grep "CUDA Version" || echo "CUDA version not found"

echo "=== nvcc ==="
nvcc --version 2>/dev/null || echo "nvcc NOT FOUND"

echo "=== Conda ==="
conda --version 2>/dev/null || echo "conda NOT FOUND"

echo "=== Conda Envs ==="
conda env list 2>/dev/null || echo "no envs"

echo "=== Disk (/root/autodl-tmp) ==="
df -h /root/autodl-tmp 2>/dev/null || echo "disk check failed"

echo "=== Memory ==="
free -h

echo "=== PATH ==="
echo "$PATH"

echo "=== CONDA PATH ==="
ls -la /root/miniconda 2>/dev/null || echo "no /root/miniconda"
ls -la /root/autodl-tmp/miniconda 2>/dev/null || echo "no /root/autodl-tmp/miniconda"
ls -la /root/autodl-tmp/miniconda3 2>/dev/null || echo "no /root/autodl-tmp/miniconda3"
