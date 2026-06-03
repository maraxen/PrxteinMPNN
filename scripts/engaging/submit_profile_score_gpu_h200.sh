#!/bin/bash
#SBATCH --job-name=profile-score-gpu-h200
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4009
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# GPU bottleneck profile for score_conditional (L=76/B=1) on H200.
# Separates host-dispatch from GPU-exec time; reports compute efficiency.

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

uv sync --extra cuda --group benchmark --group dev

echo "===== PLAIN PROFILER (dispatch split + stage timing + efficiency) ====="
uv run python scripts/benchmarks/profile_score_gpu.py \
    --seq-len 76 --batch-size 1 --n 30

# Kernel-level stats if Nsight Systems is available on the compute node.
if command -v nsys >/dev/null 2>&1; then
    echo "===== NSYS KERNEL STATS ====="
    nsys profile --stats=true -o /tmp/score_prof -f true \
        uv run python scripts/benchmarks/profile_score_gpu.py \
        --seq-len 76 --batch-size 1 --n 10 2>&1 \
        | grep -iE -A60 "CUDA Kernel|gpukernsum|cuda_gpu_kern|Time \(%\)|Total Time" || true
else
    echo "===== nsys not available on node; relying on plain profiler + jax trace ====="
fi
