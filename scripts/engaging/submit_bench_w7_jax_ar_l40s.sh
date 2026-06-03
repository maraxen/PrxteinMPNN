#!/bin/bash
#SBATCH --job-name=bench-w7-jax-ar-l40s
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# Wave 7: prxteinmpnn_jax ar_sample only.
# Reduced grid (76/500 x 1/16 x fp32 = 4 cells) + 3-hour subprocess timeout.
# stderr streams live so per-cell JIT compile progress is visible in the SLURM log.

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

source scripts/engaging/_gpu_env.sh

uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

uv run python scripts/benchmarks/bench_suite.py \
    --hardware L40s \
    --output-dir ../outputs/results/benchmarks \
    --pdb-dir tests/data \
    --seq-lens 76 500 \
    --batch-sizes 1 16 \
    --precision fp32 \
    --n-warmup 2 \
    --n-timed 5 \
    --tasks ar_sample \
    --subprocess-timeout 10800 \
    --skip-pytorch \
    --skip-colabdesign
