#!/bin/bash
#SBATCH --job-name=bench-w7-jax-ar-h200
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4009
#SBATCH --gres=gpu:1
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

uv sync --extra cuda --group benchmark --group dev

uv run python scripts/benchmarks/bench_suite.py \
    --hardware H200 \
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
