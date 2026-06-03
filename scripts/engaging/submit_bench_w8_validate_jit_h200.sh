#!/bin/bash
#SBATCH --job-name=bench-w8-jit-validate-h200
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4009
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# Wave 8: filter_jit validation — prxteinmpnn_jax score_conditional on H200.
# Validates the InferencePlan→eqx.Module + @eqx.filter_jit fix (commit 7f9ab56).
# Pre-fix baseline: L=76 → 164ms, L=500 → 596ms (38-106× slower than PyTorch).
# Expected post-fix: within 2-5× of PyTorch (4.3ms / 5.6ms at B=1).

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

source scripts/engaging/_gpu_env.sh

uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

uv run python scripts/benchmarks/bench_suite.py \
    --hardware H200 \
    --output-dir ../outputs/results/benchmarks \
    --pdb-dir tests/data \
    --seq-lens 76 500 \
    --batch-sizes 1 16 \
    --precision fp32 \
    --n-warmup 3 \
    --n-timed 10 \
    --tasks score_conditional \
    --subprocess-timeout 3600 \
    --skip-pytorch \
    --skip-colabdesign
