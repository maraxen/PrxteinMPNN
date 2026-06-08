#!/bin/bash
#SBATCH --job-name=bench-smoke
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4008
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

export REFERENCE_PATH="${HOME}/repos/LigandMPNN"

# Detect GPU and set JAX_EXTRA + XLA_FLAGS (includes Blackwell autotuning workaround)
source scripts/engaging/_gpu_env.sh

uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

# Smoke test: single cell (L=76 → bucket=128, B=1, bf16, ar_sample only).
# With bucketing + XLA cache enabled this should compile in <5 min.
# A cold compile time >10 min indicates the fix did not land.
uv run python scripts/benchmarks/bench_prxteinmpnn_jax.py \
    --task ar_sample \
    --hardware Blackwell_SM120 \
    --output-json ../outputs/results/benchmarks/smoke_Blackwell_SM120_ar_sample.json \
    --pdb-dir tests/data \
    --seq-lens 76 \
    --batch-sizes 1 \
    --precision bf16 \
    --n-warmup 1 \
    --n-timed 3

# Smoke test: ligandmpnn_pytorch score_conditional (1 cell, L=76, B=1, fp32).
# Exercises the prody + featurize path that was previously uncovered.
uv run python scripts/benchmarks/bench_ligandmpnn_pytorch.py \
    --task score_conditional \
    --hardware Blackwell_SM120 \
    --output-json ../outputs/results/benchmarks/smoke_Blackwell_SM120_pytorch_score.json \
    --pdb-dir tests/data \
    --seq-lens 76 \
    --batch-sizes 1 \
    --precision fp32 \
    --n-warmup 1 \
    --n-timed 3
