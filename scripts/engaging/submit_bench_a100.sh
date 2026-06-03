#!/bin/bash
#SBATCH --job-name=bench-full-a100
#SBATCH --partition=mit_preemptable
#SBATCH --nodelist=node2002
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

export REFERENCE_PATH="${HOME}/repos/LigandMPNN"

source scripts/engaging/_gpu_env.sh
uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev

uv run python scripts/benchmarks/bench_suite.py \
    --hardware A100 \
    --output-dir ../outputs/results/benchmarks \
    --fixture-dir ../outputs/benchmark_fixtures \
    --pdb-dir tests/data \
    --seq-lens 76 150 300 500 \
    --batch-sizes 1 4 16 \
    --precision bf16 fp32 \
    --n-warmup 10 \
    --n-timed 20 \
    --tasks score_conditional

uv run python scripts/benchmarks/bench_suite.py \
    --hardware A100 \
    --output-dir ../outputs/results/benchmarks \
    --fixture-dir ../outputs/benchmark_fixtures \
    --pdb-dir tests/data \
    --seq-lens 76 150 300 500 \
    --batch-sizes 1 4 16 \
    --precision bf16 fp32 \
    --n-warmup 2 \
    --n-timed 5 \
    --tasks ar_sample
