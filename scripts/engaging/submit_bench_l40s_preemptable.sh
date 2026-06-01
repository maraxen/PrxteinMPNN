#!/bin/bash
#SBATCH --job-name=bench-full-l40s
#SBATCH --partition=mit_preemptable
#SBATCH --nodelist=node3203
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5:30:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design

export REFERENCE_PATH="${HOME}/repos/LigandMPNN"

uv run python prxteinmpnn/scripts/benchmarks/bench_suite.py \
    --hardware L40s \
    --output-dir outputs/results/benchmarks \
    --fixture-dir outputs/benchmark_fixtures \
    --pdb-dir prxteinmpnn/tests/data \
    --seq-lens 76 150 300 500 \
    --batch-sizes 1 4 16 \
    --precision bf16 fp32 \
    --n-warmup 10 \
    --n-timed 20
