#!/bin/bash
#SBATCH --job-name=bench-w6-pytorch-l40s
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# Wave 6: ligandmpnn_pytorch score_conditional only.
# Retries the score_conditional task with randn+prody+symmetry fixes deployed (620008f).

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

export REFERENCE_PATH="${HOME}/repos/LigandMPNN"

uv sync --extra cuda --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

uv run python scripts/benchmarks/bench_suite.py \
    --hardware L40s \
    --output-dir ../outputs/results/benchmarks \
    --pdb-dir tests/data \
    --seq-lens 76 150 300 500 \
    --batch-sizes 1 4 16 \
    --precision fp32 \
    --n-warmup 10 \
    --n-timed 20 \
    --tasks score_conditional \
    --skip-prxteinmpnn \
    --skip-colabdesign
