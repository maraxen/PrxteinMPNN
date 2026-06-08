#!/bin/bash
#SBATCH --job-name=bench-dedup-l40s
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

set -euo pipefail

# SLURM_SUBMIT_DIR is set by sbatch to the submission directory.
# BASH_SOURCE fallback covers sourcing the script directly (not via sbatch).
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

export REFERENCE_PATH="${HOME}/repos/LigandMPNN"

source scripts/engaging/_gpu_env.sh
uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

uv run python scripts/benchmarks/bench_dedup_hetero.py \
    --hardware L40s \
    --output-json outputs/results/benchmarks/L40s_dedup_hetero_bench.json \
    --pdb-dir tests/data \
    --n-total 32 \
    --k-values 1,2,4,8,16,32 \
    --n-warmup 10 \
    --n-timed 20
