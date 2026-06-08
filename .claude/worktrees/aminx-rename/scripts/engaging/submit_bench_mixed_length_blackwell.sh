#!/bin/bash
#SBATCH --job-name=bench-mixed-blackwell
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4008
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

# Blackwell XLA flag guard (required for SM120)
# Lines 38-45 from _gpu_env.sh already handle this, but we reinforce it here
if [[ ${SLURM_JOB_NODELIST:-} == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

uv run python scripts/benchmarks/bench_mixed_length.py \
    --hardware Blackwell \
    --lengths 76 150 300 500 \
    --n-warmup 10 \
    --n-timed 20 \
    --pdb-dir tests/data \
    --reference-path "${REFERENCE_PATH}" \
    --output-json outputs/results/benchmarks/Blackwell_mixed_length_bench.json
