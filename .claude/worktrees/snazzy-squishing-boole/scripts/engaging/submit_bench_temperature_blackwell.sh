#!/bin/bash
#SBATCH --job-name=bench-temp-blackwell
#SBATCH --partition=pi_so3
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# SLURM_SUBMIT_DIR is set by sbatch to the submission directory.
# BASH_SOURCE fallback covers sourcing the script directly (not via sbatch).
# Must submit from project root: cd ~/projects/tev_design/prxteinmpnn && sbatch scripts/engaging/submit_bench_temperature_blackwell.sh
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"
set -euo pipefail

# Load GPU environment configuration
source "scripts/engaging/_gpu_env.sh"
uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
source scripts/engaging/_cudnn_path.sh

# Blackwell workaround: mandatory XLA flag guard
# SM120 (Blackwell) hangs on XLA autotuning. Without the flag: 9.56e-04 s/mol-step.
# With it: 8.18e-07 — a 1170× difference. See CLUSTER.md section 2.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

# Ensure output directory exists
mkdir -p outputs/logs/slurm

# Run temperature array benchmark for Blackwell
uv run python scripts/benchmarks/bench_temperature_array.py \
    --hardware Blackwell \
    --m-values 1 2 4 8 \
    --seq-len 76 \
    --n-warmup 10 \
    --n-timed 20 \
    --pdb-dir tests/data \
    --output-json "outputs/logs/slurm/bench_temperature_blackwell_${SLURM_JOB_ID}.json"

exit $?
