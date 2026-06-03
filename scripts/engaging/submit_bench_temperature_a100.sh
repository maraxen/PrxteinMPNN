#!/bin/bash
#SBATCH --job-name=bench-temp-a100
#SBATCH --partition=mit_gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# SLURM_SUBMIT_DIR is set by sbatch to the submission directory.
# BASH_SOURCE fallback covers sourcing the script directly (not via sbatch).
# Must submit from project root: cd ~/projects/tev_design/prxteinmpnn && sbatch scripts/engaging/submit_bench_temperature_a100.sh
_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

# Load GPU environment configuration
source "scripts/engaging/_gpu_env.sh"

# Ensure output directory exists
mkdir -p outputs/logs/slurm

# Run temperature array benchmark for A100
uv run python scripts/benchmarks/bench_temperature_array.py \
    --hardware A100 \
    --m-values 1 2 4 8 \
    --seq-len 76 \
    --n-warmup 10 \
    --n-timed 20 \
    --pdb-dir tests/data \
    --output-json "outputs/logs/slurm/bench_temperature_a100_${SLURM_JOB_ID}.json"

exit $?
