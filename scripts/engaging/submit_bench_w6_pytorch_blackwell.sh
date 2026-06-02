#!/bin/bash
#SBATCH --job-name=bench-w6-pytorch-blackwell
#SBATCH --partition=pi_so3
#SBATCH --nodelist=node4008
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/logs/slurm/%j.out
#SBATCH --error=outputs/logs/slurm/%j.err

# Wave 6: ligandmpnn_pytorch score_conditional only.
# Retries the score_conditional task with randn+prody+symmetry fixes deployed (620008f).

set -euo pipefail

cd /orcd/pool/008/so3_shared/marielle/projects/tev_design/prxteinmpnn

# JAX Blackwell workaround (SM120 autotuning hang) — kept for colabdesign if ever added
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
    echo "Blackwell workaround active: XLA_FLAGS=${XLA_FLAGS}"
fi

export REFERENCE_PATH="${HOME}/repos/LigandMPNN"

uv sync --extra cuda --group benchmark --group dev

uv run python scripts/benchmarks/bench_suite.py \
    --hardware Blackwell_SM120 \
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
