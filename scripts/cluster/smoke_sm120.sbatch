#!/bin/bash
#SBATCH --job-name=aminx-smoke-sm120
#SBATCH --partition=pi_so3
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=outputs/logs/slurm/smoke_sm120_%j.out

# SM120 Blackwell XLA workaround (mandatory — without this: 1170x slowdown)
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

_PROJ="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${_PROJ}"

echo "Node: $(hostname -s), XLA_FLAGS=${XLA_FLAGS:-<unset>}"
uv run python scripts/cluster/smoke_sm120.py
