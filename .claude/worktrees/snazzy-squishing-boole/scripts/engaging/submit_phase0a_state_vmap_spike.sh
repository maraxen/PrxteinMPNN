#!/bin/bash
# Phase 0a spike (roadmap §227): state_vmap exact path vs vmap-of-single-state —
# `tests/sampling/spikes/test_state_vmap_exact_spike.py` under `parity_fast`, and
# optionally `parity_heavy` when `REFERENCE_PATH` is a directory (same sbatch script).
#
# Submit from this repository root (prxteinmpnn):
#   sbatch scripts/engaging/submit_phase0a_state_vmap_spike.sh
#
# Sync: the parent monorepo `tev_design/Justfile` target `just push` syncs the
# workspace to Engaging; adjust `engaging_dir` there if your remote checkout path
# differs.
#
# Local / dry-run (runs the same pytest as the batch job, no sbatch wrapper):
#   bash scripts/engaging/submit_phase0a_state_vmap_spike.sh
#
# Optional GPU (not required for this spike): add to the sbatch invocation, e.g.
#   sbatch --gres=gpu:1 scripts/engaging/submit_phase0a_state_vmap_spike.sh
# or uncomment the `#SBATCH --gres=gpu:1` line below and tune mem/CPU for GPU JAX.

#SBATCH --job-name=prx_phase0a_sv_spike
#SBATCH --partition=mit_preemptable,pi_so3
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/logs/slurm/phase0a_state_vmap_spike_%j.out
#SBATCH --error=outputs/logs/slurm/phase0a_state_vmap_spike_%j.err
# To request a GPU instead of CPU-only default, uncomment:
# #SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

mkdir -p outputs/logs/slurm

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
# Surface JAX/XLA HLO narrative warnings in SLURM and local logs (pytest default hides UserWarning).
export PYTEST_ADDOPTS="-W default${PYTEST_ADDOPTS:+ ${PYTEST_ADDOPTS}}"

echo "===== Phase 0a state_vmap spike (parity_fast) ====="
echo "JOB=${SLURM_JOB_ID:-local} HOST=$(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"

uv run pytest tests/sampling/spikes/test_state_vmap_exact_spike.py -m parity_fast -q

if [[ -n "${REFERENCE_PATH:-}" && -d "${REFERENCE_PATH}" ]]; then
  echo "===== Phase 0a state_vmap spike (parity_heavy, REFERENCE_PATH set) ====="
  uv run pytest tests/sampling/spikes/test_state_vmap_exact_spike.py -m parity_heavy -q
else
  echo "REFERENCE_PATH unset or not a directory; skipping parity_heavy (opt-in)."
fi

echo "===== Done ====="
