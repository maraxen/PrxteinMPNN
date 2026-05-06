#!/bin/bash
# Fast GPU repro: only tests marked ``parity_targeted`` (current parity_heavy red set).
#
# Same sync / REFERENCE_PATH / uv sync contract as ``submit_parity_heavy_ligand.sh``,
# but ~4 tests instead of the full heavy matrix (saves queue + wall time).
#
# Submit (after ``just push`` from tev_design):
#   sbatch scripts/engaging/submit_parity_targeted.sh
#
# Local (needs REFERENCE_PATH + assets):
#   bash scripts/engaging/submit_parity_targeted.sh
#
#SBATCH --job-name=prx_parity_tgt
#SBATCH --partition=mit_preemptable,pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/slurm/parity_targeted_%j.out
#SBATCH --error=outputs/logs/slurm/parity_targeted_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/logs/slurm

WORKSPACE_ROOT="${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/../uv.lock" && -f "${REPO_ROOT}/../pyproject.toml" ]]; then
  WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
fi

export REFERENCE_PATH="${REFERENCE_PATH:-/home/maarxaru/repos/LigandMPNN}"
export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRXTEIN_PARITY_TIER="${PRXTEIN_PARITY_TIER:-parity_heavy}"
export PYTEST_ADDOPTS="-W default${PYTEST_ADDOPTS:+ ${PYTEST_ADDOPTS}}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "===== parity_targeted (subset of parity_heavy failures) GPU ====="
echo "JOB=${SLURM_JOB_ID:-local} HOST=$(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "REFERENCE_PATH=${REFERENCE_PATH} (is_dir=$([[ -d "${REFERENCE_PATH}" ]] && echo yes || echo no))"

if [[ ! -d "${REFERENCE_PATH}" ]]; then
  echo "ERROR: REFERENCE_PATH is not a directory: ${REFERENCE_PATH}"
  exit 1
fi

cd "${WORKSPACE_ROOT}"
uv sync --extra cuda --group dev
cd "${REPO_ROOT}"

uv run python - <<'PY'
import jax
import jaxlib
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
print("devices", jax.devices())
PY

if [[ "${PRXTEIN_SKIP_DIAG:-0}" != "1" ]]; then
  echo "===== diag: protein feature edge stages ====="
  uv run python scripts/diag_protein_feature_parity.py 2>&1 || echo "WARNING: diag_protein_feature_parity.py exited non-zero"
fi

uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_targeted -v --tb=short -ra

echo "===== Done ====="
