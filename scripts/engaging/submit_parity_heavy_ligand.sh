#!/bin/bash
# Slow LigandMPNN / parity_heavy suite (README + CLAUDE.md parity_heavy gate).
#
# Prerequisites on Engaging:
#   - Sync: from tev_design repo root, `just push` (see parent `Justfile`; default
#     remote is engaging:/home/maarxaru/projects/tev_design).
#   - Reference PyTorch LigandMPNN checkout on sys.path (default below).
#   - Heavy parity requires this file (see tests.parity.reference_utils):
#       ${REFERENCE_PATH}/model_params/ligandmpnn_v_32_020_25.pt
#     It is gitignored locally; copy once to the cluster (all checkpoints), e.g.:
#       ssh engaging 'mkdir -p /home/maarxaru/repos/LigandMPNN/model_params'
#       rsync -avz ./reference_ligandmpnn_clone/model_params/ \
#         engaging:/home/maarxaru/repos/LigandMPNN/model_params/
#
# Submit (from prxteinmpnn repo root on cluster, after sync):
#   export REFERENCE_PATH=/home/maarxaru/repos/LigandMPNN   # optional if default matches
#   sbatch scripts/engaging/submit_parity_heavy_ligand.sh
#
# For a short repro of the current red tests only, use instead:
#   sbatch scripts/engaging/submit_parity_targeted.sh
#
# Local dry-run (same pytest line; needs REFERENCE_PATH locally if not skipping):
#   bash scripts/engaging/submit_parity_heavy_ligand.sh
#
# Diagnostics: runs ``scripts/diag_protein_feature_parity.py`` before pytest (stage
# comparison + stack versions). Set ``PRXTEIN_SKIP_DIAG=1`` to skip. Unbuffered
# Python is enabled so Slurm ``.out`` receives line-by-line pytest progress.
#
#SBATCH --job-name=prx_parity_heavy
#SBATCH --partition=mit_preemptable,pi_so3
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/logs/slurm/parity_heavy_ligand_%j.out
#SBATCH --error=outputs/logs/slurm/parity_heavy_ligand_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/logs/slurm

# Monorepo uv.lock lives next to prxteinmpnn/ (tev_design workspace root).
WORKSPACE_ROOT="${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/../uv.lock" && -f "${REPO_ROOT}/../pyproject.toml" ]]; then
  WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
fi

# Default Engaging layout (override with sbatch --export=ALL,REFERENCE_PATH=/other/path)
export REFERENCE_PATH="${REFERENCE_PATH:-/home/maarxaru/repos/LigandMPNN}"
export PYTHONPATH="${REPO_ROOT}/scripts:${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PRXTEIN_PARITY_TIER="${PRXTEIN_PARITY_TIER:-parity_heavy}"
export PYTEST_ADDOPTS="-W default${PYTEST_ADDOPTS:+ ${PYTEST_ADDOPTS}}"
# Prefer GPU backend on CUDA nodes (jax[cuda12] from workspace sync).
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "===== parity_heavy (tests/parity + test_ligandmpnn_equivalence) GPU ====="
echo "JOB=${SLURM_JOB_ID:-local} HOST=$(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "REFERENCE_PATH=${REFERENCE_PATH} (is_dir=$([[ -d "${REFERENCE_PATH}" ]] && echo yes || echo no))"

if [[ ! -d "${REFERENCE_PATH}" ]]; then
  echo "ERROR: REFERENCE_PATH is not a directory: ${REFERENCE_PATH}"
  echo "  Fix: export REFERENCE_PATH=/path/to/LigandMPNN then sbatch --export=ALL,REFERENCE_PATH,..."
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

# Stage-level JAX vs PyTorch edge comparison (see scripts/diag_protein_feature_parity.py).
if [[ "${PRXTEIN_SKIP_DIAG:-0}" != "1" ]]; then
  echo "===== diag: protein feature edge stages (stdout, non-fatal on error) ====="
  uv run python scripts/diag_protein_feature_parity.py 2>&1 || echo "WARNING: diag_protein_feature_parity.py exited non-zero"
fi

uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v --tb=short -ra

echo "===== Done ====="
