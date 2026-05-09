#!/bin/bash
# Full CPU test suite (non-parity) for Sprint D refactor validation.
#
# Runs all tests except parity/ and the one known-broken weight-loader test.
# No GPU or REFERENCE_PATH required.
#
# Submit (after sync):
#   just -g cluster-push-submit tev_design scripts/engaging/submit_test_suite_cpu.sh
#
#SBATCH --job-name=prx_test_suite
#SBATCH --partition=mit_normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/logs/slurm/test_suite_%j.out
#SBATCH --error=outputs/logs/slurm/test_suite_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outputs/logs/slurm

WORKSPACE_ROOT="${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/../uv.lock" && -f "${REPO_ROOT}/../pyproject.toml" ]]; then
  WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export JAX_PLATFORMS="cpu"

echo "===== prxteinmpnn full CPU test suite ====="
echo "JOB=${SLURM_JOB_ID:-local} HOST=$(hostname)"
echo "REPO_ROOT=${REPO_ROOT}"

cd "${WORKSPACE_ROOT}"
uv sync --group dev
cd "${REPO_ROOT}"

uv run python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

uv run pytest tests/ \
  --ignore=tests/parity \
  --deselect="tests/io/test_weights.py::test_smart_factory_model_loading[global_label_membrane_mpnn_v_48_020]" \
  -q --tb=short -ra

echo "===== Done ====="
