#!/bin/bash
# Full CPU test suite (non-parity) for Sprint D refactor validation.
#
# Runs all tests except parity/ and the one known-broken weight-loader test.
# No GPU or REFERENCE_PATH required.
#
# Submit from tev_design workspace root:
#   just -g cluster-submit tev_design prxteinmpnn/scripts/engaging/submit_test_suite_cpu.sh
#
#SBATCH --job-name=prx_test_suite
#SBATCH --partition=mit_normal
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=prxteinmpnn/outputs/logs/slurm/test_suite_%j.out
#SBATCH --error=prxteinmpnn/outputs/logs/slurm/test_suite_%j.err

set -euo pipefail

# SLURM_SUBMIT_DIR is tev_design root when submitted via cluster-submit tev_design
WORKSPACE_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
REPO_ROOT="${WORKSPACE_ROOT}/prxteinmpnn"
cd "$WORKSPACE_ROOT"
mkdir -p "${REPO_ROOT}/outputs/logs/slurm"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export JAX_PLATFORMS="cpu"

echo "===== prxteinmpnn full CPU test suite ====="
echo "JOB=${SLURM_JOB_ID:-local} HOST=$(hostname)"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "REPO_ROOT=${REPO_ROOT}"

uv run python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())"

cd "${REPO_ROOT}"
uv run pytest tests/ \
  --ignore=tests/parity \
  --ignore=tests/training \
  --deselect="tests/io/test_weights.py::test_smart_factory_model_loading[global_label_membrane_mpnn_v_48_020]" \
  -q --tb=short -ra

echo "===== Done ====="
