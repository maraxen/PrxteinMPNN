#!/usr/bin/env bash
# Reference-backed parity (requires Dauparas-style assets on disk).
#
# Usage (local or cluster working copy):
#   export REFERENCE_PATH=/absolute/path/to/ligandmpnn_reference_assets
#   bash .praxia/run_parity_heavy.sh
#
# Engaging: sync code first, then e.g.
#   just -g cluster-push-submit tev_design path/to/this/script.sh
# See ~/Projects/tev_design/CLAUDE.md for transparent cluster access.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${REFERENCE_PATH:-}" ]]; then
  echo "REFERENCE_PATH must be set to the directory containing reference checkpoints (e.g. model_params/*.pt)." >&2
  exit 1
fi

export PYTHONPATH="${ROOT}/scripts:${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

uv run pytest \
  tests/parity \
  tests/model/test_ligandmpnn_equivalence.py \
  -m parity_heavy \
  -v "$@"
