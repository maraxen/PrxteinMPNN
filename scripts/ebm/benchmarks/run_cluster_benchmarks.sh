#!/usr/bin/env bash
# Cluster-side runner for the E11a/E11b/E11c ProteinEBM benchmark harnesses
# (backlog nodes #3307/#3308/#3309, EPIC #3294). Runs all three JAX-vs-PyTorch
# throughput/latency comparisons at the design spec's bucket-aligned lengths
# {64,128,256,512} against the real 85M-param ProteinEBM-x checkpoint.
#
# Expects (pushed separately, outside the aminx project's git tree, since
# neither is committed to the repo):
#   ~/proteinebm_bench_assets/ProteinEBM              (reference PyTorch repo)
#   ~/proteinebm_bench_assets/model_6_expert_frozen_1m_md.pt  (checkpoint)
#
# SM120/Blackwell workaround: the pi-so3-gpu/gpu-h200 myxcel presets already
# set XLA_FLAGS=--xla_gpu_shard_autotuning=false in their environment: this
# script also sets it explicitly (idempotent, safe) per ~/.claude/rules/CLUSTER.md
# so the job is correct even if invoked under a different preset by mistake.
#
# CUDA extras: `torch` lives under the `benchmark` extra and jax's CUDA
# plugin (bundling the matching libcudnn) lives under `cuda12` -- NEITHER is
# in the base dependency set, so a bare `uv run` (which implicitly syncs only
# the base set) silently drops both, even if a prior manual sync installed
# them. `import torch` then fails with `ImportError: libcudnn.so.9: cannot
# open shared object file` on GPU nodes (confirmed via a diagnostic job on
# node4007). Explicit `uv sync --extra cuda12 --extra benchmark` fixes this.
set -euo pipefail

ASSETS_DIR="${PROTEINEBM_ASSETS_DIR:-$HOME/proteinebm_bench_assets}"
CHECKPOINT="${ASSETS_DIR}/model_6_expert_frozen_1m_md.pt"
REFERENCE_REPO="${ASSETS_DIR}/ProteinEBM"
LENGTHS="${BENCHMARK_LENGTHS:-64,128,256,512}"
N_REPEATS="${BENCHMARK_N_REPEATS:-30}"
OUT_DIR="${BENCHMARK_OUT_DIR:-outputs/ebm_benchmarks}"

mkdir -p "${OUT_DIR}"

echo "=== uv sync --extra cuda12 --extra benchmark ==="
uv sync --extra cuda12 --extra benchmark

# Mandatory Blackwell/SM120 workaround (see CLUSTER.md; safe no-op elsewhere).
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_shard_autotuning=false}"

echo "=== Node: ${_NODE} | XLA_FLAGS=${XLA_FLAGS} ==="
echo "=== Checkpoint: ${CHECKPOINT} | Reference repo: ${REFERENCE_REPO} ==="
echo "=== Lengths: ${LENGTHS} | n_repeats: ${N_REPEATS} ==="

uv run python scripts/ebm/benchmarks/decoy_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/decoy_benchmark_full.json"

uv run python scripts/ebm/benchmarks/ddg_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/ddg_benchmark_full.json"

uv run python scripts/ebm/benchmarks/biasing_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/biasing_benchmark_full.json"

echo "=== All three benchmarks complete; results in ${OUT_DIR} ==="
