#!/usr/bin/env bash
# Cluster-side runner for the E11a/E11b/E11c/E11d ProteinEBM benchmark
# harnesses (backlog nodes #3307/#3308/#3309/E11d, EPIC #3294). Runs all four
# JAX-vs-PyTorch throughput/latency comparisons at the design spec's
# bucket-aligned lengths {64,128,256,512} against the real 85M-param
# ProteinEBM-x checkpoint.
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
#
# jax/jaxlib grad-path regression (decoy/ddg only): jaxlib 0.10.2's XLA:GPU
# compiler crashes on `-jax.grad(energy)` (the `score_grad_ms` metric) on
# every modern GPU generation tested (Blackwell/H100/A100/L40S) --
# 'scf.if' control-flow shape mismatch, confirmed via a version bisection
# (see .praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md §11) to be
# a genuine 0.9.2->0.10.2 regression, not present in 0.8.0-0.9.2. Pin these
# two scripts to the last known-good version via an ad-hoc `uv run --with`
# override -- NOT a pyproject.toml/lockfile change, so it affects only
# these two invocations, not the rest of the project's resolved jax
# version. biasing/langevin don't hit this bug (no jax.grad in their hot
# path) and are deliberately left on the project's normal jax version so
# their numbers stay comparable to the already-recorded §9 results.
JAX_GRAD_PATH_PIN=(--with "jax[cuda12]==0.9.2" --with "jaxlib==0.9.2")
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

uv run "${JAX_GRAD_PATH_PIN[@]}" python scripts/ebm/benchmarks/decoy_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/decoy_benchmark_full.json"

uv run "${JAX_GRAD_PATH_PIN[@]}" python scripts/ebm/benchmarks/ddg_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/ddg_benchmark_full.json"

uv run python scripts/ebm/benchmarks/biasing_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/biasing_benchmark_full.json"

uv run python scripts/ebm/benchmarks/langevin_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/langevin_benchmark_full.json"

echo "=== All four benchmarks complete; results in ${OUT_DIR} ==="
