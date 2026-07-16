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
N_REPEATS="${BENCHMARK_N_REPEATS:-10}"
OUT_DIR="${BENCHMARK_OUT_DIR:-outputs/ebm_benchmarks}"

# Batch-size sweep, capped PER LENGTH to avoid a guaranteed OOM crash: the epic already found a
# real memory ceiling (L=512, n_decoys=16 -> ~17GB, RESOURCE_EXHAUSTED above that; see
# .praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md §7 finding 2). The reference's own
# score_decoys.py halves its batch size past a length threshold for the identical reason (its own
# `if decoys['atom37'].shape[1] > 175: bsize = args.bsize // 2`). This is the same discipline: sweep
# the reference's real usage scale (256/400) only where it's plausible to fit, shrink at the
# lengths already confirmed to be memory-constrained, rather than let one crash abort the whole run.
BATCH_SIZES_SMALL_L="${BENCHMARK_BATCH_SIZES_SMALL_L:-4 16 64 256}"    # L in {64, 128}
BATCH_SIZES_MED_L="${BENCHMARK_BATCH_SIZES_MED_L:-4 16 64}"            # L = 256
BATCH_SIZES_LARGE_L="${BENCHMARK_BATCH_SIZES_LARGE_L:-4 16}"           # L = 512 (confirmed ceiling)
LANGEVIN_BATCH_SIZES_SMALL_L="${BENCHMARK_LANGEVIN_BATCH_SMALL_L:-4 16 64 400}"
LANGEVIN_BATCH_SIZES_MED_L="${BENCHMARK_LANGEVIN_BATCH_MED_L:-4 16 64}"
LANGEVIN_BATCH_SIZES_LARGE_L="${BENCHMARK_LANGEVIN_BATCH_LARGE_L:-4 16}"

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

run_per_length_group() {
  local script="$1" out_prefix="$2" small_batches="$3" med_batches="$4" large_batches="$5"
  shift 5
  uv run "${JAX_GRAD_PATH_PIN[@]}" python "scripts/ebm/benchmarks/${script}" \
    --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
    --lengths 64,128 --batch-sizes ${small_batches} --n-repeats "${N_REPEATS}" "$@" \
    --out "${OUT_DIR}/${out_prefix}_L64-128.json"
  uv run "${JAX_GRAD_PATH_PIN[@]}" python "scripts/ebm/benchmarks/${script}" \
    --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
    --lengths 256 --batch-sizes ${med_batches} --n-repeats "${N_REPEATS}" "$@" \
    --out "${OUT_DIR}/${out_prefix}_L256.json"
  uv run "${JAX_GRAD_PATH_PIN[@]}" python "scripts/ebm/benchmarks/${script}" \
    --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
    --lengths 512 --batch-sizes ${large_batches} --n-repeats "${N_REPEATS}" "$@" \
    --out "${OUT_DIR}/${out_prefix}_L512.json"
}

echo "=== decoy_benchmark (grad-path pinned to jax==0.9.2) ==="
run_per_length_group decoy_benchmark.py decoy_benchmark_full \
  "${BATCH_SIZES_SMALL_L}" "${BATCH_SIZES_MED_L}" "${BATCH_SIZES_LARGE_L}"

echo "=== ddg_benchmark (grad-path pinned to jax==0.9.2) ==="
run_per_length_group ddg_benchmark.py ddg_benchmark_full \
  "${BATCH_SIZES_SMALL_L}" "${BATCH_SIZES_MED_L}" "${BATCH_SIZES_LARGE_L}"

echo "=== biasing_benchmark (no batch-size axis; fixed N_STATES=2) ==="
uv run python scripts/ebm/benchmarks/biasing_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/biasing_benchmark_full.json"

echo "=== langevin_benchmark (no grad path; unpinned) ==="
run_per_length_group langevin_benchmark.py langevin_benchmark_full \
  "${LANGEVIN_BATCH_SIZES_SMALL_L}" "${LANGEVIN_BATCH_SIZES_MED_L}" "${LANGEVIN_BATCH_SIZES_LARGE_L}"

echo "=== heterogeneous_batch_benchmark (new; no grad path; unpinned) ==="
uv run python scripts/ebm/benchmarks/heterogeneous_batch_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --n-structures 256 --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/heterogeneous_batch_benchmark_full.json"

echo "=== langevin_annealing_benchmark (new; no grad path; unpinned) ==="
uv run python scripts/ebm/benchmarks/langevin_annealing_benchmark.py \
  --checkpoint "${CHECKPOINT}" --reference-repo "${REFERENCE_REPO}" \
  --lengths "${LENGTHS}" --n-repeats "${N_REPEATS}" \
  --out "${OUT_DIR}/langevin_annealing_benchmark_full.json"

echo "=== All benchmarks complete; results in ${OUT_DIR} ==="
