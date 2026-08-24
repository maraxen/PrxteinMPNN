#!/bin/bash
# T2.GATE throughput/recompile-parity leg, swept across GPU architectures.
#
# EPIC #1541's R1 DoD calls the cluster-throughput gate a STANDING gate ("not a
# one-shot at flip"), and it had not been re-run since 2026-07-06 against xtrax
# 0.4.0a1 -- two prereleases before the current 0.4.0a5 pin. See
# .praxia/docs/audits/260817_aminx-deeper-work-assessment.md §3.2.
#
# Why sweep architectures rather than run once: the gate asserts identical JIT
# recompile counts and a throughput ratio, both of which are properties of the
# XLA backend, not of aminx. A parity result on one SM does not transfer to
# another.
#
# Why 3 repeats per GPU (the array): the CPU sibling of this bench reduces across
# cases with max() over microsecond-scale timings, which made its verdict
# irreproducible -- five observations spanned 1.01-1.64 and straddled all three of
# its outcome bands. This bench is better built (np.mean over n_timed=50 per case)
# and the L3 smoke on L40S returned a clean 1.04, but "the timings resolve well
# enough on GPU" is a claim to demonstrate, not assume. Repeats are what let the
# spread be reported instead of a single point.
#
# Usage -- one submission per GPU architecture, GATE_GPU is a label only
# (the actual pinning is --gres):
#
#   sbatch --partition=mit_normal_gpu --gres=gpu:l40s:1 \
#          --export=ALL,GATE_GPU=l40s scripts/slurm/t2gate_gpu_sweep.slurm
#
# Results land in outputs/results/t2gate_<gpu>_rep<N>.json, to be pulled locally
# and registered with `bth run --replay` (the affordance the bench itself
# provides for cluster-computed results) rather than tracked cluster-side, which
# would put them in a second catalog.

#SBATCH --job-name=t2gate-gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/logs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/logs/slurm/%x_%A_%a.err

set -euo pipefail

# Absolute project path, NOT $SLURM_SUBMIT_DIR. Measured on job 20640802:
# `myxcel submit-job` leaves SLURM_SUBMIT_DIR as the HOME directory
# (/orcd/home/002/maarxaru), not the project, so a job that cd's there fails to
# find its own script. Direct sbatch sets it correctly, but this script must work
# under both launch paths, so it does not depend on it at all.
PROJ="${HOME}/projects/aminx"
cd "${PROJ}"

export PATH="${HOME}/.local/bin:${PATH}"

# uv cache AND venv both live on scratch. /home and both ORCD pools reject writes
# with EDQUOT despite df reporting free space and 1% inode use (~/.cache is a
# symlink into the full pool/007), so scratch is the only writable location with
# room. Note UV_PROJECT_ENVIRONMENT is required at RUN time too, not just at sync
# time -- the venv is outside the project tree, and without it `uv run` would try
# to build a fresh one on a compute node that has no outbound internet.
export UV_CACHE_DIR=/orcd/scratch/orcd/011/maarxaru/.cache/uv
export UV_PROJECT_ENVIRONMENT=/orcd/scratch/orcd/011/maarxaru/venvs/aminx

GPU_LABEL="${GATE_GPU:-unknown}"
REP="${SLURM_ARRAY_TASK_ID:-0}"
OUT="outputs/results/t2gate_${GPU_LABEL}_rep${REP}.json"
mkdir -p outputs/results outputs/logs/slurm

# Blackwell (SM120) workaround, mandatory on pi_so3's node4007/node4008: XLA
# autotuning hangs there even on recent jaxlib (1170x slowdown measured, SLURM
# 15294138). Keyed on hostname rather than partition, since pi_so3 also contains
# node4009 (H200) and may gain other non-Blackwell nodes. No-op elsewhere.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
  export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
  echo "[gate] Blackwell node ${_NODE}: XLA_FLAGS=${XLA_FLAGS}"
fi

echo "[gate] label=${GPU_LABEL} rep=${REP} node=${_NODE}"
echo "[gate] cwd=$(pwd)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Fail loudly rather than silently benchmarking the CPU: a CPU fallback would
# still produce a plausible-looking ratio and quietly corrupt the sweep.
uv run --no-sync python -c "
import sys, jax
print('[gate] python:', sys.executable)
d = jax.devices()
print('[gate] devices:', d)
assert d and d[0].platform == 'gpu', f'expected a GPU device, got {d}'
"

uv run --no-sync python scripts/benchmarks/bench_xtrax_vs_aminx_dispatch_gpu.py \
  --n-warmup 10 --n-timed 50 --out "${OUT}"

echo "[gate] wrote ${OUT}"
