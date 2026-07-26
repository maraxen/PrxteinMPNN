#!/bin/bash
#SBATCH --job-name=ebm-accuracy-runs
#SBATCH --partition=pi_so3
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/maarxaru/projects/aminx/outputs/logs/slurm/%j.out
#SBATCH --error=/home/maarxaru/projects/aminx/outputs/logs/slurm/%j.err

# Real accuracy runs for ProteinEBM parity vs the design-spec paper targets:
#   * decoy ranking  -> Spearman(E, TM-score) at pinned t=0.05, paper target 0.838
#   * ddG stability  -> Spearman(ddG, DMS_score),              paper target 0.686
#
# These are ACCURACY (correlation-vs-paper) runs, distinct from the throughput
# benchmarks under scripts/ebm/benchmarks/. Both scripts checkpoint per-item to a
# sibling .partial.jsonl, so a walltime timeout preserves completed items.
#
# Env knobs (all optional; smoke runs set the caps + a distinct OUT_TAG):
#   ACC_OUT_TAG        output filename tag              (default: full)
#   ACC_MAX_NATIVES    cap natives for decoy ranking    (default: all 133)
#   ACC_MAX_ASSAYS     cap Tsuboyama assays for ddG      (default: all 64)
#   ACC_MAX_DECOYS     cap decoys/native                 (default: all ~957)
#   ACC_MAX_MUTANTS    cap mutants/assay                 (default: all)
#   ACC_SKIP_DECOY / ACC_SKIP_DDG   set to 1 to skip that benchmark
#
# The existing .venv is used WITHOUT `uv sync` (--no-sync): orbax/pandas are
# installed in it but are not declared pyproject deps, so a sync would prune them.

set -uo pipefail
cd /home/maarxaru/projects/aminx
export PYTHONUNBUFFERED=1  # live per-item logging under SLURM redirection (else block-buffered)

ASSETS="${HOME}/proteinebm_bench_assets/ProteinEBM/eval_data"
ORBAX_MODEL="${ACC_ORBAX_MODEL:-/orcd/scratch/orcd/011/maarxaru/proteinebm_accuracy/orbax_model}"
OUT_TAG="${ACC_OUT_TAG:-full}"
OUT_DIR="outputs/ebm_benchmarks/real_accuracy"
mkdir -p "${OUT_DIR}"

# Blackwell (SM120, node4007/4008) XLA autotuning workaround -- no-op elsewhere.
_NODE="${SLURM_JOB_NODELIST:-$(hostname -s)}"
if [[ "${_NODE}" == *node4007* ]] || [[ "${_NODE}" == *node4008* ]]; then
    export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
fi

# Persistent XLA compilation cache on scratch: the 16-layer transformer's first compile is slow
# (~15-20 min cold and can blow a short walltime), but only needs to happen ONCE -- cache it so
# every native/assay after the first, and every re-run on any node, reuses it.
export JAX_COMPILATION_CACHE_DIR="${ACC_JAX_CACHE:-/orcd/scratch/orcd/011/maarxaru/jax_compilation_cache}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

# cuDNN on LD_LIBRARY_PATH from the existing venv (no sync).
_CUDNN_LIB="$(uv run --no-sync python -c 'import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "${_CUDNN_LIB}" && -d "${_CUDNN_LIB}" ]]; then
    export LD_LIBRARY_PATH="${_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "=== ProteinEBM accuracy runs (tag=${OUT_TAG}) ==="
echo "Node:        $(hostname -s)"
echo "GPU:         $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
echo "Orbax model: ${ORBAX_MODEL}"
echo "Assets:      ${ASSETS}"
echo "XLA_FLAGS:   ${XLA_FLAGS:-<unset>}"
echo "JAX cache:   ${JAX_COMPILATION_CACHE_DIR}"

# Bathos tracking: full-scale (unset ACC_MAX_*) runs are the claim's confirmatory
# evidence (campaign ${ACC_BTH_CAMPAIGN}) and get wrapped in `bth run` for real
# provenance; capped/smoke runs stay untracked (bth run's sidecar enforcement is
# meant for the confirmatory runs the claim actually references).
BTH_CAMPAIGN="${ACC_BTH_CAMPAIGN:-aba3bfe6}"
USE_BTH_RUN=0
if [[ -z "${ACC_MAX_NATIVES:-}${ACC_MAX_ASSAYS:-}${ACC_MAX_DECOYS:-}${ACC_MAX_MUTANTS:-}" ]]; then
    USE_BTH_RUN=1
fi
# NOTE: `bth run`'s own flags (--campaign/--tag/--out) MUST precede the `--`
# separator -- anything after `--` is the literal argv `bth run` execs. An
# earlier version put --out AFTER a pre-built RUN_PREFIX ending in `--`, so
# `bth run` tried to exec a binary literally named "--out" and failed instantly
# with FileNotFoundError (confirmed via ~/.bth/catalog/logs/events.node4008.*
# from job 18885835's run.error records).

# ---- decoy ranking (paper target 0.838) ----------------------------------
if [[ "${ACC_SKIP_DECOY:-0}" != "1" ]]; then
    echo "--- decoy ranking ---"
    DECOY_OUT="${OUT_DIR}/decoy_ranking_${OUT_TAG}.json"
    DECOY_ARGS=(
        --decoys-dir "${ASSETS}/decoys"
        --tmscore    "${ASSETS}/decoy_data/tmscore.txt"
        --orbax-model "${ORBAX_MODEL}"
        --out "${DECOY_OUT}"
    )
    [[ -n "${ACC_MAX_NATIVES:-}" ]] && DECOY_ARGS+=(--max-natives "${ACC_MAX_NATIVES}")
    [[ -n "${ACC_MAX_DECOYS:-}"  ]] && DECOY_ARGS+=(--max-decoys-per-native "${ACC_MAX_DECOYS}")
    if [[ "${USE_BTH_RUN}" == "1" ]]; then
        bth run --campaign "${BTH_CAMPAIGN}" --tag "epic:proteinebm" --tag "ebm:e5-e6-accuracy" --tag "report:260716-parity" --out "${DECOY_OUT}" -- \
            uv run --no-sync python scripts/ebm/real_decoy_ranking_benchmark.py "${DECOY_ARGS[@]}"
    else
        uv run --no-sync python scripts/ebm/real_decoy_ranking_benchmark.py "${DECOY_ARGS[@]}"
    fi
    echo "=== decoy ranking exit: $? ==="
fi

# ---- ddG stability (paper target 0.686) ----------------------------------
if [[ "${ACC_SKIP_DDG:-0}" != "1" ]]; then
    echo "--- ddG stability ---"
    DDG_OUT="${OUT_DIR}/ddg_stability_${OUT_TAG}.json"
    DDG_ARGS=(
        --proteingym-dir "${ASSETS}/proteingym"
        --orbax-model "${ORBAX_MODEL}"
        --out "${DDG_OUT}"
    )
    [[ -n "${ACC_MAX_ASSAYS:-}"  ]] && DDG_ARGS+=(--max-assays "${ACC_MAX_ASSAYS}")
    [[ -n "${ACC_MAX_MUTANTS:-}" ]] && DDG_ARGS+=(--max-mutants-per-assay "${ACC_MAX_MUTANTS}")
    if [[ "${USE_BTH_RUN}" == "1" ]]; then
        bth run --campaign "${BTH_CAMPAIGN}" --tag "epic:proteinebm" --tag "ebm:e5-e6-accuracy" --tag "report:260716-parity" --out "${DDG_OUT}" -- \
            uv run --no-sync python scripts/ebm/real_ddg_stability_benchmark.py "${DDG_ARGS[@]}"
    else
        uv run --no-sync python scripts/ebm/real_ddg_stability_benchmark.py "${DDG_ARGS[@]}"
    fi
    echo "=== ddG stability exit: $? ==="
fi

echo "=== all accuracy runs done; outputs in ${OUT_DIR}/ ==="
ls -la "${OUT_DIR}/" 2>&1
