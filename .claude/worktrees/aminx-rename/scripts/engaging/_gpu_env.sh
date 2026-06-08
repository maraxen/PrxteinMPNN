#!/bin/bash
# GPU environment helper: detect GPU architecture and set JAX_EXTRA + XLA_FLAGS.
#
# USAGE: Source this script AFTER cd-ing to the project root.
# This script is meant to be SOURCED (not executed), so it can export variables
# into the calling shell environment.
#
# SBATCH Scripts: Place immediately after `cd <project>` (before `uv sync`):
#     source scripts/engaging/_gpu_env.sh
#     uv sync --extra "${JAX_EXTRA}" --group benchmark --group dev
#
# The helper detects the GPU name and compute capability via nvidia-smi
# and sets JAX_EXTRA (cuda or cuda12) and XLA_FLAGS (Blackwell workaround).
# If nvidia-smi is absent, JAX_EXTRA defaults to "cuda" with a warning.

set -euo pipefail

# Default to cuda unless we detect otherwise
export JAX_EXTRA="${JAX_EXTRA:-cuda}"
_GPU_NAME=""
_COMPUTE_CAP=""
_BLACKWELL_FLAG_SET=0

# Try to detect GPU via nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    # Query GPU name and compute capability
    _QUERY_OUTPUT=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || echo "")

    if [[ -n "${_QUERY_OUTPUT}" ]]; then
        # Extract first GPU (line 1)
        _GPU_LINE=$(echo "${_QUERY_OUTPUT}" | head -1)
        _GPU_NAME=$(echo "${_GPU_LINE}" | cut -d',' -f1 | xargs)  # name, trim spaces
        _COMPUTE_CAP=$(echo "${_GPU_LINE}" | cut -d',' -f2 | xargs)  # compute_cap, trim spaces

        # Determine JAX_EXTRA based on compute capability
        # H200 = 9.0, A100 = 8.0, L40S = 8.9, Blackwell (SM120) = 12.0 or 10.0
        case "${_COMPUTE_CAP}" in
            12.0|10.0)
                # Blackwell (SM120 or newer): apply XLA autotuning workaround and default to cuda
                export XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_shard_autotuning=false"
                _BLACKWELL_FLAG_SET=1
                export JAX_EXTRA="cuda"
                ;;
            8.9)
                # L40S (compute_cap 8.9): currently resolves to cuda12 plugin via default cuda extra
                # Default to cuda; flip to cuda12 here ONLY if the L40S run's backend assert
                # reports CPU fallback.
                export JAX_EXTRA="cuda"
                ;;
            9.0)
                # H200 (compute_cap 9.0)
                export JAX_EXTRA="cuda"
                ;;
            8.0)
                # A100 (compute_cap 8.0)
                export JAX_EXTRA="cuda"
                ;;
            *)
                # Unknown compute capability: default to cuda
                export JAX_EXTRA="cuda"
                ;;
        esac
    else
        echo "Warning: nvidia-smi query failed; defaulting JAX_EXTRA=cuda"
    fi
else
    echo "Warning: nvidia-smi not found; defaulting JAX_EXTRA=cuda"
fi

# Summary output
if [[ -n "${_GPU_NAME}" && -n "${_COMPUTE_CAP}" ]]; then
    _BLACKWELL_MSG=""
    if [[ ${_BLACKWELL_FLAG_SET} -eq 1 ]]; then
        _BLACKWELL_MSG=" [Blackwell XLA autotuning workaround active]"
    fi
    echo "GPU Environment: ${_GPU_NAME} (compute_cap=${_COMPUTE_CAP}) → JAX_EXTRA=${JAX_EXTRA}${_BLACKWELL_MSG}"
else
    echo "GPU Environment: nvidia-smi unavailable; using JAX_EXTRA=${JAX_EXTRA}"
fi
