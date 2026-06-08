#!/bin/bash
# Add nvidia.cudnn lib dir to LD_LIBRARY_PATH if available.
# Source this script immediately after 'uv sync' to ensure GPU frameworks
# (torch, jax) can find libcudnn.so.9.

_CUDNN_LIB="$(uv run python -c 'import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "${_CUDNN_LIB}" && -d "${_CUDNN_LIB}" ]]; then
    export LD_LIBRARY_PATH="${_CUDNN_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    echo "cuDNN on LD_LIBRARY_PATH: ${_CUDNN_LIB}"
else
    echo "WARNING: could not locate nvidia.cudnn lib dir; cuDNN may be unavailable"
fi
