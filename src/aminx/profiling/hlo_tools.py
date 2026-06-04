"""HLO export and static memory comparison helpers.

Vendored from jaxbeans ``core/profiling.py`` for Phase 0 CI (no jaxbeans package
required on isolated checkouts). Upstream path:
``jaxbeans/src/jaxbeans/core/profiling.py`` (jaxbeans 0.1.0 tree).

Upstream sync: unknown (path-only vendored excerpt).

Behavior matches upstream; only docstrings/headers differ.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import jax
from jax import profiler


def trace_kernel(name: str):
  """Decorator to add a named trace event for XLA profiling."""

  def decorator(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
      with profiler.TraceAnnotation(name):
        return fn(*args, **kwargs)

    return wrapper

  return decorator


def start_trace(path: str = "/tmp/jax_trace") -> None:
  """Start JAX profiler trace (useful for capturing chrome://tracing files)."""
  profiler.start_trace(path)


def stop_trace() -> None:
  """Stop JAX profiler trace."""
  profiler.stop_trace()


def export_hlo(func: Callable, *args: Any, **kwargs: Any) -> str:
  """Lower a JAX function and export its HLO as a string."""
  lowered = jax.jit(func).lower(*args, **kwargs)
  if lowered is None:
    raise ValueError("Failed to lower function to HLO.")
  compiler_ir = lowered.compiler_ir("hlo")
  if compiler_ir is None:
    raise ValueError("Failed to get compiler IR from lowered function.")
  return compiler_ir.as_hlo_text()


def analyze_memory(func: Callable, *args: Any, **kwargs: Any) -> str:
  """Perform static memory analysis on a lowered JAX kernel."""
  lowered = jax.jit(func).lower(*args, **kwargs)
  compiled = lowered.compile()
  analysis = compiled.memory_analysis()
  return str(analysis)


def assert_zero_copy_overhead(
  func_legacy: Callable,
  func_adapter: Callable,
  *args: Any,
  **kwargs: Any,
) -> None:
  """Verify adapter-based kernel has identical memory analysis to legacy."""
  mem_legacy = analyze_memory(func_legacy, *args, **kwargs)
  mem_adapter = analyze_memory(func_adapter, *args, **kwargs)

  if mem_legacy != mem_adapter:
    raise AssertionError(
      "Memory analysis mismatch between legacy and adapter kernels.\n"
      f"Legacy Memory:\n{mem_legacy}\n"
      f"Adapter Memory:\n{mem_adapter}",
    )
