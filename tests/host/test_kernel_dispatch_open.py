"""COMP-7: Tests for open kernel_dispatch — accepts resolved DecodeFn from spec.

RED phase — tests fail until:
  - kernel_dispatch.py resolves a DecodeFn from spec.sampling_strategy
  - The hardwired sample_ar.kernel call at line 132 is replaced with resolved_fn
  - resolve_kernel_fn(strategy) -> callable exists in host/kernel_dispatch.py
"""

from __future__ import annotations

import inspect


# ---------------------------------------------------------------------------
# 1. resolve_kernel_fn exists and is callable
# ---------------------------------------------------------------------------

def test_resolve_kernel_fn_importable():
    """resolve_kernel_fn is importable from host.kernel_dispatch."""
    from prxteinmpnn.host.kernel_dispatch import resolve_kernel_fn  # noqa: F401


def test_resolve_kernel_fn_temperature_returns_callable():
    """resolve_kernel_fn('temperature') returns a callable."""
    from prxteinmpnn.host.kernel_dispatch import resolve_kernel_fn

    fn = resolve_kernel_fn("temperature")
    assert callable(fn), "resolve_kernel_fn('temperature') must return a callable"


def test_resolve_kernel_fn_straight_through_returns_callable():
    """resolve_kernel_fn('straight_through') returns a callable."""
    from prxteinmpnn.host.kernel_dispatch import resolve_kernel_fn

    fn = resolve_kernel_fn("straight_through")
    assert callable(fn), "resolve_kernel_fn('straight_through') must return a callable"


def test_resolve_kernel_fn_temperature_differs_from_straight_through():
    """'temperature' and 'straight_through' resolve to different kernel fns."""
    from prxteinmpnn.host.kernel_dispatch import resolve_kernel_fn

    fn_temp = resolve_kernel_fn("temperature")
    fn_ste = resolve_kernel_fn("straight_through")
    assert fn_temp is not fn_ste, "temperature and straight_through must resolve to distinct fns"




# ---------------------------------------------------------------------------
# 3. spec.sampling_strategy drives the resolved fn
# ---------------------------------------------------------------------------

def test_spec_sampling_strategy_temperature_maps_to_ar():
    """sampling_strategy='temperature' maps to the AR (autoregressive) path."""
    from prxteinmpnn.host.kernel_dispatch import resolve_kernel_fn
    from prxteinmpnn.inference import sample_autoregressive as sample_ar

    fn = resolve_kernel_fn("temperature")
    assert fn is sample_ar.kernel, (
        "resolve_kernel_fn('temperature') must resolve to sample_ar.kernel"
    )


def test_resolve_kernel_fn_unknown_strategy_returns_callable():
    """resolve_kernel_fn with unknown strategy returns a callable (default fallback, no raise)."""
    from prxteinmpnn.host.kernel_dispatch import resolve_kernel_fn

    fn = resolve_kernel_fn("totally_unknown_strategy")
    assert callable(fn)
