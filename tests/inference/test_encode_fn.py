"""COMP-6: Tests for make_encode_fn in inference/encode.py.

RED phase — tests fail until:
  - inference/encode.py exists and exports make_encode_fn
  - make_encode_fn(model, use_rolling_state=False) -> EncodeFn
  - EncodeFn returns EncoderOutput
  - use_rolling_state flag controls scan vs vmap over S
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Module and import contract
# ---------------------------------------------------------------------------

def test_encode_module_importable():
    """inference.encode module is importable."""
    import prxteinmpnn.inference.encode  # noqa: F401


def test_make_encode_fn_importable():
    """make_encode_fn is importable from inference.encode."""
    from prxteinmpnn.inference.encode import make_encode_fn  # noqa: F401


def test_make_encode_fn_has_use_rolling_state_param():
    """make_encode_fn accepts a use_rolling_state keyword argument."""
    import inspect
    from prxteinmpnn.inference.encode import make_encode_fn

    sig = inspect.signature(make_encode_fn)
    assert "use_rolling_state" in sig.parameters


# ---------------------------------------------------------------------------
# 2. Returned callable contract
# ---------------------------------------------------------------------------

def test_make_encode_fn_returns_callable():
    """make_encode_fn(model) returns a callable."""
    from prxteinmpnn.inference.encode import make_encode_fn

    # A minimal duck-type model
    import equinox as eqx

    class DummyModel(eqx.Module):
        pass

    fn = make_encode_fn(DummyModel(), use_rolling_state=False)
    assert callable(fn)


# ---------------------------------------------------------------------------
# 3. use_rolling_state flag is honoured at construction time
# ---------------------------------------------------------------------------

def test_make_encode_fn_records_rolling_state_flag():
    """make_encode_fn stores use_rolling_state so it can be inspected."""
    from prxteinmpnn.inference.encode import make_encode_fn
    import equinox as eqx

    class DummyModel(eqx.Module):
        pass

    fn_vmap = make_encode_fn(DummyModel(), use_rolling_state=False)
    fn_scan = make_encode_fn(DummyModel(), use_rolling_state=True)

    # They must be distinct objects (not the same closure sharing state)
    assert fn_vmap is not fn_scan


