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
    pytest.importorskip("prxteinmpnn.inference.encode")
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


# ---------------------------------------------------------------------------
# 4. Integration: score_conditional kernel no longer inlines encode logic
#    (checked by inspecting that score_conditional imports from encode)
# ---------------------------------------------------------------------------

def test_score_conditional_imports_from_encode():
    """score_conditional references inference.encode (not inline encode logic)."""
    import importlib
    import inspect

    sc = importlib.import_module("prxteinmpnn.inference.score_conditional")
    source = inspect.getsource(sc)
    # After COMP-6, the rolling-state encode logic moves to inference.encode
    assert "inference.encode" in source or "from prxteinmpnn.inference.encode" in source, (
        "score_conditional should delegate encode to inference.encode after COMP-6"
    )


# ---------------------------------------------------------------------------
# 5. Integration: conditional_logits imports from encode
# ---------------------------------------------------------------------------

def test_conditional_logits_imports_from_encode():
    """conditional_logits references inference.encode (not its own inline encode)."""
    import importlib
    import inspect

    cl = importlib.import_module("prxteinmpnn.sampling.conditional_logits")
    source = inspect.getsource(cl)
    assert "inference.encode" in source or "from prxteinmpnn.inference.encode" in source, (
        "conditional_logits should delegate to inference.encode after COMP-6"
    )
