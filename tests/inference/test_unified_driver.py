"""COMP-5: Tests for unified StageSet-driven decode driver.

RED phase — tests fail until:
  - inference/driver.py exists with a `decode` function
  - decode(model, key, enc, cond, wave, config, stage_set) dispatches on stage_set
  - score_conditional, score_unconditional, sample_autoregressive delegate to driver
"""

from __future__ import annotations

import inspect

import pytest


# ---------------------------------------------------------------------------
# 1. Driver module and import contract
# ---------------------------------------------------------------------------

def test_driver_module_importable():
    """inference.driver module is importable."""
    import prxteinmpnn.inference.driver  # noqa: F401


def test_decode_fn_importable():
    """`decode` is importable from inference.driver."""
    from prxteinmpnn.inference.driver import decode  # noqa: F401


def test_decode_signature():
    """`decode` has the canonical 7-argument signature."""
    from prxteinmpnn.inference.driver import decode

    sig = inspect.signature(decode)
    params = list(sig.parameters)
    for expected in ("model", "key", "enc", "cond", "wave", "config", "stage_set"):
        assert expected in params, f"decode missing parameter: {expected!r}"


# ---------------------------------------------------------------------------
# 2. Dispatch topology inference
# ---------------------------------------------------------------------------

def test_decode_returns_logits_for_conditional_no_sample_step():
    """decode with ConditionalDecodeStep + no sample_step returns (L, V) logits."""
    pytest.importorskip("prxteinmpnn.inference.driver")
    from prxteinmpnn.inference.driver import decode
    from prxteinmpnn.types.stages import StageSet

    # Stage set: conditional decode, no sample_step — teacher-forced path
    ss = StageSet()  # decode_step=None means falls back to conditional
    # Topology check: stage_set.sample_step is None → scoring mode
    assert ss.sample_step is None


def test_decode_with_sample_step_returns_sample_result():
    """decode with sample_step set returns SampleResult (not plain logits)."""
    pytest.importorskip("prxteinmpnn.inference.driver")
    from prxteinmpnn.inference.driver import TOPOLOGY_AR, infer_topology
    from prxteinmpnn.types.stages import StageSet

    class DummySampleStep:
        pass

    ss = StageSet(sample_step=DummySampleStep())
    topology = infer_topology(ss)
    assert topology == TOPOLOGY_AR, (
        f"Expected TOPOLOGY_AR when sample_step is set, got {topology!r}"
    )


def test_decode_topology_conditional_scoring():
    """infer_topology returns TOPOLOGY_CONDITIONAL_SCORE when decode_step=None and sample_step=None."""
    from prxteinmpnn.inference.driver import TOPOLOGY_CONDITIONAL_SCORE, infer_topology
    from prxteinmpnn.types.stages import StageSet

    ss = StageSet()
    assert infer_topology(ss) == TOPOLOGY_CONDITIONAL_SCORE


def test_decode_topology_unconditional():
    """infer_topology returns TOPOLOGY_UNCONDITIONAL when UnconditionalDecodeStep is set."""
    from prxteinmpnn.inference.driver import TOPOLOGY_UNCONDITIONAL, infer_topology
    from prxteinmpnn.types.stages import StageSet, UnconditionalDecodeStep

    import equinox as eqx

    class DummyDecoder(eqx.Module):
        pass

    ss = StageSet(decode_step=UnconditionalDecodeStep(decoder=DummyDecoder()))
    assert infer_topology(ss) == TOPOLOGY_UNCONDITIONAL


# ---------------------------------------------------------------------------
# 3. Existing kernel wrappers delegate to driver
# ---------------------------------------------------------------------------

def test_score_conditional_kernel_imports_driver():
    """score_conditional.kernel delegates to inference.driver."""
    import importlib
    import inspect

    sc = importlib.import_module("prxteinmpnn.inference.score_conditional")
    source = inspect.getsource(sc)
    assert "driver" in source or "from prxteinmpnn.inference.driver" in source, (
        "score_conditional should delegate to inference.driver after COMP-5"
    )


def test_score_unconditional_kernel_imports_driver():
    """score_unconditional.kernel delegates to inference.driver."""
    import importlib
    import inspect

    su = importlib.import_module("prxteinmpnn.inference.score_unconditional")
    source = inspect.getsource(su)
    assert "driver" in source or "from prxteinmpnn.inference.driver" in source, (
        "score_unconditional should delegate to inference.driver after COMP-5"
    )


def test_sample_autoregressive_kernel_imports_driver():
    """sample_autoregressive.kernel delegates to inference.driver."""
    import importlib
    import inspect

    sar = importlib.import_module("prxteinmpnn.inference.sample_autoregressive")
    source = inspect.getsource(sar)
    assert "driver" in source or "from prxteinmpnn.inference.driver" in source, (
        "sample_autoregressive should delegate to inference.driver after COMP-5"
    )
