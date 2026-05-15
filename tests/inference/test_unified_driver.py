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

def test_stage_set_default_is_scoring_mode():
    """StageSet default configuration indicates scoring mode (no sample_step)."""
    from prxteinmpnn.types.stages import StageSet

    # Default: no sample_step, no decode_step — teacher-forced scoring path
    ss = StageSet()
    # Topology check: stage_set.sample_step is None → scoring mode
    assert ss.sample_step is None


def test_infer_topology_with_sample_step_returns_ar():
    """infer_topology returns TOPOLOGY_AR when sample_step is set."""
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


def test_infer_topology_with_conditional_decode_step():
    """infer_topology returns TOPOLOGY_CONDITIONAL_SCORE when ConditionalDecodeStep is set (not Unconditional)."""
    import equinox as eqx
    from prxteinmpnn.inference.driver import TOPOLOGY_CONDITIONAL_SCORE, infer_topology
    from prxteinmpnn.types.stages import ConditionalDecodeStep, StageSet

    class DummyDecoder(eqx.Module):
        pass

    class DummyEmbed(eqx.Module):
        pass

    ss = StageSet(decode_step=ConditionalDecodeStep(decoder=DummyDecoder(), w_s_embed=DummyEmbed()))
    assert infer_topology(ss) == TOPOLOGY_CONDITIONAL_SCORE


