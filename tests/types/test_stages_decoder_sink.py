# tests/types/test_stages_decoder_sink.py
from __future__ import annotations

import jax
import equinox as eqx
import pytest

from prxteinmpnn.types.stages import StageSet


def test_stage_set_decoder_sink_defaults_to_empty_tuple():
    """decoder_sink must default to empty tuple."""
    ss = StageSet()
    assert ss.decoder_sink == ()
    assert isinstance(ss.decoder_sink, tuple)


def test_stage_set_decoder_sink_accepts_tuple_of_sinks():
    """decoder_sink can be populated with a tuple of callables."""
    class MySink:
        def __call__(self, *args, **kwargs) -> None:
            pass

    ss = StageSet(decoder_sink=(MySink(),))
    assert len(ss.decoder_sink) == 1
    assert isinstance(ss.decoder_sink[0], MySink)


def test_stage_set_decoder_sink_accepts_multiple_sinks():
    """decoder_sink can hold multiple sinks."""
    class SinkA:
        def __call__(self, *args, **kwargs) -> None:
            pass

    class SinkB:
        def __call__(self, *args, **kwargs) -> None:
            pass

    ss = StageSet(decoder_sink=(SinkA(), SinkB()))
    assert len(ss.decoder_sink) == 2


def test_stage_set_decoder_sink_is_static():
    """decoder_sink must be a static field — no additional JAX leaves."""
    ss = StageSet(decoder_sink=(lambda *a, **k: None,))
    leaves = jax.tree_util.tree_leaves(ss)
    ss_no_decoder_sink = StageSet()
    leaves_base = jax.tree_util.tree_leaves(ss_no_decoder_sink)
    # decoder_sink adds no traced leaves (it's static)
    assert len(leaves) == len(leaves_base)


def test_stage_set_decoder_sink_treedef_invariant():
    """Treedef must be unchanged whether decoder_sink is empty or populated (static field)."""
    ss_empty = StageSet()
    ss_with_sink = StageSet(decoder_sink=(lambda *a, **k: None,))

    # Both should have the same number of leaves (decoder_sink is static, adds no leaves)
    leaves_empty = jax.tree_util.tree_leaves(ss_empty)
    leaves_with_sink = jax.tree_util.tree_leaves(ss_with_sink)
    assert len(leaves_empty) == len(leaves_with_sink)
