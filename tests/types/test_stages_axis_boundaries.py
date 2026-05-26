# tests/types/test_stages_axis_boundaries.py
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from prxteinmpnn.types.stages import StageSet, EncoderSinkFn
from prxteinmpnn.types.boundaries import AxisBoundary


def test_stage_set_encoder_sink_defaults_to_empty_tuple():
    ss = StageSet()
    assert ss.encoder_sink == ()
    assert isinstance(ss.encoder_sink, tuple)


def test_stage_set_encoder_sink_accepts_tuple_of_sinks():
    class MySink:
        ordered = True
        def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
            pass

    ss = StageSet(encoder_sink=(MySink(),))
    assert len(ss.encoder_sink) == 1
    assert isinstance(ss.encoder_sink[0], MySink)


def test_stage_set_encoder_sink_accepts_multiple_sinks():
    class SinkA:
        ordered = False
        def __call__(self, enc, *args) -> None: pass

    class SinkB:
        ordered = False
        def __call__(self, enc, *args) -> None: pass

    ss = StageSet(encoder_sink=(SinkA(), SinkB()))
    assert len(ss.encoder_sink) == 2


def test_stage_set_axis_boundaries_defaults_to_empty_dict():
    ss = StageSet()
    assert ss.axis_boundaries == {}
    assert isinstance(ss.axis_boundaries, dict)


def test_stage_set_axis_boundaries_accepts_boundary_per_axis():
    b = AxisBoundary()
    ss = StageSet(axis_boundaries={"n_noises": b})
    assert "n_noises" in ss.axis_boundaries
    assert ss.axis_boundaries["n_noises"] is b


def test_stage_set_axis_boundaries_is_static():
    """axis_boundaries must be a static field — no additional JAX leaves."""
    ss = StageSet(axis_boundaries={"n_noises": AxisBoundary()})
    leaves = jax.tree_util.tree_leaves(ss)
    ss_no_boundaries = StageSet()
    leaves_base = jax.tree_util.tree_leaves(ss_no_boundaries)
    # axis_boundaries adds no traced leaves (it's static)
    assert len(leaves) == len(leaves_base)


def test_stage_set_encoder_sink_iteration_pattern():
    """Kernel dispatch iterates tuple; empty tuple is no-op without branching."""
    calls = []

    class TrackingSink:
        ordered = False
        def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
            calls.append(1)

    ss = StageSet(encoder_sink=(TrackingSink(), TrackingSink()))

    # Simulate kernel pattern: for sink in stage_set.encoder_sink: sink(...)
    for sink in ss.encoder_sink:
        sink(None, 0, 0, 0)

    assert len(calls) == 2


def test_stage_set_empty_encoder_sink_no_iteration():
    ss = StageSet()  # encoder_sink = ()
    calls = []
    for sink in ss.encoder_sink:
        calls.append(1)
    assert calls == []  # no-op
