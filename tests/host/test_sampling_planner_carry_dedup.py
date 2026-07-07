"""Test that make_sampling_planner correctly passes carry_specs and dedup_specs."""

import numpy as np
import pytest

from aminx.host.plan import make_sampling_planner
from aminx.run.specs import SamplingSpecification
from xtrax.tiling import CarrySpec
from xtrax.tiling.dedup import DedupSpec


def dummy_transition(carry, x):
    """Simple carry transition for testing."""
    return carry, x


def test_sampling_specification_has_carry_specs():
    """Test that SamplingSpecification has carry_specs field."""
    carry_spec = CarrySpec(
        axis_name="n_samples",
        init=0.0,
        transition=dummy_transition,
    )
    spec = SamplingSpecification(
        inputs=[],
        num_samples=4,
        carry_specs=[carry_spec],
    )
    assert spec.carry_specs == [carry_spec]


def test_sampling_specification_has_dedup_specs():
    """Test that SamplingSpecification has dedup_specs field."""
    dedup_spec = DedupSpec(
        axis_name="n_structures",
        unique_indices=np.array([0, 1], dtype=np.int32),
        index_map=np.array([0, 1, 0, 1], dtype=np.int32),
        k=2,
    )
    spec = SamplingSpecification(
        inputs=[],
        dedup_specs=[dedup_spec],
    )
    assert spec.dedup_specs == [dedup_spec]


def test_make_sampling_planner_with_empty_carry():
    """Test that make_sampling_planner works with empty carry_specs."""
    spec = SamplingSpecification(
        inputs=[],
        num_samples=4,
        carry_specs=[],
    )
    plan = make_sampling_planner(spec)
    assert plan is not None


def test_make_sampling_planner_with_empty_dedup():
    """Test that make_sampling_planner works with empty dedup_specs."""
    spec = SamplingSpecification(
        inputs=[],
        dedup_specs=[],
    )
    plan = make_sampling_planner(spec)
    assert plan is not None


def test_make_sampling_planner_with_both_empty():
    """Test that make_sampling_planner works with empty carry_specs and dedup_specs."""
    spec = SamplingSpecification(
        inputs=[],
        num_samples=4,
        carry_specs=[],
        dedup_specs=[],
    )
    plan = make_sampling_planner(spec)
    assert plan is not None
