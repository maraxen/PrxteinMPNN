"""Tests for duck-type compatibility with ConformationalStates protocol.

This module verifies that any object with an `n_states: int` attribute
is accepted wherever RunSpecification expects conformational states.
"""

from __future__ import annotations

import pytest

from prxteinmpnn.run.specs import RunSpecification


class SimpleConformationalState:
    """Minimal duck-type implementation of ConformationalStates protocol."""

    def __init__(self, n_states: int):
        self.n_states = n_states


class ConformationalStateWithoutNStates:
    """Object missing the required n_states attribute."""

    def __init__(self):
        self.other_attr = 42


def test_conformational_states_protocol_basic():
    """Verify ConformationalStates protocol requires n_states attribute."""
    # Create a simple object with n_states
    state = SimpleConformationalState(n_states=5)

    # Should be able to use it in RunSpecification
    spec = RunSpecification(
        inputs="test.pdb",
        conformational_states=state,
    )

    # Access the conformational_states and verify n_states is accessible
    assert spec.conformational_states is not None
    assert spec.conformational_states.n_states == 5


def test_conformational_states_protocol_int_type():
    """Verify n_states is typed as int and accepted as such."""
    # Create object with n_states as int
    state = SimpleConformationalState(n_states=10)

    spec = RunSpecification(
        inputs="test.pdb",
        conformational_states=state,
    )

    # Verify we can use n_states as an integer
    n = spec.conformational_states.n_states
    assert isinstance(n, int)
    assert n == 10

    # Verify it behaves like an int in arithmetic
    result = n * 2
    assert result == 20


def test_conformational_states_protocol_missing_attribute():
    """Verify that objects missing n_states fail at runtime when accessed."""
    # Create object without n_states
    state = ConformationalStateWithoutNStates()

    spec = RunSpecification(
        inputs="test.pdb",
        conformational_states=state,
    )

    # RunSpecification should accept it (no static validation)
    assert spec.conformational_states is not None

    # But accessing n_states should raise AttributeError
    with pytest.raises(AttributeError, match="n_states"):
        _ = spec.conformational_states.n_states


def test_conformational_states_protocol_none():
    """Verify that conformational_states can be None."""
    spec = RunSpecification(
        inputs="test.pdb",
        conformational_states=None,
    )

    assert spec.conformational_states is None


def test_conformational_states_protocol_multiple_states():
    """Verify protocol works with varying n_states values."""
    for num_states in [1, 3, 5, 10, 100]:
        state = SimpleConformationalState(n_states=num_states)

        spec = RunSpecification(
            inputs="test.pdb",
            conformational_states=state,
        )

        assert spec.conformational_states.n_states == num_states


def test_conformational_states_protocol_structural_subtyping():
    """Verify structural subtyping: any object with n_states satisfies protocol."""

    class CustomEnsemble:
        """Custom object that conforms to ConformationalStates structurally."""

        def __init__(self, num_conformations: int):
            self.n_states = num_conformations
            self.metadata = "custom ensemble"

    ensemble = CustomEnsemble(num_conformations=7)
    spec = RunSpecification(
        inputs="test.pdb",
        conformational_states=ensemble,
    )

    assert spec.conformational_states.n_states == 7
    assert spec.conformational_states.metadata == "custom ensemble"
