"""Tests for data structure definitions."""

import chex
import jax.numpy as jnp
import pytest
from dataclasses import FrozenInstanceError
from prxteinmpnn.utils.data_structures import ModelInputs, ProteinStructure


def test_protein_structure_frozen():
    """Test that ProteinStructure dataclass is immutable.

    Raises:
        FrozenInstanceError: If the dataclass is mutable.
    """
    p = ProteinStructure(
        coordinates=jnp.zeros((1, 1, 3)),
        aatype=jnp.zeros((1,)),
        atom_mask=jnp.zeros((1, 1)),
        residue_index=jnp.zeros((1,)),
        chain_index=jnp.zeros((1,)),
    )
    with pytest.raises(FrozenInstanceError):
        p.aatype = jnp.ones((1,))  # type: ignore[assignment]


def test_model_inputs_frozen_and_defaults():
    """Test that ModelInputs is immutable and has correct defaults.

    Raises:
        FrozenInstanceError: If the dataclass is mutable.
        AssertionError: If default values are incorrect.
    """
    inputs = ModelInputs()
    # Check defaults are empty arrays
    chex.assert_shape(inputs.sequence, (0,))
    chex.assert_shape(inputs.mask, (0,))

    # Check immutability
    with pytest.raises(FrozenInstanceError):
        inputs.sequence = jnp.ones((10,))  # type: ignore[assignment]