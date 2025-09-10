"""Tests for data structure definitions."""

import jax.numpy as jnp
import pytest
from dataclasses import FrozenInstanceError
from prxteinmpnn.utils.data_structures import Protein


def test_protein_structure_frozen():
    """Test that ProteinStructure dataclass is immutable.

    Raises:
        FrozenInstanceError: If the dataclass is mutable.
    """
    p = Protein(
        coordinates=jnp.zeros((1, 1, 3)),
        aatype=jnp.zeros((1,)),
        atom_mask=jnp.zeros((1, 1)),
        residue_index=jnp.zeros((1,)),
        chain_index=jnp.zeros((1,)),
    )
    with pytest.raises(FrozenInstanceError):
        p.aatype = jnp.ones((1,))  # type: ignore[assignment]