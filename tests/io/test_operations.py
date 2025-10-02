"""Tests for Grain operations for processing protein structures."""

import io
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import requests

from prxteinmpnn.io.operations import (

    pad_and_collate_proteins,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinBatch, ProteinTuple

class TestPadAndCollate:
    """Tests for the pad_and_collate_proteins function."""

    def test_pad_and_collate(self) -> None:
        """Test correct batching and padding of proteins."""
        p1_tuple = ProteinTuple(
            coordinates=np.ones((10, 37, 3)),
            aatype=np.ones(10, dtype=np.int8),
            atom_mask=np.ones((10, 37)),
            residue_index=np.arange(10),
            chain_index=np.zeros(10, dtype=np.int32),
            dihedrals=None,
            source=None,
            mapping=None,
        )
        p2_tuple = ProteinTuple(
            coordinates=np.ones((15, 3)),
            aatype=np.ones(15, dtype=np.int8),
            atom_mask=np.ones((15, 37)),
            residue_index=np.arange(15),
            chain_index=np.zeros(15, dtype=np.int32),
            dihedrals=None,
            source=None,
            mapping=None,
        )

        elements: list[ProteinTuple] = [p1_tuple, p2_tuple]
        batch: Protein = pad_and_collate_proteins(elements)

        assert isinstance(batch, Protein)
        assert batch.coordinates.shape == (2, 15, 3)
        assert batch.aatype.shape == (2, 15)
        assert batch.mask.shape == (2, 15, 3)
        assert batch.residue_index.shape == (2, 15)
        assert batch.chain_index.shape == (2, 15)

        # Check that the first protein is padded correctly
        assert jnp.all(batch.coordinates[0, 10:] == 0)
        assert jnp.all(batch.aatype[0, 10:] == 0)

    def test_collate_empty_list_raises_error(self) -> None:
        """Test that collating an empty list raises a ValueError."""
        with pytest.raises(ValueError, match="Cannot collate an empty list"):
            pad_and_collate_proteins([])
