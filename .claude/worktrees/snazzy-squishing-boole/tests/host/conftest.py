"""Fixtures specific to host coverage tests."""

import inspect

import jax
import pytest
from jax import random

from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN


@pytest.fixture(scope="module")
def minimal_model() -> PrxteinLigandMPNN:
    """Create a minimal PrxteinLigandMPNN for testing.

    Uses tiny dimensions (1 encoder layer, 1 decoder layer, 16 hidden dims)
    for fast instantiation and testing.
    """
    key = jax.random.key(42)
    model = PrxteinLigandMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=3,
        num_context_layers=1,
        num_positional_embeddings=8,
        num_amino_acids=21,
        vocab_size=21,
        dropout_rate=0.0,  # Disable dropout for determinism
        ligand_l_chunk=16,
        ligand_mpnn_use_side_chain_context=False,
        key=key,
    )
    return model
