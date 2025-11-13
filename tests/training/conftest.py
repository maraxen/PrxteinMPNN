"""Shared fixtures for training tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

if TYPE_CHECKING:
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for training data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train").mkdir()
    (data_dir / "val").mkdir()
    return data_dir


@pytest.fixture
def small_model() -> PrxteinMPNN:
    """Create a small PrxteinMPNN model for testing."""
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    
    key = jax.random.PRNGKey(0)
    model = PrxteinMPNN(
        node_features=32,
        edge_features=32,
        hidden_features=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=8,
        num_amino_acids=21,
        vocab_size=21,
        key=key,
    )
    return model


@pytest.fixture
def mock_batch() -> Protein:
    """Create a mock protein batch for testing.
    
    Returns a batch with proper structure for trainer.train_step:
    - coordinates: (batch_size, seq_len, 4, 3)
    - mask: (batch_size, seq_len)
    - residue_index: (batch_size, seq_len)
    - chain_index: (batch_size, seq_len)
    - aatype: (batch_size, seq_len)
    """
    from prxteinmpnn.utils.data_structures import Protein
    
    batch_size = 2
    seq_len = 8
    
    # Use coordinates that form valid protein structure
    # All atoms at same position for simplicity
    coordinates = jnp.ones((batch_size, seq_len, 4, 3)) * jnp.array([[[0.0, 0.0, 0.0]]])
    
    return Protein(
        coordinates=coordinates,
        mask=jnp.ones((batch_size, seq_len)),
        residue_index=jnp.tile(jnp.arange(seq_len), (batch_size, 1)),
        chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        aatype=jnp.tile(jnp.arange(seq_len) % 21, (batch_size, 1)),
        full_atom_mask=jnp.ones((batch_size, seq_len, 1)),
        one_hot_sequence=jax.nn.one_hot(
            jnp.tile(jnp.arange(seq_len) % 21, (batch_size, 1)),
            21,
        ),
    )


@pytest.fixture
def mock_logits() -> jax.Array:
    """Create mock logits for testing."""
    return jax.random.normal(jax.random.PRNGKey(0), (10, 21))


@pytest.fixture
def mock_targets() -> jax.Array:
    """Create mock target sequences."""
    return jnp.arange(10) % 21


@pytest.fixture
def mock_mask() -> jax.Array:
    """Create mock mask."""
    return jnp.ones(10)