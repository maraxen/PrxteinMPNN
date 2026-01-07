"""Shared test fixtures for training tests."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax import random

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture(scope="session")
def protein_structure() -> Protein:
    """Load a sample protein structure from a PDB file."""
    pdb_path = Path(__file__).parent.parent / "data" / "1ubq.pdb"
    # parse_input returns Protein objects directly
    return next(parse_input(str(pdb_path)))


@pytest.fixture
def small_model() -> PrxteinMPNN:
    """Create a small PrxteinMPNN model for testing."""
    return PrxteinMPNN(
        node_features=32,
        edge_features=32,
        hidden_features=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=30,
        key=random.PRNGKey(0),
    )


@pytest.fixture
def mock_batch(protein_structure: Protein) -> Protein:
    """Create a mock batch of protein structures."""
    import numpy as np
    def _expand_if_array(x):
        if isinstance(x, (jnp.ndarray, np.ndarray)):
            return jnp.expand_dims(x, axis=0)
        return x  # Keep non-array fields as-is
    return jax.tree_util.tree_map(
        _expand_if_array, protein_structure,
        is_leaf=lambda x: isinstance(x, (str, type(None))),
    )


@pytest.fixture
def mock_logits() -> jax.Array:
    """Create mock logits for testing loss functions."""
    return jax.random.normal(random.PRNGKey(0), (10, 21))


@pytest.fixture
def mock_targets() -> jax.Array:
    """Create mock targets for testing loss functions."""
    return jax.random.randint(random.PRNGKey(0), (10,), 0, 21)


@pytest.fixture
def mock_mask() -> jax.Array:
    """Create a mock mask for testing loss functions."""
    return jnp.ones(10)


@pytest.fixture(params=[False, True], ids=["eager", "jit"])
def apply_jit(request):
    """Returns a function that conditionally JITs the input function."""
    should_jit = request.param

    def _wrapper(fn, **kwargs):
        if should_jit:
            return jax.jit(fn, **kwargs)
        return fn

    return _wrapper


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for checkpointing."""
    return tmp_path / "checkpoints"


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for data."""
    return tmp_path / "data"
