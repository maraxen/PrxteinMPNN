"""Tests for the catjac utility functions."""

import tempfile
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from chex import assert_trees_all_close

from prxteinmpnn.utils.catjac import (
  _combine_mapped_jacobians,
  _gather_mapped_jacobian,
  combine_jacobians_h5_stream,
  make_combine_jac,
)


@pytest.fixture
def sample_jacobians():
    """Fixture for a sample CategoricalJacobian tensor."""
    key = jax.random.PRNGKey(0)
    return jnp.tile(jax.random.normal(key, shape=(2, 1, 5, 21, 5, 21)), (1, 3, 1, 1, 1, 1))


@pytest.fixture
def sample_mapping():
    """Fixture for a single inter-protein index mapping."""
    i_indices = jnp.arange(5)
    return jnp.stack([i_indices, i_indices], axis=-1)


@pytest.fixture
def sample_sequences():
    """Fixture for sample ProteinSequences."""
    return jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])


@pytest.fixture
def sample_weights():
    """Fixture for sample weights."""
    return jnp.array([1.0, 0.5])


# --- Tests for internal "mapped" functions ---

def test_combine_mapped_jacobians_add(sample_jacobians, sample_mapping):
    """Test the _combine_mapped_jacobians function for addition."""
    jac_A, jac_B = sample_jacobians[0], sample_jacobians[1]
    combined = _combine_mapped_jacobians(jac_A, jac_B, sample_mapping, weights=jnp.array([1.0, 1.0]))
    k = jac_A.shape[0]
    assert combined.shape == (k * k, *jac_A.shape[1:])
    assert not jnp.allclose(combined, jnp.tile(jac_A, (k, *[1] * (len(jac_A.shape) - 1))))


def test_combine_mapped_jacobians_subtract_identity(sample_jacobians, sample_mapping):
    """Test that subtracting an identical Jacobian from itself results in zero."""
    jac_A = sample_jacobians[0]
    weights = jnp.array([1.0, -1.0])
    combined = _combine_mapped_jacobians(jac_A, jac_A, sample_mapping, weights)
    k = jac_A.shape[0]
    assert combined.shape == (k * k, *jac_A.shape[1:])
    assert_trees_all_close(combined, jnp.zeros_like(combined), atol=1e-6)


# --- Tests for the public-facing factory ---

def test_make_combine_jac_add(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with default 'add' operation."""
    combine_fn = make_combine_jac()
    combined, mapping = combine_fn(sample_jacobians, sample_sequences, None)
    assert combined.shape == (1, 9, 5, 21, 5, 21)
    assert mapping.shape == (1, 5, 2)


def test_make_combine_jac_subtract(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'subtract' operation."""
    combine_fn = make_combine_jac()
    combined, mapping = combine_fn(sample_jacobians, sample_sequences, jnp.array([1.0, -1.0]))
    assert combined.shape == (1, 9, 5, 21, 5, 21)
    assert mapping.shape == (1, 5, 2)


def test_make_combine_jac_subtract_identity_is_zero(sample_jacobians, sample_sequences):
    """Test that subtracting an identical Jacobian from itself results in zero."""
    identical_jacobians = jnp.stack([sample_jacobians[0], sample_jacobians[0]])
    identical_sequences = jnp.stack([sample_sequences[0], sample_sequences[0]])
    combine_fn = make_combine_jac()
    jac, _ = combine_fn(identical_jacobians, identical_sequences, jnp.array([1.0, -1.0]))
    assert_trees_all_close(jac, jnp.zeros_like(jac), atol=1e-6)


def test_make_combine_jac_custom_function(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with a custom combine function."""
    def custom_combine(jac_A, jac_B, mapping, weights, **kwargs):
        mapped_jac_B = _gather_mapped_jacobian(jac_B, mapping)
        w = jnp.ones(2, dtype=jac_A.dtype) if weights is None else weights
        jac_A_exp = jac_A[:, None, ...]
        mapped_jac_B_exp = mapped_jac_B[None, :, ...]
        combined = (jac_A_exp * w[0]) + (2.0 * mapped_jac_B_exp * w[1]) # Custom: scale by 2.0
        k = jac_A.shape[0]
        return combined.reshape((k * k, *jac_A.shape[1:]))

    combine_fn = make_combine_jac(combine_fn=custom_combine)
    combined, _ = combine_fn(sample_jacobians, sample_sequences, None)
    assert combined.shape == (1, 9, 5, 21, 5, 21)

    default_fn = make_combine_jac()
    simple_added, _ = default_fn(sample_jacobians, sample_sequences, None)
    assert not jnp.allclose(combined, simple_added)


@pytest.fixture
def temp_h5_file(sample_jacobians, sample_sequences):
  """Create a temporary HDF5 file with sample data."""
  with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
    tmp_path = Path(tmp.name)

  with h5py.File(tmp_path, "w") as f:
    f.create_dataset("categorical_jacobians", data=np.array(sample_jacobians))
    f.create_dataset("one_hot_sequences", data=np.array(sample_sequences))

  yield tmp_path
  tmp_path.unlink(missing_ok=True)


def test_combine_jacobians_h5_stream_basic(temp_h5_file, sample_weights):
  """Test basic functionality of combine_jacobians_h5_stream."""
  combine_jacobians_h5_stream(
    h5_path=temp_h5_file,
    batch_size=1,
    weights=sample_weights,
  )
  with h5py.File(temp_h5_file, "r") as f:
    assert "combined_catjac" in f
    assert "mappings" in f
    k = f["categorical_jacobians"].shape[1]
    assert f["combined_catjac"].shape == (1, k * k, 5, 21, 5, 21)


def test_combine_jacobians_h5_stream_mismatched_lengths(temp_h5_file):
  """Test ValueError when jacobians and sequences have mismatched lengths."""
  with h5py.File(temp_h5_file, "a") as f:
    del f["one_hot_sequences"]
    f.create_dataset("one_hot_sequences", data=np.array(jnp.ones((3, 5))))

  with pytest.raises(ValueError, match="Jacobian and sequence arrays must have the same length."):
    combine_jacobians_h5_stream(h5_path=temp_h5_file, weights=jnp.ones(2))


def test_combine_jacobians_h5_stream_larger_dataset():
  """Test combine_jacobians_h5_stream with a larger dataset."""
  with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
    tmp_path = Path(tmp.name)

  n_samples = 4
  jacobians = np.random.randn(n_samples, 3, 5, 21, 5, 21)
  sequences = np.random.randint(0, 20, (n_samples, 5))

  with h5py.File(tmp_path, "w") as f:
    f.create_dataset("categorical_jacobians", data=jacobians)
    f.create_dataset("one_hot_sequences", data=sequences)

  combine_jacobians_h5_stream(h5_path=tmp_path, batch_size=2, weights=jnp.ones(n_samples))

  with h5py.File(tmp_path, "r") as f:
    num_pairs = n_samples * (n_samples - 1) // 2
    assert f["combined_catjac"].shape[0] == num_pairs

  tmp_path.unlink(missing_ok=True)
