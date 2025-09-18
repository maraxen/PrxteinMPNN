"""Tests for the catjac utility functions."""

import jax
import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

# The new function signatures require jac_A and jac_B to be passed
from prxteinmpnn.utils.catjac import (
    _add_jacobians_mapped,
    _subtract_jacobians_mapped,
    _gather_mapped_jacobian,
    make_combine_jac,
)
# We assume a mock or real align_sequences is available for the factory tests
from prxteinmpnn.utils.align import align_sequences
import tempfile
import h5py
import numpy as np
from pathlib import Path
from prxteinmpnn.utils.catjac import combine_jacobians_h5_stream


@pytest.fixture
def sample_jacobians():
    """Fixture for a sample CategoricalJacobian tensor."""
    key = jax.random.PRNGKey(0)
    # Shape: (N, noise_levels, L, 21, L, 21) -> (2, 3, 5, 21, 5, 21)
    return jax.random.normal(key, shape=(2, 3, 5, 21, 5, 21))


@pytest.fixture
def sample_mapping():
    """Fixture for a single inter-protein index mapping."""
    # This now represents a single (L, 2) index map, like align_sequences produces.
    # It maps all 5 positions identically.
    i_indices = jnp.arange(5)
    k_indices = jnp.arange(5)
    return jnp.stack([i_indices, k_indices], axis=-1)


@pytest.fixture
def sample_sequences():
    """Fixture for sample ProteinSequences."""
    # Shape: (N, L) -> (2, 5)
    return jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])


@pytest.fixture
def sample_weights():
    """Fixture for sample weights."""
    # Shape: (N,) -> (2,)
    return jnp.array([1.0, 0.5])


@pytest.fixture
def sample_pair_tuple(sample_jacobians, sample_mapping, sample_weights):
    """Fixture for a tuple representing a single pair to be combined."""
    # New signature: (jac_A, jac_B, mapping, weight)
    return (
        sample_jacobians[0],  # jac_A
        sample_jacobians[1],  # jac_B
        sample_mapping,       # The index map
        sample_weights[0],    # A scalar weight for the pair
    )


# --- Tests for internal "mapped" functions ---

def test_add_jacobians(sample_pair_tuple):
    """Test the _add_jacobians_mapped function with the new signature."""
    jac_A, _, _, _ = sample_pair_tuple
    combined = _add_jacobians_mapped(*sample_pair_tuple)
    assert combined.shape == jac_A.shape
    assert not jnp.allclose(combined, jac_A)


def test_subtract_jacobians(sample_pair_tuple):
    """Test the _subtract_jacobians_mapped function with the new signature."""
    jac_A, _, _, _ = sample_pair_tuple
    combined = _subtract_jacobians_mapped(*sample_pair_tuple)
    assert combined.shape == jac_A.shape
    assert not jnp.allclose(combined, jac_A)


# --- Tests for the public-facing factory ---

def test_make_combine_jac_add(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'add' operation."""
    combine_fn = make_combine_jac("add")
    combined, mapping = combine_fn(sample_jacobians, sample_sequences, None)
    # For N=2, triu_indices gives 1 pair, so the batch size is 1.
    assert combined.shape == (1, 3, 5, 21, 5, 21)
    assert mapping.shape == (1, 5, 2)


def test_make_combine_jac_subtract(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'subtract' operation."""
    combine_fn = make_combine_jac("subtract")
    combined, mapping = combine_fn(sample_jacobians, sample_sequences, None)
    # For N=2, triu_indices gives 1 pair, so the batch size is 1.
    assert combined.shape == (1, 3, 5, 21, 5, 21)
    assert mapping.shape == (1, 5, 2)


def test_make_combine_jac_subtract_identity_is_zero(
    sample_jacobians, sample_sequences
):
    """Test that subtracting an identical Jacobian from itself results in zero."""
    # Create a batch where both proteins are identical
    identical_jacobians = jnp.stack([sample_jacobians[0], sample_jacobians[0]])
    identical_sequences = jnp.stack([sample_sequences[0], sample_sequences[0]])

    combine_fn = make_combine_jac("subtract")

    # The result of J_0 - map(J_0) for the single pair (0, 1) should be zero.
    jac, _ = combine_fn(identical_jacobians, identical_sequences, None)

    # Check that the result is close to zero within a small tolerance
    assert_trees_all_close(jac, jnp.zeros_like(jac), atol=1e-6)
    
def test_make_combine_jac_custom_function(
    sample_jacobians, sample_sequences, sample_pair_tuple
):
    """Test the make_combine_jac factory with a custom combine function."""
    def custom_combine(jac_A, jac_B, mapping, weights):
        # Simple custom function that adds twice the mapped jac_B
        mapped_jac_B = _gather_mapped_jacobian(jac_B, mapping)
        return jac_A + 2.0 * mapped_jac_B * (weights if weights is not None else 1.0)

    combine_fn = make_combine_jac(custom_combine)

    combined, mapping = combine_fn(sample_jacobians, sample_sequences, None)
    assert combined.shape == (1, 3, 5, 21, 5, 21)
    # Ensure the result is different from simple addition
    simple_add_fn = make_combine_jac("add")
    simple_added, _ = simple_add_fn(sample_jacobians, sample_sequences, None)
    assert not jnp.allclose(combined, simple_added)
    





@pytest.fixture
def temp_h5_file(sample_jacobians, sample_sequences):
  """Create a temporary HDF5 file with sample data."""
  with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
    tmp_path = Path(tmp.name)
  
  # Convert JAX arrays to numpy for HDF5 storage
  jacobians_np = np.array(sample_jacobians)
  sequences_np = np.array(sample_sequences)
  
  with h5py.File(tmp_path, "w") as f:
    f.create_dataset("categorical_jacobians", data=jacobians_np, chunks=True, maxshape=(None, *jacobians_np.shape[1:]))
    f.create_dataset("one_hot_sequences", data=sequences_np, chunks=True, maxshape=(None, *sequences_np.shape[1:]))
  
  yield tmp_path
  
  # Cleanup
  tmp_path.unlink(missing_ok=True)


def test_combine_jacobians_h5_stream_basic(temp_h5_file, sample_weights):
  """Test basic functionality of combine_jacobians_h5_stream."""
  def simple_combine(jac_A, jac_B, mapping, weights):
    return _add_jacobians_mapped(jac_A, jac_B, mapping, weights)
  
  combine_jacobians_h5_stream(
    h5_path=temp_h5_file,
    combine_fn=simple_combine,
    fn_kwargs={},
    batch_size=1,
    weights=sample_weights,
  )
  
  # Verify output dataset was created
  with h5py.File(temp_h5_file, "r") as f:
    assert "combined_catjac" in f
    combined_data = f["combined_catjac"]
    # Should have n_samples * (n_samples - 1) / 2 = 2 * 1 / 2 = 1 pair
    assert combined_data.shape[0] == 1 # pyright: ignore[reportAttributeAccessIssue]
    assert combined_data.dtype == np.float32 # pyright: ignore[reportAttributeAccessIssue]


def test_combine_jacobians_h5_stream_mismatched_jacobian_sequence_length(temp_h5_file):
  """Test ValueError when jacobians and sequences have mismatched lengths."""
  # Add mismatched sequence data
  with h5py.File(temp_h5_file, "a") as f:
    # Original has 2 jacobians, add 3 sequences
    f["one_hot_sequences"].resize((3, 5)) # pyright: ignore[reportAttributeAccessIssue]
  
  weights = jnp.ones(2)
  
  with pytest.raises(ValueError, match="Jacobian, sequence, and weights arrays must have the same length."):
    combine_jacobians_h5_stream(
      h5_path=temp_h5_file,
      combine_fn=_add_jacobians_mapped,
      fn_kwargs={},
      batch_size=1,
      weights=weights,
    )


def test_combine_jacobians_h5_stream_mismatched_weights_length(temp_h5_file):
  """Test ValueError when weights array doesn't match number of samples."""
  # Weights with wrong length
  weights = jnp.ones(3)  # Should be 2 to match sample data
  
  with pytest.raises(ValueError, match="Jacobian, sequence, and weights arrays must have the same length."):
    combine_jacobians_h5_stream(
      h5_path=temp_h5_file,
      combine_fn=_add_jacobians_mapped,
      fn_kwargs={},
      batch_size=1,
      weights=weights,
    )


def test_combine_jacobians_h5_stream_with_fn_kwargs(temp_h5_file, sample_weights):
  """Test combine_jacobians_h5_stream with function kwargs."""
  def custom_combine_with_kwargs(jac_A, jac_B, mapping, weights, scale_factor=1.0):
    mapped_jac_B = _gather_mapped_jacobian(jac_B, mapping)
    return jac_A + scale_factor * mapped_jac_B * weights
  
  fn_kwargs = {"scale_factor": 2.0}
  
  combine_jacobians_h5_stream(
    h5_path=temp_h5_file,
    combine_fn=custom_combine_with_kwargs,
    fn_kwargs=fn_kwargs,
    batch_size=1,
    weights=sample_weights,
  )
  
  with h5py.File(temp_h5_file, "r") as f:
    assert "combined_catjac" in f
    combined_data = f["combined_catjac"]
    assert combined_data.shape[0] == 1 # pyright: ignore[reportAttributeAccessIssue]


def test_combine_jacobians_h5_stream_different_batch_sizes(temp_h5_file, sample_weights):
  """Test combine_jacobians_h5_stream with different batch sizes."""
  for batch_size in [1, 2, 4]:
    # Reset file for each test
    with h5py.File(temp_h5_file, "a") as f:
      if "combined_catjac" in f:
        del f["combined_catjac"]
    
    combine_jacobians_h5_stream(
      h5_path=temp_h5_file,
      combine_fn=_add_jacobians_mapped,
      fn_kwargs={},
      batch_size=batch_size,
      weights=sample_weights,
    )
    
    with h5py.File(temp_h5_file, "r") as f:
      combined_data = f["combined_catjac"]
      assert combined_data.shape[0] == 1 # pyright: ignore[reportAttributeAccessIssue]
      assert combined_data.dtype == np.float32 # pyright: ignore[reportAttributeAccessIssue]
    
    #clear for next iteration
    with h5py.File(temp_h5_file, "a") as f:
      if "combined_catjac" in f:
        del f["combined_catjac"]
      if "mappings" in f:
        del f["mappings"]


def test_combine_jacobians_h5_stream_output_shape_consistency(temp_h5_file, sample_weights):
  """Test that output shape is consistent regardless of processing order."""
  combine_jacobians_h5_stream(
    h5_path=temp_h5_file,
    combine_fn=_subtract_jacobians_mapped,
    fn_kwargs={},
    batch_size=1,
    weights=sample_weights,
  )
  
  with h5py.File(temp_h5_file, "r") as f:
    jacobians = f["categorical_jacobians"]
    combined_data = f["combined_catjac"]
    
    # Output should have same shape as input jacobians except for batch dimension
    expected_shape = (1, *jacobians.shape[1:]) # pyright: ignore[reportAttributeAccessIssue]
    assert combined_data.shape == expected_shape # pyright: ignore[reportAttributeAccessIssue]


@pytest.fixture
def larger_h5_file():
  """Create a larger HDF5 file for batch processing tests."""
  with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
    tmp_path = Path(tmp.name)
  
  # Create larger test data
  key = jax.random.PRNGKey(42)
  jacobians = jax.random.normal(key, shape=(4, 3, 5, 21, 5, 21))
  sequences = jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1]])
  
  jacobians_np = np.array(jacobians)
  sequences_np = np.array(sequences)
  
  with h5py.File(tmp_path, "w") as f:
    f.create_dataset("categorical_jacobians", data=jacobians_np, chunks=True)
    f.create_dataset("one_hot_sequences", data=sequences_np, chunks=True)
  
  yield tmp_path
  
  tmp_path.unlink(missing_ok=True)


def test_combine_jacobians_h5_stream_larger_dataset(larger_h5_file):
  """Test combine_jacobians_h5_stream with a larger dataset."""
  weights = jnp.ones(4)
  
  combine_jacobians_h5_stream(
    h5_path=larger_h5_file,
    combine_fn=_add_jacobians_mapped,
    fn_kwargs={},
    batch_size=2,
    weights=weights,
  )
  
  with h5py.File(larger_h5_file, "r") as f:
    combined_data = f["combined_catjac"]
    # Should have 4 * 3 / 2 = 6 pairs
    assert combined_data.shape[0] == 6 # pyright: ignore[reportAttributeAccessIssue]
    assert combined_data.dtype == np.float32 # pyright: ignore[reportAttributeAccessIssue]


def test_combine_jacobians_h5_stream_empty_dataset():
  """Test combine_jacobians_h5_stream with empty dataset."""
  with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
    tmp_path = Path(tmp.name)
  
  # Create empty datasets
  with h5py.File(tmp_path, "w") as f:
    f.create_dataset("categorical_jacobians", shape=(0, 3, 5, 21, 5, 21), chunks=True)
    f.create_dataset("one_hot_sequences", shape=(0, 5), chunks=True)
  
  weights = jnp.array([])
  
  try:
    combine_jacobians_h5_stream(
      h5_path=tmp_path,
      combine_fn=_add_jacobians_mapped,
      fn_kwargs={},
      batch_size=1,
      weights=weights,
    )
    
    with h5py.File(tmp_path, "r") as f:
      combined_data = f["combined_catjac"]
      assert combined_data.shape[0] == 0 # pyright: ignore[reportAttributeAccessIssue]
  finally:
    tmp_path.unlink(missing_ok=True)