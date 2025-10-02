"""Unit tests for conformational inference functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import chex
import h5py
import jax
import jax.numpy as jnp
import pytest
from grain.python import IterDataset

from prxteinmpnn.ensemble.dbscan import ConformationalStates
from prxteinmpnn.run.conformational_inference import (
  _compute_states_batches,
  _derive_states_in_memory,
  _derive_states_streaming,
  _get_logits_fn,
  derive_states,
  infer_conformations,
)
from prxteinmpnn.run.specs import ConformationalInferenceSpecification


@pytest.fixture
def mock_spec():
  """Create a mock ConformationalInferenceSpecification for testing.
  
  Returns:
    ConformationalInferenceSpecification: A test specification object.
  """
  return ConformationalInferenceSpecification(
    inputs=["test_input.pdb"],
    inference_strategy="conditional",
    inference_features=["logits", "node_features"],
    mode="global",
    random_seed=42,
    gmm_n_components=3,
    eps_std_scale=1.0,
    min_cluster_weight=0.1,
  )


@pytest.fixture
def mock_model_parameters():
  """Create mock model parameters for testing.
  
  Returns:
    dict: Mock model parameters.
  """
  return {"weights": jnp.ones((10, 10)), "biases": jnp.zeros(10)}


@pytest.fixture
def mock_protein_ensemble():
  """Create a mock protein ensemble for testing.
  
  Returns:
    MagicMock: Mock protein ensemble with required attributes.
  """
  ensemble = MagicMock()
  ensemble.coordinates = jnp.ones((5, 100, 4, 3))  # 5 frames, 100 residues
  ensemble.one_hot_sequence = jnp.ones((5, 100, 20))
  ensemble.mask = jnp.ones((5, 100))
  ensemble.residue_index = jnp.arange(100)
  ensemble.chain_index = jnp.zeros(100)
  ensemble.aatype=jnp.ones((5, 100))
  ensemble.full_coordinates = jnp.ones((5, 100, 14, 3))
  return ensemble


@pytest.fixture
def mock_protein_iterator(mock_protein_ensemble):
  """Create a mock protein iterator for testing.
  
  Args:
    mock_protein_ensemble: Mock protein ensemble fixture.
    
  Returns:
    MagicMock: Mock iterator that yields the ensemble.
  """
  iterator = MagicMock(spec=IterDataset)
  iterator.__iter__ = lambda x: iter([mock_protein_ensemble])
  return iterator


class TestGetLogitsFn:
  """Test the _get_logits_fn helper function."""

  def test_conditional_strategy(self, mock_spec, mock_model_parameters):
    """Test conditional inference strategy selection.
    
    Args:
      mock_spec: Mock specification fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    with patch("prxteinmpnn.run.conformational_inference.make_conditional_logits_fn") as mock_fn:
      mock_fn.return_value = lambda *args: (jnp.array([1]), jnp.array([2]), jnp.array([3]))
      
      logits_fn, is_conditional = _get_logits_fn(mock_spec, mock_model_parameters)
      
      assert is_conditional is True
      mock_fn.assert_called_once_with(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_spec.decoding_order_fn,
      )

  def test_unconditional_strategy(self, mock_spec, mock_model_parameters):
    """Test unconditional inference strategy selection.
    
    Args:
      mock_spec: Mock specification fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    mock_spec.inference_strategy = "unconditional"
    mock_spec.reference_sequence = jnp.ones((100, 20))
    
    with patch("prxteinmpnn.run.conformational_inference.make_unconditional_logits_fn") as mock_fn:
      mock_fn.return_value = lambda *args, **kwargs: (jnp.array([1]), jnp.array([2]), jnp.array([3]))
      
      logits_fn, is_conditional = _get_logits_fn(mock_spec, mock_model_parameters)
      
      assert is_conditional is False
      mock_fn.assert_called_once_with(
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_spec.decoding_order_fn,
      )

  def test_coordinates_strategy(self, mock_spec, mock_model_parameters):
    """Test coordinates inference strategy selection.
    
    Args:
      mock_spec: Mock specification fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    mock_spec.inference_strategy = "coordinates"
    
    logits_fn, is_conditional = _get_logits_fn(mock_spec, mock_model_parameters)
    
    assert is_conditional is False
    result = logits_fn()
    assert len(result) == 3
    assert all(isinstance(arr, jax.Array) for arr in result)

  def test_vmm_strategy_not_implemented(self, mock_spec, mock_model_parameters):
    """Test that VMM strategy raises NotImplementedError.
    
    Args:
      mock_spec: Mock specification fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    mock_spec.inference_strategy = "vmm"
    
    with pytest.raises(NotImplementedError, match="VMM inference strategy is not yet implemented"):
      _get_logits_fn(mock_spec, mock_model_parameters)

  def test_invalid_strategy(self, mock_spec, mock_model_parameters):
    """Test that invalid strategy raises ValueError.
    
    Args:
      mock_spec: Mock specification fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    mock_spec.inference_strategy = "invalid"
    
    with pytest.raises(ValueError, match="Invalid inference strategy: invalid"):
      _get_logits_fn(mock_spec, mock_model_parameters)


class TestComputeStatesBatches:
  """Test the _compute_states_batches function."""

  @patch("prxteinmpnn.run.conformational_inference._get_logits_fn")
  @patch("jax.vmap")
  @patch("jax.random.split")
  def test_conditional_batches(
    self,
    mock_split,
    mock_vmap,
    mock_get_logits,
    mock_spec,
    mock_protein_iterator,
    mock_model_parameters,
  ):
    """Test batch computation for conditional inference.
    
    Args:
      mock_split: Mock JAX random split function.
      mock_vmap: Mock JAX vmap function.
      mock_get_logits: Mock get logits function.
      mock_spec: Mock specification fixture.
      mock_protein_iterator: Mock protein iterator fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    # Setup mocks
    mock_get_logits.return_value = (lambda *args: None, True)
    mock_split.return_value = jnp.ones((5, 2))
    mock_vmap.return_value = lambda *args: (
      jnp.ones((5, 100, 20)),
      jnp.ones((5, 100, 128)),
      jnp.ones((5, 100, 100, 64)),
    )
    
    # Test
    batches = list(_compute_states_batches(mock_spec, mock_protein_iterator, mock_model_parameters))
    
    assert len(batches) == 1
    logits, node_features, edge_features, backbone_coords, full_coords = batches[0]
    chex.assert_shape(logits, (5, 100, 20))
    chex.assert_shape(node_features, (5, 100, 128))

  @patch("prxteinmpnn.run.conformational_inference._get_logits_fn")
  @patch("jax.vmap")
  @patch("jax.random.split")
  def test_unconditional_batches(
    self,
    mock_split,
    mock_vmap,
    mock_get_logits,
    mock_spec,
    mock_protein_iterator,
    mock_model_parameters,
  ):
    """Test batch computation for unconditional inference.
    
    Args:
      mock_split: Mock JAX random split function.
      mock_vmap: Mock JAX vmap function.
      mock_get_logits: Mock get logits function.
      mock_spec: Mock specification fixture.
      mock_protein_iterator: Mock protein iterator fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    # Setup mocks
    mock_get_logits.return_value = (lambda *args, **kwargs: None, False)
    mock_split.return_value = jnp.ones((5, 2))
    mock_vmap.return_value = lambda *args: (
      jnp.ones((5, 100, 20)),
      jnp.ones((5, 100, 128)),
      jnp.ones((5, 100, 100, 64)),
    )
    
    # Test
    batches = list(_compute_states_batches(mock_spec, mock_protein_iterator, mock_model_parameters))
    
    assert len(batches) == 1
    mock_vmap.assert_called_once()


class TestDeriveStatesInMemory:
  """Test the _derive_states_in_memory function."""

  @patch("prxteinmpnn.run.conformational_inference._compute_states_batches")
  def test_successful_computation(
    self,
    mock_compute_batches,
    mock_spec,
    mock_protein_iterator,
    mock_model_parameters,
  ):
    """Test successful in-memory state derivation.
    
    Args:
      mock_compute_batches: Mock compute batches function.
      mock_spec: Mock specification fixture.
      mock_protein_iterator: Mock protein iterator fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    # Setup mock data
    batch1 = (
      jnp.ones((2, 100, 20)),
      jnp.ones((2, 100, 128)),
      jnp.ones((2, 100, 100, 64)),
      jnp.ones((2, 100, 4, 3)),
      jnp.ones((2, 100, 14, 3)),
    )
    batch2 = (
      jnp.ones((3, 100, 20)),
      jnp.ones((3, 100, 128)),
      jnp.ones((3, 100, 100, 64)),
      jnp.ones((3, 100, 4, 3)),
      jnp.ones((3, 100, 14, 3)),
    )
    mock_compute_batches.return_value = [batch1, batch2]
    
    # Test
    result = _derive_states_in_memory(mock_spec, mock_protein_iterator, mock_model_parameters)
    
    assert "logits" in result
    assert "node_features" in result
    assert "metadata" in result
    chex.assert_shape(result["logits"], (5, 100, 20))
    chex.assert_shape(result["node_features"], (5, 100, 128))

  @patch("prxteinmpnn.run.conformational_inference._compute_states_batches")
  def test_none_features_handling(
    self,
    mock_compute_batches,
    mock_spec,
    mock_protein_iterator,
    mock_model_parameters,
  ):
    """Test handling of None features in in-memory computation.
    
    Args:
      mock_compute_batches: Mock compute batches function.
      mock_spec: Mock specification fixture.
      mock_protein_iterator: Mock protein iterator fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    # Setup mock with None features
    batch = (None, None, None, None, None)
    mock_compute_batches.return_value = [batch]
    
    # Test
    result = _derive_states_in_memory(mock_spec, mock_protein_iterator, mock_model_parameters)
    
    assert result["logits"] is None
    assert result["node_features"] is None
    assert result["edge_features"] is None


class TestDeriveStatesStreaming:
  """Test the _derive_states_streaming function."""

  def test_missing_output_path(self, mock_spec, mock_protein_iterator, mock_model_parameters):
    """Test error when output_h5_path is not provided.
    
    Args:
      mock_spec: Mock specification fixture.
      mock_protein_iterator: Mock protein iterator fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    mock_spec.output_h5_path = None
    
    with pytest.raises(ValueError, match="output_h5_path must be provided for streaming"):
      _derive_states_streaming(mock_spec, mock_protein_iterator, mock_model_parameters)

  @patch("prxteinmpnn.run.conformational_inference._compute_states_batches")
  @patch("h5py.File")
  def test_successful_streaming(
    self,
    mock_h5py_file,
    mock_compute_batches,
    mock_spec,
    mock_protein_iterator,
    mock_model_parameters,
  ):
    """Test successful streaming to HDF5 file.
    
    Args:
      mock_h5py_file: Mock HDF5 file object.
      mock_compute_batches: Mock compute batches function.
      mock_spec: Mock specification fixture.
      mock_protein_iterator: Mock protein iterator fixture.
      mock_model_parameters: Mock model parameters fixture.
    """
    # Setup
    mock_spec.output_h5_path = "/tmp/test.h5"
    
    # Mock HDF5 file and datasets
    mock_file = MagicMock()
    mock_h5py_file.return_value.__enter__ = lambda x: mock_file
    mock_h5py_file.return_value.__exit__ = lambda *args: None
    
    mock_dataset = MagicMock()
    mock_dataset.shape = [0, 100, 20]
    mock_file.create_dataset.return_value = mock_dataset
    
    # Mock batch data
    batch = (
      jnp.ones((2, 100, 20)),
      jnp.ones((2, 100, 128)),
      None,
      None,
      None,
    )
    mock_compute_batches.return_value = [batch]
    
    # Test
    result = _derive_states_streaming(mock_spec, mock_protein_iterator, mock_model_parameters)
    
    assert result["output_h5_path"] == "/tmp/test.h5"
    assert "metadata" in result
    mock_file.create_dataset.assert_called()
    mock_dataset.resize.assert_called()


class TestDeriveStates:
  """Test the main derive_states function."""

  @patch("prxteinmpnn.run.conformational_inference.prep_protein_stream_and_model")
  @patch("prxteinmpnn.run.conformational_inference._derive_states_in_memory")
  def test_in_memory_mode(self, mock_in_memory, mock_prep, mock_spec):
    """Test derive_states in in-memory mode.
    
    Args:
      mock_in_memory: Mock in-memory function.
      mock_prep: Mock prep function.
      mock_spec: Mock specification fixture.
    """
    mock_spec.output_h5_path = None
    mock_prep.return_value = (MagicMock(), {})
    mock_in_memory.return_value = {"logits": jnp.ones((5, 100, 20))}
    
    result = derive_states(mock_spec)
    
    mock_in_memory.assert_called_once()
    assert "logits" in result

  @patch("prxteinmpnn.run.conformational_inference.prep_protein_stream_and_model")
  @patch("prxteinmpnn.run.conformational_inference._derive_states_streaming")
  def test_streaming_mode(self, mock_streaming, mock_prep, mock_spec):
    """Test derive_states in streaming mode.
    
    Args:
      mock_streaming: Mock streaming function.
      mock_prep: Mock prep function.
      mock_spec: Mock specification fixture.
    """
    mock_spec.output_h5_path = "/tmp/test.h5"
    mock_prep.return_value = (MagicMock(), {})
    mock_streaming.return_value = {"output_h5_path": "/tmp/test.h5"}
    
    result = derive_states(mock_spec)
    
    mock_streaming.assert_called_once()
    assert result["output_h5_path"] == "/tmp/test.h5"

  def test_spec_creation_from_kwargs(self):
    """Test creating spec from kwargs when not provided."""
    with patch("prxteinmpnn.run.conformational_inference.prep_protein_stream_and_model") as mock_prep:
      with patch("prxteinmpnn.run.conformational_inference._derive_states_in_memory") as mock_in_memory:
        mock_prep.return_value = (MagicMock(), {})
        mock_in_memory.return_value = {}
        
        derive_states(inputs=["test_input.pdb"],inference_strategy="conditional")
        
        mock_prep.assert_called_once()


class TestInferConformations:
  """Test the main infer_conformations function."""

  @patch("prxteinmpnn.run.conformational_inference.derive_states")
  @patch("prxteinmpnn.run.conformational_inference.make_fit_gmm_in_memory")
  @patch("prxteinmpnn.run.conformational_inference.infer_states")
  def test_successful_inference_in_memory(
    self,
    mock_infer_states,
    mock_make_fit_gmm_in_memory,
    mock_derive_states,
    mock_spec,
  ):
    """Test successful conformational inference with in-memory data.
    
    Args:
      mock_infer_states: Mock infer states function.
      mock_make_fit_gmm_in_memory: Mock GMM fitting function.
      mock_derive_states: Mock derive states function.
      mock_spec: Mock specification fixture.
    """
    # Setup
    mock_spec.output_h5_path = None
    mock_spec.inference_strategy = "logits"
    mock_spec.mode = "global"
    
    # Mock data
    test_states = jnp.ones((10, 100, 20))
    mock_derive_states.return_value = {"logits": test_states}
    
    mock_gmm = MagicMock()
    mock_gmm_fitter = MagicMock(return_value=mock_gmm)
    mock_make_fit_gmm_in_memory.return_value = mock_gmm_fitter
    
    mock_conformational_states = MagicMock(spec=ConformationalStates)
    mock_infer_states.return_value = mock_conformational_states
    
    # Test
    result = infer_conformations(mock_spec)
    
    assert result == mock_conformational_states
    mock_derive_states.assert_called_once_with(mock_spec)
    mock_make_fit_gmm_in_memory.assert_called_once()
    mock_infer_states.assert_called_once()

  @patch("prxteinmpnn.run.conformational_inference.derive_states")
  @patch("h5py.File")
  def test_successful_inference_streaming(
    self,
    mock_h5py_file,
    mock_derive_states,
    mock_spec,
  ):
    """Test successful conformational inference with streamed data.
    
    Args:
      mock_h5py_file: Mock HDF5 file object.
      mock_derive_states: Mock derive states function.
      mock_spec: Mock specification fixture.
    """
    # Setup
    mock_spec.output_h5_path = "/tmp/test.h5"
    mock_spec.inference_strategy = "logits"
    
    mock_derive_states.return_value = {"output_h5_path": "/tmp/test.h5"}
    
    # Mock HDF5 data
    mock_file = MagicMock()
    mock_dataset = jnp.ones((10, 100, 20))
    mock_h5py_file.return_value.__enter__.return_value = mock_file
    mock_h5py_file.return_value.__exit__.return_value = None
    mock_file.__getitem__.return_value = mock_dataset
    
    with patch("prxteinmpnn.run.conformational_inference.make_fit_gmm_streaming") as mock_make_fit_gmm:
      with patch("prxteinmpnn.run.conformational_inference.infer_states") as mock_infer_states:
        mock_gmm_fitter = MagicMock()
        mock_make_fit_gmm.return_value = mock_gmm_fitter
        mock_infer_states.return_value = MagicMock(spec=ConformationalStates)
        
        result = infer_conformations(mock_spec)
        
        assert isinstance(result, MagicMock)
        mock_h5py_file.assert_called_once_with("/tmp/test.h5", "r")

  @patch("prxteinmpnn.run.conformational_inference.derive_states")
  def test_no_data_error(self, mock_derive_states, mock_spec):
    """Test error when no data is available for GMM fitting.
    
    Args:
      mock_derive_states: Mock derive states function.
      mock_spec: Mock specification fixture.
    """
    mock_spec.output_h5_path = None
    mock_spec.inference_strategy = "logits"
    mock_derive_states.return_value = {"logits": None}
    
    with pytest.raises(ValueError, match="No data available for GMM fitting"):
      infer_conformations(mock_spec)

  @patch("prxteinmpnn.run.conformational_inference.derive_states")
  @patch("prxteinmpnn.run.conformational_inference.make_fit_gmm_in_memory")
  @patch("prxteinmpnn.run.conformational_inference.infer_states")
  def test_per_residue_mode(
    self,
    mock_infer_states,
    mock_make_fit_gmm_in_memory,
    mock_derive_states,
    mock_spec,
  ):
    """Test conformational inference in per-residue mode.
    
    Args:
      mock_infer_states: Mock infer states function.
      mock_make_fit_gmm_in_memory: Mock GMM fitting function.
      mock_derive_states: Mock derive states function.
      mock_spec: Mock specification fixture.
    """
    # Setup
    mock_spec.mode = "per"
    mock_spec.inference_strategy = "logits"
    mock_spec.output_h5_path = None
    
    test_states = jnp.ones((10, 100, 20))
    mock_derive_states.return_value = {"logits": test_states}
    
    mock_gmm_fitter = MagicMock()
    mock_make_fit_gmm_in_memory.return_value = mock_gmm_fitter
    mock_infer_states.return_value = MagicMock(spec=ConformationalStates)
    
    # Test
    infer_conformations(mock_spec)
    
    # Check that make_fit_gmm was called with per-residue parameters
    mock_make_fit_gmm_in_memory.assert_called_once_with(
      n_components=mock_spec.gmm_n_components,
      gmm_max_iters=100,
    )


class TestIntegration:
  """Integration tests for the conformational inference module."""

  def test_end_to_end_workflow(self):
    """Test the complete workflow with minimal real data."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create a minimal test specification
      spec = ConformationalInferenceSpecification(
        inputs=["tests/data/minimal_ensemble.pdb"],
        inference_strategy="coordinates",
        inference_features=["backbone_coordinates"],
        mode="global",
        random_seed=42,
        gmm_n_components=2,
        eps_std_scale=1.0,
        min_cluster_weight=0.1,
        output_h5_path=Path(tmpdir) / "test.h5",
      )
      
      # Mock the protein stream and model
      with patch("prxteinmpnn.run.conformational_inference.prep_protein_stream_and_model") as mock_prep:
        # Create mock ensemble
        ensemble = MagicMock()
        ensemble.coordinates = jnp.ones((3, 10, 4, 3))
        ensemble.mask = jnp.ones((10,))
        ensemble.residue_index = jnp.arange(10)
        ensemble.chain_index = jnp.zeros(10)
        ensemble.full_coordinates = jnp.ones((3, 10, 14, 3))
        
        iterator = MagicMock()
        iterator.__iter__ = lambda x: iter([ensemble])
        
        mock_prep.return_value = (iterator, {})
        
        # Test derive_states function
        result = derive_states(spec)
        
        assert "output_h5_path" in result
        assert Path(result["output_h5_path"]).exists()
