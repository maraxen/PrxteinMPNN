"""Tests for the core user interface module."""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.run import (
  categorical_jacobian,
  sample,
  score,
  tuple_to_protein,
  compute_cross_protein_jacobian_diffs,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinEnsemble, ProteinTuple


class TestTupleToProtein:
  """Test the tuple_to_protein utility function."""

  def test_tuple_to_protein_basic(self) -> None:
    """Test basic conversion from ProteinTuple to Protein."""
    coords = np.array([[1.0, 2.0, 3.0]])
    aatype = np.array([1])
    atom_mask = np.array([1.0])
    residue_index = np.array([0])
    chain_index = np.array([0])
    dihedrals = None

    protein_tuple: ProteinTuple = (
      coords,
      aatype,
      atom_mask,
      residue_index,
      chain_index,
      dihedrals,
    )

    protein = tuple_to_protein(protein_tuple)

    assert isinstance(protein, Protein)
    assert protein.coordinates.dtype == jnp.float32
    assert protein.aatype.dtype == jnp.int8
    assert protein.atom_mask.dtype == jnp.float16
    assert protein.residue_index.dtype == jnp.int32
    assert protein.chain_index.dtype == jnp.int32
    assert protein.dihedrals is None
    assert protein.one_hot_sequence.shape == (1, 21)

  def test_tuple_to_protein_with_dihedrals(self) -> None:
    """Test conversion with dihedrals included."""
    coords = np.array([[1.0, 2.0, 3.0]])
    aatype = np.array([1])
    atom_mask = np.array([1.0])
    residue_index = np.array([0])
    chain_index = np.array([0])
    dihedrals = np.array([[0.1, 0.2, 0.3]])

    protein_tuple: ProteinTuple = (
      coords,
      aatype,
      atom_mask,
      residue_index,
      chain_index,
      dihedrals,
    )

    protein = tuple_to_protein(protein_tuple)

    assert protein.dihedrals is not None
    assert protein.dihedrals.dtype == jnp.float32
    assert protein.dihedrals.shape == (1, 3)


class TestScore:
  """Test the score function."""

  @pytest.mark.asyncio
  async def test_score_single_structure_single_sequence(self) -> None:
    """Test scoring a single structure with a single sequence."""
    mock_protein_tuple = (
      [[[1.0, 2.0, 3.0]]],  # coords
      [1],  # aatype
      [[1.0]],  # atom_mask
      [0],  # residue_index
      [0],  # chain_index
      None,  # dihedrals
    )

    with (
      patch("prxteinmpnn.run.load") as mock_load,
      patch("prxteinmpnn.run.batch_and_pad_proteins") as mock_batch,
      patch("prxteinmpnn.run.get_mpnn_model") as mock_model,
      patch("prxteinmpnn.run.make_score_sequence") as mock_score_fn,
      patch("prxteinmpnn.run.string_to_protein_sequence") as mock_str_to_seq,
    ):
      mock_load.return_value = ([mock_protein_tuple], ["test.pdb"])
      mock_str_to_seq.return_value = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

      mock_batched_ensemble = Mock(spec=ProteinEnsemble)
      # Configure mock attributes properly
      mock_aatype = Mock()
      mock_aatype.shape = (1, 10)
      mock_batched_ensemble.aatype = mock_aatype
      mock_batched_ensemble.coordinates = jnp.ones((1, 10, 37, 3))
      mock_batched_ensemble.atom_mask = jnp.ones((1, 10, 37))
      mock_batched_ensemble.residue_index = jnp.arange(10)[None, :]
      mock_batched_ensemble.chain_index = jnp.zeros((1, 10))
      mock_batched_ensemble.mapping = None

      mock_batched_sequences = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
      mock_batch.return_value = (mock_batched_ensemble, mock_batched_sequences)

      mock_model.return_value = {"params": {}}
      mock_score_single = Mock(return_value=(1.0, jnp.ones((10, 21)), {}))
      mock_score_fn.return_value = mock_score_single

      with patch("jax.lax.map") as mock_map:
        mock_map.return_value = (
          jnp.array([[1.0]]),  # scores
          jnp.ones((1, 1, 10, 21)),  # logits
          {},  # metadata
        )

        result = await score(
          inputs="test.pdb",
          sequences_to_score=["ACDEFGHIKL"],
          backbone_noise=0.1,
        )

        assert "scores" in result
        assert "logits" in result
        assert "mapping" in result
        assert "metadata" in result

        # Type assertions with proper casting
        scores = result["scores"]
        logits = result["logits"]
        metadata = result["metadata"]

        assert isinstance(scores, jax.Array) and scores.shape == (1, 1)
        assert isinstance(metadata, dict) and "protein_sources" in metadata

  @pytest.mark.asyncio
  async def test_score_multiple_noise_levels(self) -> None:
    """Test scoring with multiple backbone noise levels."""
    mock_protein_tuple = (
      [[[1.0, 2.0, 3.0]]],
      [1],
      [[1.0]],
      [0],
      [0],
      None,
    )

    with (
      patch("prxteinmpnn.run.load") as mock_load,
      patch("prxteinmpnn.run.batch_and_pad_proteins") as mock_batch,
      patch("prxteinmpnn.run.get_mpnn_model") as mock_model,
      patch("prxteinmpnn.run.make_score_sequence") as mock_score_fn,
      patch("prxteinmpnn.run.string_to_protein_sequence") as mock_str_to_seq,
    ):
      mock_load.return_value = ([mock_protein_tuple], ["test.pdb"])
      mock_str_to_seq.return_value = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

      mock_batched_ensemble = Mock(spec=ProteinEnsemble)
      # Configure mock attributes properly  
      mock_aatype = Mock()
      mock_aatype.shape = (1, 10)
      mock_batched_ensemble.aatype = mock_aatype
      mock_batched_ensemble.coordinates = jnp.ones((1, 10, 37, 3))
      mock_batched_ensemble.atom_mask = jnp.ones((1, 10, 37))
      mock_batched_ensemble.residue_index = jnp.arange(10)[None, :]
      mock_batched_ensemble.chain_index = jnp.zeros((1, 10))
      mock_batched_ensemble.mapping = None

      mock_batched_sequences = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
      mock_batch.return_value = (mock_batched_ensemble, mock_batched_sequences)

      mock_model.return_value = {"params": {}}
      mock_score_single = Mock(return_value=(1.0, jnp.ones((10, 21)), {}))
      mock_score_fn.return_value = mock_score_single

      with patch("jax.lax.map") as mock_map:
        mock_map.return_value = (
          jnp.array([[[1.0], [1.1]]]),  # scores for 2 noise levels
          jnp.ones((1, 2, 10, 21)),  # logits for 2 noise levels
          {},  # metadata
        )

        result = await score(
          inputs="test.pdb",
          sequences_to_score=["ACDEFGHIKL"],
          backbone_noise=[0.1, 0.2],
        )

        scores = result["scores"]
        logits = result["logits"]
        assert isinstance(scores, jax.Array) and scores.shape == (1, 2, 1) # batch dimension
        assert isinstance(logits, jax.Array) and logits.shape == (1, 2, 10, 21)


class TestSample:
  """Test the sample function."""

  @pytest.mark.asyncio
  async def test_sample_basic(self) -> None:
    """Test basic sampling functionality."""
    mock_protein_tuple = (
      [[[1.0, 2.0, 3.0]]],
      [1],
      [[1.0]],
      [0],
      [0],
      None,
    )

    with (
      patch("prxteinmpnn.run.load") as mock_load,
      patch("prxteinmpnn.run.batch_and_pad_proteins") as mock_batch,
      patch("prxteinmpnn.run.get_mpnn_model") as mock_model,
      patch("prxteinmpnn.run.make_sample_sequences") as mock_sample_fn,
    ):
      mock_load.return_value = ([mock_protein_tuple], ["test.pdb"])

      mock_batched_ensemble = Mock(spec=ProteinEnsemble)
      mock_batched_ensemble.coordinates = jnp.ones((1, 10, 37, 3))
      mock_batched_ensemble.atom_mask = jnp.ones((1, 10, 37))
      mock_batched_ensemble.residue_index = jnp.arange(10)[None, :]
      mock_batched_ensemble.chain_index = jnp.zeros((1, 10))
      mock_batched_ensemble.mapping = None

      mock_batch.return_value = (mock_batched_ensemble, None)

      mock_model.return_value = {"params": {}}
      mock_sampler = Mock(return_value=(jnp.ones((10,)), jnp.ones((10, 21)), {}))
      mock_sample_fn.return_value = mock_sampler

      with patch("jax.lax.map") as mock_map:
        mock_map.return_value = (
          jnp.ones((1, 1, 3, 10)),  # sampled sequences
          jnp.ones((1, 1, 3, 10, 21)),  # logits
          {},  # metadata
        )

        result = await sample(
          inputs="test.pdb",
          num_samples=3,
          backbone_noise=0.1,
        )

        assert "sampled_sequences" in result
        assert "logits" in result
        assert "mapping" in result
        assert "metadata" in result


class TestCategoricalJacobian:
  """Test the categorical_jacobian function."""

  @pytest.mark.asyncio
  async def test_categorical_jacobian_basic(self) -> None:
    """Test basic categorical Jacobian computation."""
    mock_protein_tuple = (
      [[[1.0, 2.0, 3.0]]],
      [1],
      [[1.0]],
      [0],
      [0],
      None,
    )

    with (
      patch("prxteinmpnn.run.load") as mock_load,
      patch("prxteinmpnn.run.batch_and_pad_proteins") as mock_batch,
      patch("prxteinmpnn.run.get_mpnn_model") as mock_model,
      patch("prxteinmpnn.run.make_conditional_logits_fn") as mock_logits_fn,
    ):
      mock_load.return_value = ([mock_protein_tuple], ["test.pdb"])

      mock_batched_ensemble = Mock(spec=ProteinEnsemble)
      mock_batched_ensemble.coordinates = jnp.ones((1, 10, 37, 3))
      mock_batched_ensemble.atom_mask = jnp.ones((1, 10, 37))
      mock_batched_ensemble.residue_index = jnp.arange(10)[None, :]
      mock_batched_ensemble.chain_index = jnp.zeros((1, 10))
      mock_batched_ensemble.one_hot_sequence = jnp.ones((1, 10, 21))
      mock_batched_ensemble.mapping = None

      mock_batch.return_value = (mock_batched_ensemble, None)

      mock_model.return_value = {"params": {}}
      mock_conditional_logits = Mock(return_value=(jnp.ones((10, 21)), {}, {}))
      mock_logits_fn.return_value = mock_conditional_logits

      with patch("jax.lax.map") as mock_map, patch("jax.jacfwd") as mock_jacfwd:
        mock_jacfwd.return_value = jnp.ones((10 * 21, 10 * 21))
        mock_map.return_value = jnp.ones((1, 1, 10, 21, 10, 21))  # Jacobians

        result = await categorical_jacobian(
          inputs="test.pdb",
          backbone_noise=0.1,
        )

        assert "categorical_jacobians" in result
        assert "cross_protein_diffs" in result
        assert "mapping" in result
        assert "metadata" in result

        jacobians = result["categorical_jacobians"]
        assert isinstance(jacobians, jax.Array) and jacobians.shape == (1, 1, 10, 21, 10, 21)

  @pytest.mark.asyncio
  async def test_categorical_jacobian_with_cross_diff(self) -> None:
    """Test categorical Jacobian computation with cross-protein differences."""
    mock_protein_tuple = (
      [[[1.0, 2.0, 3.0]]],
      [1],
      [[1.0]],
      [0],
      [0],
      None,
    )

    with (
      patch("prxteinmpnn.run.load") as mock_load,
      patch("prxteinmpnn.run.batch_and_pad_proteins") as mock_batch,
      patch("prxteinmpnn.run.get_mpnn_model") as mock_model,
      patch("prxteinmpnn.run.make_conditional_logits_fn") as mock_logits_fn,
      patch("prxteinmpnn.run.compute_cross_protein_jacobian_diffs") as mock_cross_diff,
    ):
      mock_load.return_value = ([mock_protein_tuple], ["test.pdb"])

      # Mock ensemble with mapping
      mock_mapping = jnp.ones((1, 10, 2))  # 1 pair, 10 positions, 2 indices
      mock_batched_ensemble = Mock(spec=ProteinEnsemble)
      mock_batched_ensemble.coordinates = jnp.ones((1, 10, 37, 3))
      mock_batched_ensemble.atom_mask = jnp.ones((1, 10, 37))
      mock_batched_ensemble.residue_index = jnp.arange(10)[None, :]
      mock_batched_ensemble.chain_index = jnp.zeros((1, 10))
      mock_batched_ensemble.one_hot_sequence = jnp.ones((1, 10, 21))
      mock_batched_ensemble.mapping = mock_mapping

      mock_batch.return_value = (mock_batched_ensemble, None)

      mock_model.return_value = {"params": {}}
      mock_conditional_logits = Mock(return_value=(jnp.ones((10, 21)), {}, {}))
      mock_logits_fn.return_value = mock_conditional_logits

      mock_cross_diff.return_value = jnp.ones((1, 1, 10, 21, 10, 21))

      with patch("jax.lax.map") as mock_map, patch("jax.jacfwd") as mock_jacfwd:
        mock_jacfwd.return_value = jnp.ones((10 * 21, 10 * 21))
        mock_map.return_value = jnp.ones((1, 1, 10, 21, 10, 21))  # Jacobians

        result = await categorical_jacobian(
          inputs="test.pdb",
          backbone_noise=0.1,
          calculate_cross_diff=True,
        )

        assert "categorical_jacobians" in result
        assert "cross_protein_diffs" in result
        assert result["cross_protein_diffs"] is not None
        assert "mapping" in result
        assert result["mapping"] is not None
        mock_cross_diff.assert_called_once()


class TestComputeCrossProteinJacobianDiffs:
  """Test the compute_cross_protein_jacobian_diffs function."""

  def test_empty_mapping(self) -> None:
    """Test with empty mapping array."""
    jacobians = jnp.ones((2, 3, 5, 21, 5, 21))  # batch_size=2, noise_levels=3, L=5
    mapping = jnp.empty((0, 5, 2))  # Empty mapping

    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    assert result.shape == (0, 3, 5, 21, 5, 21)
    assert result.size == 0

  def test_single_pair_perfect_alignment(self) -> None:
    """Test with a single protein pair with perfect alignment."""
    # Create test Jacobians with different values for each protein
    jacobians = jnp.array([
      jnp.ones((3, 4, 21, 4, 21)) * 1.0,  # Protein 0
      jnp.ones((3, 4, 21, 4, 21)) * 2.0,  # Protein 1
    ])  # Shape: (2, 3, 4, 21, 4, 21)
    
    # Perfect alignment mapping: positions [0,1,2,3] in both proteins
    mapping = jnp.array([
      [[0, 0], [1, 1], [2, 2], [3, 3]]  # One pair, perfect alignment
    ])  # Shape: (1, 4, 2)
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    assert result.shape == (1, 3, 4, 21, 4, 21)
    
    # Check that differences are computed correctly (1.0 - 2.0 = -1.0)
    expected_diff = -1.0
    assert jnp.allclose(result[0, 0, 0, 0, 0, 0], expected_diff)
    assert jnp.allclose(result[0, 1, 2, 10, 3, 15], expected_diff)

  def test_partial_alignment_with_gaps(self) -> None:
    """Test with partial alignment containing gaps."""
    jacobians = jnp.array([
      jnp.ones((2, 5, 21, 5, 21)) * 3.0,  # Protein 0
      jnp.ones((2, 5, 21, 5, 21)) * 7.0,  # Protein 1
    ])  # Shape: (2, 2, 5, 21, 5, 21)
    
    # Partial alignment with gaps (-1 indicates no alignment)
    mapping = jnp.array([
      [[0, 0], [-1, -1], [2, 1], [3, 3], [-1, -1]]  # Positions 0,2,3 aligned
    ])  # Shape: (1, 5, 2)
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    assert result.shape == (1, 2, 5, 21, 5, 21)
    
    # Check aligned positions have correct differences (3.0 - 7.0 = -4.0)
    expected_diff = -4.0
    assert jnp.allclose(result[0, 0, 0, 0, 0, 0], expected_diff)  # Position 0
    assert jnp.allclose(result[0, 1, 2, 5, 3, 10], expected_diff)  # Position 2
    assert jnp.allclose(result[0, 0, 3, 15, 2, 8], expected_diff)  # Position 3
    
    # Check non-aligned positions are NaN
    assert jnp.isnan(result[0, 0, 1, 0, 0, 0])  # Position 1 (gap)
    assert jnp.isnan(result[0, 1, 4, 10, 1, 5])  # Position 4 (gap)

  def test_multiple_protein_pairs(self) -> None:
    """Test with multiple protein pairs."""
    jacobians = jnp.array([
      jnp.ones((1, 3, 21, 3, 21)) * 1.0,  # Protein 0
      jnp.ones((1, 3, 21, 3, 21)) * 2.0,  # Protein 1
      jnp.ones((1, 3, 21, 3, 21)) * 3.0,  # Protein 2
    ])  # Shape: (3, 1, 3, 21, 3, 21)
    
    # Two pairs: (0,1) and (0,2)
    mapping = jnp.array([
      [[0, 0], [1, 1], [-1, -1]],  # Pair (0,1): positions 0,1 aligned
      [[0, 0], [-1, -1], [2, 2]],  # Pair (0,2): positions 0,2 aligned
      [[-1, -1], [1, 1], [2, 2]],  # Pair (1,2): positions 1,2 aligned
    ])  # Shape: (2, 3, 2)
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    assert result.shape == (3, 1, 3, 21, 3, 21)
    
    # Check pair (0,1): 1.0 - 2.0 = -1.0
    assert jnp.allclose(result[0, 0, 0, 5, 1, 10], -1.0)
    assert jnp.allclose(result[0, 0, 1, 0, 0, 0], -1.0)
    assert jnp.isnan(result[0, 0, 2, 0, 0, 0])  # Position 2 not aligned
    
    # Check pair (0,2): 1.0 - 3.0 = -2.0
    assert jnp.allclose(result[1, 0, 0, 0, 2, 15], -2.0)
    assert jnp.isnan(result[1, 0, 1, 5, 0, 5])  # Position 1 not aligned
    assert jnp.allclose(result[1, 2, 10, 0, 0], -2.0)

  def test_invalid_protein_indices(self) -> None:
    """Test with mapping containing invalid protein indices."""
    jacobians = jnp.ones((2, 1, 3, 21, 3, 21))  # Only 2 proteins
    
    # Mapping with invalid protein index (protein 5 doesn't exist)
    mapping = jnp.array([
      [[0, 5], [1, 1], [-1, -1]]  # First position maps to invalid protein 5
    ])  # Shape: (1, 3, 2)
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    assert result.shape == (1, 1, 3, 21, 3, 21)
    
    # All positions should be NaN due to invalid protein index
    assert jnp.isnan(result[0, 0, 0, 0, 0, 0])
    assert jnp.isnan(result[0, 0, 1, 10, 2, 15])
    assert jnp.isnan(result[0, 0, 2, 5, 1, 8])

  def test_multiple_noise_levels(self) -> None:
    """Test correct handling of multiple noise levels."""
    # Different values for each noise level
    jacobians = jnp.array([
      jnp.stack([  # Protein 0
        jnp.ones((2, 21, 2, 21)) * 1.0,  # Noise level 0
        jnp.ones((2, 21, 2, 21)) * 1.5,  # Noise level 1
        jnp.ones((2, 21, 2, 21)) * 2.0,  # Noise level 2
      ]),
      jnp.stack([  # Protein 1
        jnp.ones((2, 21, 2, 21)) * 3.0,  # Noise level 0
        jnp.ones((2, 21, 2, 21)) * 4.0,  # Noise level 1
        jnp.ones((2, 21, 2, 21)) * 5.0,  # Noise level 2
      ]),
    ])  # Shape: (2, 3, 2, 21, 2, 21)
    
    mapping = jnp.array([
      [[0, 0], [1, 1]]  # Perfect alignment
    ])  # Shape: (1, 2, 2)
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    assert result.shape == (1, 3, 2, 21, 2, 21)
    
    # Check differences for each noise level
    assert jnp.allclose(result[0, 0, 0, 0, 0, 0], 1.0 - 3.0)  # -2.0
    assert jnp.allclose(result[0, 1, 1, 5, 1, 10], 1.5 - 4.0)  # -2.5
    assert jnp.allclose(result[0, 2, 0, 15, 1, 20], 2.0 - 5.0)  # -3.0

  def test_edge_case_single_position(self) -> None:
    """Test edge case with single position proteins."""
    jacobians = jnp.array([
      jnp.ones((1, 1, 21, 1, 21)) * 10.0,  # Protein 0
      jnp.ones((1, 1, 21, 1, 21)) * 5.0,   # Protein 1
    ])  # Shape: (2, 1, 1, 21, 1, 21)
    
    mapping = jnp.array([
      [[0, 0]]  # Single position alignment
    ])  # Shape: (1, 1, 2)
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)

    assert result.shape == (1, 1, 1, 21, 1, 21)
    assert jnp.allclose(result[0, 0, 0, 0, 0, 0], 10.0 - 5.0)  # 5.0

  def test_asymmetric_jacobian_values(self) -> None:
    """Test with asymmetric Jacobian values to ensure proper indexing."""
    # Create asymmetric Jacobians where J[pos_i, aa_i, pos_j, aa_j] has unique values
    jacobians = jnp.zeros((2, 1, 2, 21, 2, 21))
    
    # Protein 0: set specific positions to unique values
    jacobians = jacobians.at[0, 0, 0, 5, 1, 10].set(100.0)
    jacobians = jacobians.at[0, 0, 1, 3, 0, 7].set(200.0)
    
    # Protein 1: set same positions to different values
    jacobians = jacobians.at[1, 0, 0, 5, 1, 10].set(150.0)
    jacobians = jacobians.at[1, 0, 1, 3, 0, 7].set(250.0)
    
    mapping = jnp.array([
      [[0, 0], [1, 1]]  # Perfect alignment
    ])
    
    result = compute_cross_protein_jacobian_diffs(jacobians, mapping)
    
    # Check specific positions
    assert jnp.allclose(result[0, 0, 0, 5, 1, 10], 100.0 - 150.0)  # -50.0
    assert jnp.allclose(result[0, 0, 1, 3, 0, 7], 200.0 - 250.0)   # -50.0

    # Check that other positions are 0 (since base Jacobians are 0)
    assert jnp.allclose(result[0, 0, 0, 0, 0, 0], 0.0)
