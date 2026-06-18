"""Tests for RS-7a: temperature invariance and HDF5 deferral in scoring."""
# type: ignore[arg-type, call-arg]
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from aminx.model import Aminx
from aminx.scoring.score import make_score_sequence
from aminx.run.specs import ScoringSpecification


@pytest.fixture
def mock_model() -> Aminx:
  """Fixture for a mock Aminx model."""
  model_mock = MagicMock(spec=Aminx)

  def mock_features(
      key,
      coords,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      backbone_noise_mode=None,
      structure_mapping=None,
      **kwargs
  ):
    del (
        key,
        coords,
        mask,
        residue_index,
        chain_index,
        backbone_noise,
        backbone_noise_mode,
        structure_mapping,
        kwargs
    )
    n_residues = 76
    edge_f = jnp.zeros((n_residues, 48, 128))
    edge_i = jnp.zeros((n_residues, 48), dtype=jnp.int32)
    node_f = jnp.zeros((n_residues, 128))
    padding = jnp.zeros((n_residues,))
    return edge_f, edge_i, node_f, padding

  model_mock.features = MagicMock(side_effect=mock_features)

  # Mock encoder
  model_mock.encoder = MagicMock()
  model_mock.encoder.side_effect = lambda ef, ei, m, initial_node_features=None, key=None: (
      initial_node_features if initial_node_features is not None else jnp.zeros((76, 128)),
      ef
  )

  # Mock __call__ for per-state array interface (composability contract)
  def mock_call(coords, mask, residue_index, chain_index, **kwargs):
    """Return (node_features, edge_features, neighbor_indices)"""
    n_residues = 76
    node_f = jnp.zeros((n_residues, 128))
    edge_f = jnp.zeros((n_residues, 48, 128))
    edge_i = jnp.zeros((n_residues, 48), dtype=jnp.int32)
    return node_f, edge_f, edge_i

  model_mock.side_effect = mock_call

  # Mock decoder
  model_mock.decoder = MagicMock()
  model_mock.decoder.call_conditional.side_effect = lambda nb, eb, i, m, a, oh, w, **kw: jnp.zeros((nb.shape[0], 128))

  # Mock projections
  model_mock.w_out = MagicMock()
  model_mock.w_out.side_effect = lambda x: jnp.zeros(21)

  model_mock.w_s_embed = MagicMock()
  model_mock.w_s_embed.weight = jnp.zeros((21, 128))

  return model_mock


def test_score_temperature_invariant(mock_model: Aminx, protein_structure) -> None:
  """Test that score_sequence returns same NLL for different multi_state_temperature values."""
  # Arrange: Create score function and prepare inputs
  with patch(
      "aminx.scoring.score.jax.jit", new=lambda fn, *args, **kwargs: fn,
  ):
    score_fn = make_score_sequence(mock_model)

  prng_key = jax.random.key(42)
  coords = protein_structure.coordinates
  mask = protein_structure.mask
  residue_index = protein_structure.residue_index
  chain_index = protein_structure.chain_index
  sequence = protein_structure.aatype

  # Act: Score with temperature=1.0
  nll_temp_1, _, _ = score_fn(
      prng_key,
      sequence,
      coords,
      mask,
      residue_index,
      chain_index,
      _k_neighbors=48,
      multi_state_temperature=1.0,
  )

  # Act: Score with temperature=10.0
  nll_temp_10, _, _ = score_fn(
      prng_key,
      sequence,
      coords,
      mask,
      residue_index,
      chain_index,
      _k_neighbors=48,
      multi_state_temperature=10.0,
  )

  # Assert: NLL arrays are equal (temperature does not affect scoring)
  assert jnp.array_equal(nll_temp_1, nll_temp_10), (
      f"Expected NLL to be invariant to temperature; got {nll_temp_1} for temp=1.0 "
      f"and {nll_temp_10} for temp=10.0"
  )


def test_score_hdf5_output_not_implemented() -> None:
  """Test that score() raises NotImplementedError when output_h5_path is set."""
  # Arrange: Create ScoringSpecification with output_h5_path set
  spec = ScoringSpecification(
      inputs=["dummy.pdb"],
      sequences_to_score=["AAAA"],
      output_h5_path="/tmp/output.h5",
  )

  # Act & Assert: Verify that score() raises NotImplementedError
  from aminx.host.runner import score

  with pytest.raises(NotImplementedError, match="HDF5 streaming output not yet implemented"):
    score(spec)
