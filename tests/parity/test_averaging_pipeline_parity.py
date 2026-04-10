"""Parity checks for averaged encoding and run-level API contracts."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.run.averaging import get_averaged_encodings
from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.scoring import score
from prxteinmpnn.run.specs import SamplingSpecification, ScoringSpecification
from prxteinmpnn.utils.data_structures import Protein

pytestmark = pytest.mark.parity_fast


@pytest.fixture
def averaged_contract_ensemble() -> Protein:
  """Build a deterministic two-structure ensemble for averaging-contract tests."""
  seq_len = 4
  num_structures = 2
  coordinates = jnp.stack(
    [
      jnp.ones((seq_len, 4, 3), dtype=jnp.float32) * 1.0,
      jnp.ones((seq_len, 4, 3), dtype=jnp.float32) * 2.0,
    ],
    axis=0,
  )
  residue_index = jnp.tile(jnp.arange(seq_len, dtype=jnp.int32), (num_structures, 1))
  chain_index = jnp.zeros((num_structures, seq_len), dtype=jnp.int32)
  aatype = jnp.zeros((num_structures, seq_len), dtype=jnp.int8)
  return Protein(
    coordinates=coordinates,
    mask=jnp.ones((num_structures, seq_len), dtype=jnp.float32),
    residue_index=residue_index,
    chain_index=chain_index,
    aatype=aatype,
    one_hot_sequence=jax.nn.one_hot(aatype, 21),
  )


@pytest.fixture
def mock_protein() -> Protein:
  """Build a deterministic single-structure batch for run API parity checks."""
  seq_len = 10
  aatype = jnp.zeros((1, seq_len), dtype=jnp.int8)
  return Protein(
    coordinates=jnp.ones((1, seq_len, 4, 3), dtype=jnp.float32),
    mask=jnp.ones((1, seq_len), dtype=jnp.float32),
    residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :],
    chain_index=jnp.zeros((1, seq_len), dtype=jnp.int32),
    aatype=aatype,
    one_hot_sequence=jax.nn.one_hot(aatype, 21),
  )


@pytest.fixture
def tiny_model() -> PrxteinMPNN:
  """Create a compact inference-only model for fast run API parity checks."""
  model = PrxteinMPNN(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=5,
    key=jax.random.key(0),
  )
  return eqx.tree_inference(model, value=True)


def _fake_make_encoding_sampling_split_fn(_model: PrxteinMPNN | None) -> tuple[Callable, Callable, Callable]:
  """Return deterministic encoded tensors for averaging mode parity checks."""

  def encode_fn(
    _prng_key: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    _residue_index: jax.Array,
    _chain_index: jax.Array,
    k_neighbors: int = 48,
    backbone_noise: float | jax.Array | None = None,
  ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    del _prng_key, _residue_index, _chain_index, k_neighbors
    noise = jnp.asarray(0.0 if backbone_noise is None else backbone_noise, dtype=jnp.float32)
    structure_scale = structure_coordinates[0, 0, 0]
    seq_len = structure_coordinates.shape[0]
    node = jnp.ones((seq_len, 2), dtype=jnp.float32) * (structure_scale * 10.0 + noise)
    edge = jnp.ones((seq_len, 3, 4), dtype=jnp.float32) * (structure_scale * 100.0 + noise)
    neighbors = jnp.tile(jnp.arange(3, dtype=jnp.int32), (seq_len, 1))
    ar_mask = jnp.eye(seq_len, dtype=jnp.int32)
    return node, edge, neighbors, mask, ar_mask

  def _unused(*_args: object, **_kwargs: object) -> object:
    msg = "Unused in averaged encoding parity tests."
    raise RuntimeError(msg)

  return encode_fn, _unused, _unused


def _expected_node_and_edge(mode: str) -> tuple[jax.Array, jax.Array]:
  """Return expected deterministic averages for each averaged mode."""
  node_values = {
    "inputs": jnp.asarray([15.0, 16.0], dtype=jnp.float32),
    "noise_levels": jnp.asarray([10.5, 20.5], dtype=jnp.float32),
    "inputs_and_noise": jnp.asarray([15.5], dtype=jnp.float32),
  }
  edge_values = {
    "inputs": jnp.asarray([150.0, 151.0], dtype=jnp.float32),
    "noise_levels": jnp.asarray([100.5, 200.5], dtype=jnp.float32),
    "inputs_and_noise": jnp.asarray([150.5], dtype=jnp.float32),
  }
  if mode == "inputs_and_noise":
    expected_node = jnp.broadcast_to(node_values[mode], (4, 2))
    expected_edge = jnp.broadcast_to(edge_values[mode], (4, 3, 4))
    return expected_node, expected_edge
  expected_node = jnp.broadcast_to(node_values[mode][:, None, None], (2, 4, 2))
  expected_edge = jnp.broadcast_to(edge_values[mode][:, None, None, None], (2, 4, 3, 4))
  return expected_node, expected_edge


def _require(*, condition: bool, message: str) -> None:
  """Raise an AssertionError when a deterministic contract is violated."""
  if not condition:
    raise AssertionError(message)


@pytest.mark.parametrize("average_mode", ["inputs", "noise_levels", "inputs_and_noise"])
def test_averaged_encoding_contracts_by_mode(
  averaged_contract_ensemble: Protein,
  average_mode: str,
) -> None:
  """Validate deterministic averaged-encoding parity contracts for all supported modes."""
  with patch(
    "prxteinmpnn.run.averaging.make_encoding_sampling_split_fn",
    new=_fake_make_encoding_sampling_split_fn,
  ):
    observed = get_averaged_encodings(
      averaged_contract_ensemble,
      model=None,
      backbone_noise=(0.0, 1.0),
      noise_batch_size=2,
      random_seed=7,
      average_encoding_mode=average_mode,
    )
    repeated = get_averaged_encodings(
      averaged_contract_ensemble,
      model=None,
      backbone_noise=(0.0, 1.0),
      noise_batch_size=2,
      random_seed=7,
      average_encoding_mode=average_mode,
    )

  expected_node, expected_edge = _expected_node_and_edge(average_mode)
  avg_node, avg_edge, neighbors, mask, ar_mask = observed

  chex.assert_trees_all_equal(observed, repeated)
  chex.assert_trees_all_close(avg_node, expected_node)
  chex.assert_trees_all_close(avg_edge, expected_edge)
  chex.assert_shape(neighbors, (2, 2, 4, 3))
  chex.assert_shape(mask, (2, 2, 4))
  chex.assert_shape(ar_mask, (2, 2, 4, 4))


@pytest.mark.parametrize(
  ("average_mode", "expected_flat_samples"),
  [("inputs", 4), ("noise_levels", 2), ("inputs_and_noise", 2)],
)
def test_sample_api_averaged_contracts(
  mock_protein: Protein,
  tiny_model: PrxteinMPNN,
  average_mode: str,
  expected_flat_samples: int,
) -> None:
  """Validate averaged sampling contracts, determinism, and pseudo-perplexity invariants."""
  spec = SamplingSpecification(
    inputs=["dummy.pdb"],
    num_samples=2,
    backbone_noise=[0.0, 0.2],
    average_node_features=True,
    average_encoding_mode=average_mode,
    temperature=[0.1, 1.0],
    compute_pseudo_perplexity=True,
    random_seed=11,
  )

  with patch("prxteinmpnn.run.sampling.prep_protein_stream_and_model", return_value=([mock_protein], tiny_model)):
    first = sample(spec)
    second = sample(spec)

  chex.assert_shape(first["sequences"], (1, expected_flat_samples, 2, 10))
  chex.assert_shape(first["logits"], (1, expected_flat_samples, 2, 10, 21))
  chex.assert_shape(first["pseudo_perplexity"], (1, expected_flat_samples, 2))
  chex.assert_tree_all_finite((first["sequences"], first["logits"], first["pseudo_perplexity"]))
  _require(
    condition=bool(jnp.all(first["pseudo_perplexity"] > 0).item()),
    message="Pseudo-perplexity must remain strictly positive.",
  )
  _require(
    condition=first["logits"].shape[:-1] == first["sequences"].shape,
    message="Logits contract must match sampled sequence shape plus amino-acid axis.",
  )
  chex.assert_trees_all_equal(first["sequences"], second["sequences"])
  chex.assert_trees_all_close(first["logits"], second["logits"])
  chex.assert_trees_all_close(first["pseudo_perplexity"], second["pseudo_perplexity"])


@pytest.mark.parametrize(
  ("average_mode", "expected_outer"),
  [("inputs", 2), ("noise_levels", 1), ("inputs_and_noise", 1)],
)
def test_score_api_averaged_contracts(
  mock_protein: Protein,
  tiny_model: PrxteinMPNN,
  average_mode: str,
  expected_outer: int,
) -> None:
  """Validate averaged scoring contracts and deterministic outputs by averaged mode."""
  spec = ScoringSpecification(
    inputs=["dummy.pdb"],
    sequences_to_score=["G" * 10, "A" * 10],
    backbone_noise=[0.0, 0.2],
    average_node_features=True,
    average_encoding_mode=average_mode,
    random_seed=5,
  )

  with patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model", return_value=([mock_protein], tiny_model)):
    first = score(spec)
    second = score(spec)

  chex.assert_shape(first["scores"], (expected_outer, 2))
  chex.assert_shape(first["logits"], (expected_outer, 2, 10, 21))
  chex.assert_tree_all_finite((first["scores"], first["logits"]))
  _require(
    condition=first["logits"].shape[:-2] == first["scores"].shape,
    message="Scoring logits contract must match score shape plus residue and amino-acid axes.",
  )
  chex.assert_trees_all_close(first["scores"], second["scores"])
  chex.assert_trees_all_close(first["logits"], second["logits"])
