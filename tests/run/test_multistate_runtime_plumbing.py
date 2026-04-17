"""Runtime plumbing tests for multi-state controls in sampling and scoring."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp

from prxteinmpnn.run.sampling import SamplingSpecification, sample
from prxteinmpnn.run.scoring import score
from prxteinmpnn.run.specs import ScoringSpecification
from prxteinmpnn.utils.data_structures import Protein


def _mock_batched_protein() -> Protein:
  seq_len = 6
  return Protein(
    coordinates=jnp.ones((1, seq_len, 4, 3), dtype=jnp.float32),
    aatype=jnp.ones((1, seq_len), dtype=jnp.int8),
    one_hot_sequence=jax.nn.one_hot(jnp.ones((1, seq_len), dtype=jnp.int8), 21),
    mask=jnp.ones((1, seq_len), dtype=jnp.float32),
    residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :],
    chain_index=jnp.zeros((1, seq_len), dtype=jnp.int32),
    mapping=jnp.asarray([[0, 0, 0, 1, 1, 1]], dtype=jnp.int32),
  )


def test_sampling_multistate_controls_are_forwarded() -> None:
  protein = _mock_batched_protein()
  tie_group_map = jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
  structure_mapping = jnp.asarray([1, 1, 1, 0, 0, 0], dtype=jnp.int32)

  def fake_sampler(  # noqa: PLR0913
    _key: jax.Array,
    structure_coordinates: jax.Array,
    _mask: jax.Array,
    _residue_index: jax.Array,
    _chain_index: jax.Array,
    *,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_temperature: float = 1.0,
    **_kwargs: object,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    seq_len = structure_coordinates.shape[0]
    sequence = (
      tie_group_map.astype(jnp.int8) if tie_group_map is not None else jnp.zeros((seq_len,), dtype=jnp.int8)
    )
    logits = jnp.zeros((seq_len, 21), dtype=jnp.float32)
    if structure_mapping is not None:
      logits = logits.at[:, 0].set(structure_mapping.astype(jnp.float32))
    logits = logits.at[:, 1].set(jnp.asarray(multi_state_temperature, dtype=jnp.float32))
    return sequence, logits, jnp.arange(seq_len, dtype=jnp.int32)

  with patch(
    "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
    return_value=([protein], object()),
  ):
    with patch("prxteinmpnn.run.sampling.make_sample_sequences", return_value=fake_sampler):
      spec = SamplingSpecification(
        inputs=["dummy.pdb"],
        num_samples=1,
        backbone_noise=[0.0],
        temperature=[1.0],
        tie_group_map=tie_group_map,
        structure_mapping=structure_mapping,
        multi_state_strategy="product",
        multi_state_temperature=2.5,
      )
      result = sample(spec)

  sampled_sequence = result["sequences"][0, 0, 0, 0]
  sampled_mapping = result["logits"][0, 0, 0, 0, :, 0]
  sampled_temperature = result["logits"][0, 0, 0, 0, :, 1]

  assert jnp.array_equal(sampled_sequence, tie_group_map.astype(jnp.int8))
  assert jnp.array_equal(sampled_mapping, structure_mapping.astype(jnp.float32))
  assert jnp.allclose(sampled_temperature, 2.5)


def test_scoring_multistate_controls_are_forwarded() -> None:
  protein = _mock_batched_protein()
  tie_group_map = jnp.asarray([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
  structure_mapping = jnp.asarray([1, 1, 1, 0, 0, 0], dtype=jnp.int32)

  def fake_score_fn(  # noqa: PLR0913
    _key: jax.Array,
    _sequence: jax.Array,
    structure_coordinates: jax.Array,
    _mask: jax.Array,
    _residue_index: jax.Array,
    _chain_index: jax.Array,
    _backbone_noise: jax.Array | None = None,
    _ar_mask: jax.Array | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: str = "arithmetic_mean",
    multi_state_temperature: float = 1.0,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    del multi_state_strategy
    seq_len = structure_coordinates.shape[0]
    logits = jnp.zeros((seq_len, 21), dtype=jnp.float32)
    if structure_mapping is not None:
      logits = logits.at[:, 0].set(structure_mapping.astype(jnp.float32))
    if tie_group_map is not None:
      logits = logits.at[:, 1].set(tie_group_map.astype(jnp.float32))
    logits = logits.at[:, 2].set(jnp.asarray(multi_state_temperature, dtype=jnp.float32))
    score_value = jnp.asarray(multi_state_temperature, dtype=jnp.float32)
    return score_value, logits, jnp.arange(seq_len, dtype=jnp.int32)

  with patch(
    "prxteinmpnn.run.scoring.prep_protein_stream_and_model",
    return_value=([protein], object()),
  ):
    with patch("prxteinmpnn.run.scoring.make_score_fn", return_value=fake_score_fn):
      spec = ScoringSpecification(
        inputs=["dummy.pdb"],
        sequences_to_score=["AAAAAA"],
        backbone_noise=[0.0],
        tie_group_map=tie_group_map,
        structure_mapping=structure_mapping,
        multi_state_strategy="geometric_mean",
        multi_state_temperature=3.0,
      )
      result = score(spec)

  scored_logits = result["logits"][0, 0, 0]
  assert jnp.allclose(result["scores"], 3.0)
  assert jnp.array_equal(scored_logits[:, 0], structure_mapping.astype(jnp.float32))
  assert jnp.array_equal(scored_logits[:, 1], tie_group_map.astype(jnp.float32))
  assert jnp.allclose(scored_logits[:, 2], 3.0)


def test_scoring_defaults_use_dataset_structure_mapping() -> None:
  protein = _mock_batched_protein()

  def fake_score_fn(  # noqa: PLR0913
    _key: jax.Array,
    _sequence: jax.Array,
    structure_coordinates: jax.Array,
    _mask: jax.Array,
    _residue_index: jax.Array,
    _chain_index: jax.Array,
    _backbone_noise: jax.Array | None = None,
    _ar_mask: jax.Array | None = None,
    structure_mapping: jax.Array | None = None,
    tie_group_map: jax.Array | None = None,
    multi_state_strategy: str = "arithmetic_mean",
    multi_state_temperature: float = 1.0,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    del tie_group_map, multi_state_strategy, multi_state_temperature
    seq_len = structure_coordinates.shape[0]
    logits = jnp.zeros((seq_len, 21), dtype=jnp.float32)
    if structure_mapping is not None:
      logits = logits.at[:, 0].set(structure_mapping.astype(jnp.float32))
    return jnp.asarray(0.0, dtype=jnp.float32), logits, jnp.arange(seq_len, dtype=jnp.int32)

  with patch(
    "prxteinmpnn.run.scoring.prep_protein_stream_and_model",
    return_value=([protein], object()),
  ):
    with patch("prxteinmpnn.run.scoring.make_score_fn", return_value=fake_score_fn):
      spec = ScoringSpecification(
        inputs=["dummy.pdb"],
        sequences_to_score=["AAAAAA"],
      )
      result = score(spec)

  scored_logits = result["logits"][0, 0, 0]
  assert jnp.array_equal(scored_logits[:, 0], protein.mapping[0].astype(jnp.float32))
