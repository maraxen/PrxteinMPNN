"""Parity-focused tests for conditional logits helpers."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import pearsonr

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling.conditional_logits import (
  make_conditional_logits_fn,
  make_encoding_conditional_logits_split_fn,
)


def _synthetic_conditional_inputs(
  num_residues: int = 12,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """Build deterministic conditional inputs."""
  rng = np.random.default_rng(1)
  coordinates = jnp.array(rng.normal(size=(num_residues, 37, 3)).astype(np.float32))
  mask = jnp.ones((num_residues,), dtype=jnp.float32)
  residue_index = jnp.arange(num_residues, dtype=jnp.int32)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)
  sequence_tokens = jnp.array(rng.integers(0, 20, size=(num_residues,), dtype=np.int32))
  sequence_one_hot = jax.nn.one_hot(sequence_tokens, 21)
  return coordinates, mask, residue_index, chain_index, sequence_one_hot, sequence_tokens


@pytest.mark.parity_fast
def test_conditional_logits_helper_matches_model_branch() -> None:
  """Validate helper-vs-branch parity for conditional logits."""
  model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=10,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(2),
  )
  model = eqx.tree_inference(model, value=True)
  conditional_helper = make_conditional_logits_fn(model)

  coordinates, mask, residue_index, chain_index, sequence_one_hot, _ = _synthetic_conditional_inputs()
  ar_mask = jnp.tril(jnp.ones((mask.shape[0], mask.shape[0]), dtype=jnp.int32), k=-1)
  key = jax.random.PRNGKey(9)

  helper_logits = conditional_helper(
    key,
    coordinates,
    mask,
    residue_index,
    chain_index,
    sequence_one_hot,
    ar_mask=ar_mask,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  _, direct_logits = model(
    coordinates,
    mask,
    residue_index,
    chain_index,
    decoding_approach="conditional",
    prng_key=key,
    ar_mask=ar_mask,
    one_hot_sequence=sequence_one_hot,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )

  helper_logits_np = np.asarray(helper_logits)
  direct_logits_np = np.asarray(direct_logits)
  np.testing.assert_allclose(helper_logits_np, direct_logits_np, rtol=1e-6, atol=1e-6)
  corr = float(pearsonr(helper_logits_np.ravel(), direct_logits_np.ravel())[0])
  assert corr >= 0.999999


@pytest.mark.parity_fast
def test_split_conditional_logits_matches_full_helper() -> None:
  """Validate split encode/decode helper parity against the full conditional helper."""
  model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=10,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(3),
  )
  model = eqx.tree_inference(model, value=True)
  conditional_helper = make_conditional_logits_fn(model)
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)

  coordinates, mask, residue_index, chain_index, sequence_one_hot, _ = _synthetic_conditional_inputs()
  ar_mask = jnp.tril(jnp.ones((mask.shape[0], mask.shape[0]), dtype=jnp.int32), k=-1)
  key = jax.random.PRNGKey(5)

  encoding = encode_fn(
    coordinates,
    mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
    prng_key=key,
  )
  split_logits = decode_fn(encoding, sequence_one_hot, ar_mask=ar_mask)
  full_logits = conditional_helper(
    key,
    coordinates,
    mask,
    residue_index,
    chain_index,
    sequence_one_hot,
    ar_mask=ar_mask,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )

  split_logits_np = np.asarray(split_logits)
  full_logits_np = np.asarray(full_logits)
  np.testing.assert_allclose(split_logits_np, full_logits_np, rtol=1e-6, atol=1e-6)
  corr = float(pearsonr(split_logits_np.ravel(), full_logits_np.ravel())[0])
  assert corr >= 0.999999
