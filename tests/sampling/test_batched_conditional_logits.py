"""Parity + PRNG-key invariant tests for the batched conditional-logits split fn.

Covers Part 1 of the aminx batched conditional-logits plan (tev_design necklace
campaign, task 260629_necklace-library-campaign): batched encode-over-replicates x
decode-over-candidates must match the unbatched, Python-loop path elementwise, and
the PRNG key must actually drive backbone noise (falsifies the old "prng_key unused"
docstring at conditional_logits.py).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model import Aminx
from aminx.sampling.conditional_logits import (
  make_batched_conditional_logits_fn,
  make_batched_conditional_logits_split_fn,
  make_conditional_logits_fn,
)


def _synthetic_structure(
  num_residues: int = 12,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  rng = np.random.default_rng(1)
  coordinates = jnp.array(rng.normal(size=(num_residues, 37, 3)).astype(np.float32))
  mask = jnp.ones((num_residues,), dtype=jnp.float32)
  residue_index = jnp.arange(num_residues, dtype=jnp.int32)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)
  return coordinates, mask, residue_index, chain_index


def _synthetic_candidates(
  num_candidates: int, num_residues: int, seed: int = 7,
) -> jax.Array:
  rng = np.random.default_rng(seed)
  return jnp.array(
    rng.integers(0, 20, size=(num_candidates, num_residues), dtype=np.int32),
  )


def _make_model(key_seed: int) -> Aminx:
  model = Aminx(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=10,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(key_seed),
  )
  return eqx.tree_inference(model, value=True)


@pytest.mark.parity_fast
def test_batched_split_fn_matches_python_loop_at_bb_zero() -> None:
  """(a) PARITY — batched [r, c] == Python-loop make_conditional_logits_fn elementwise."""
  model = _make_model(2)
  coordinates, mask, residue_index, chain_index = _synthetic_structure()
  num_residues = coordinates.shape[0]
  candidates = _synthetic_candidates(num_candidates=5, num_residues=num_residues)

  n_replicates = 3
  replicate_keys = jax.random.split(jax.random.PRNGKey(11), n_replicates)
  # Explicit shared ar_mask: make_conditional_logits_fn's build_inference_bundle default
  # (1 - eye(L), bundle_builder.py:127) differs from decode_fn's own None-default
  # (zeros(L, L)) — pass the same mask to both paths to isolate batching parity from
  # that pre-existing default-mask divergence.
  ar_mask = 1 - jnp.eye(num_residues, dtype=jnp.int32)

  batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(model)
  encodings = batched_encode_fn(
    coordinates,
    mask,
    residue_index,
    chain_index,
    replicate_keys,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  batched_logits = batched_decode_fn(encodings, candidates, ar_mask=ar_mask)  # (R, C, L, 21)
  assert batched_logits.shape == (n_replicates, candidates.shape[0], num_residues, 21)

  conditional_helper = make_conditional_logits_fn(model)
  for r, rkey in enumerate(replicate_keys):
    for c in range(candidates.shape[0]):
      expected = conditional_helper(
        rkey,
        coordinates,
        mask,
        residue_index,
        chain_index,
        candidates[c],
        ar_mask=ar_mask,
        backbone_noise=jnp.array(0.0, dtype=jnp.float32),
      )
      np.testing.assert_allclose(
        np.asarray(batched_logits[r, c]), np.asarray(expected), rtol=1e-5, atol=1e-5,
      )


@pytest.mark.parity_fast
def test_batched_monolithic_fn_matches_split_fn() -> None:
  """1a (monolithic, re-encodes per candidate) must match 1b (split fn) elementwise."""
  model = _make_model(3)
  coordinates, mask, residue_index, chain_index = _synthetic_structure()
  num_residues = coordinates.shape[0]
  candidates = _synthetic_candidates(num_candidates=4, num_residues=num_residues, seed=13)
  replicate_keys = jax.random.split(jax.random.PRNGKey(17), 2)

  batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(model)
  encodings = batched_encode_fn(
    coordinates, mask, residue_index, chain_index, replicate_keys,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  split_logits = batched_decode_fn(encodings, candidates)

  monolithic_fn = make_batched_conditional_logits_fn(model)
  monolithic_logits = monolithic_fn(
    coordinates, mask, residue_index, chain_index, replicate_keys, candidates,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )

  np.testing.assert_allclose(
    np.asarray(split_logits), np.asarray(monolithic_logits), rtol=1e-6, atol=1e-6,
  )


@pytest.mark.parity_fast
def test_prng_key_drives_backbone_noise() -> None:
  """(b) bb>0 distinct keys differ, same key identical, bb=0 all identical.

  Falsifies the pre-fix conditional_logits.py:90 docstring claim that prng_key is
  "unused but kept for API consistency" — it drives backbone-noise injection.
  """
  model = _make_model(4)
  coordinates, mask, residue_index, chain_index = _synthetic_structure()
  num_residues = coordinates.shape[0]
  candidates = _synthetic_candidates(num_candidates=2, num_residues=num_residues, seed=19)

  batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(model)

  # bb=0: all replicate keys are key-invariant -> identical logits across replicates.
  keys_bb0 = jax.random.split(jax.random.PRNGKey(23), 4)
  enc_bb0 = batched_encode_fn(
    coordinates, mask, residue_index, chain_index, keys_bb0,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  logits_bb0 = batched_decode_fn(enc_bb0, candidates)
  for r in range(1, keys_bb0.shape[0]):
    np.testing.assert_allclose(
      np.asarray(logits_bb0[0]), np.asarray(logits_bb0[r]), rtol=1e-6, atol=1e-6,
    )

  # bb>0: distinct replicate keys -> distinct logits.
  keys_bb_pos = jax.random.split(jax.random.PRNGKey(29), 4)
  enc_bb_pos = batched_encode_fn(
    coordinates, mask, residue_index, chain_index, keys_bb_pos,
    backbone_noise=jnp.array(0.3, dtype=jnp.float32),
  )
  logits_bb_pos = batched_decode_fn(enc_bb_pos, candidates)
  pairwise_identical = all(
    np.allclose(np.asarray(logits_bb_pos[0]), np.asarray(logits_bb_pos[r]), rtol=1e-6, atol=1e-6)
    for r in range(1, keys_bb_pos.shape[0])
  )
  assert not pairwise_identical, "distinct replicate keys at bb>0 must yield distinct logits"

  # bb>0: same key twice -> identical logits (deterministic given the key).
  same_key = jnp.stack([jax.random.PRNGKey(31)] * 3)
  enc_same_key = batched_encode_fn(
    coordinates, mask, residue_index, chain_index, same_key,
    backbone_noise=jnp.array(0.3, dtype=jnp.float32),
  )
  logits_same_key = batched_decode_fn(enc_same_key, candidates)
  for r in range(1, same_key.shape[0]):
    np.testing.assert_allclose(
      np.asarray(logits_same_key[0]), np.asarray(logits_same_key[r]), rtol=1e-6, atol=1e-6,
    )


@pytest.mark.parity_fast
def test_batched_split_fn_safe_map_matches_vmap_tile() -> None:
  """Explicit SafeMap tile (batch_size override) must match the default Vmap-fitting path."""
  model = _make_model(5)
  coordinates, mask, residue_index, chain_index = _synthetic_structure()
  num_residues = coordinates.shape[0]
  candidates = _synthetic_candidates(num_candidates=6, num_residues=num_residues, seed=37)
  replicate_keys = jax.random.split(jax.random.PRNGKey(41), 4)

  encode_default, decode_default = make_batched_conditional_logits_split_fn(model)
  encode_tiled, decode_tiled = make_batched_conditional_logits_split_fn(
    model, replicate_batch_size=2, candidate_batch_size=2,
  )

  enc_default = encode_default(
    coordinates, mask, residue_index, chain_index, replicate_keys,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  enc_tiled = encode_tiled(
    coordinates, mask, residue_index, chain_index, replicate_keys,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  logits_default = decode_default(enc_default, candidates)
  logits_tiled = decode_tiled(enc_tiled, candidates)

  np.testing.assert_allclose(
    np.asarray(logits_default), np.asarray(logits_tiled), rtol=1e-6, atol=1e-6,
  )
