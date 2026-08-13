"""The ar_mask default must admit sequence context. Regression pins for #4222 / #4204.

`ar_mask[i, j] == 1` means position i SEES position j. An all-zero mask admits no sequence
information, so anything computed under it is a function of structure alone. That mistake
reached four sites in this codebase, each with a comment or docstring asserting the
opposite, and every one of them failed silently: right shape, right dtype, no error.

The tests here are differential invariants, not value checks. A value check would have
passed throughout the defect -- the logits were perfectly well-formed, they just did not
depend on the sequence. The only thing that separates a working conditional from a broken
one is whether CHANGING the sequence changes the answer.

`test_explicit_zero_mask_is_still_unconditional` is the other half and is deliberate: an
all-zero mask is a legitimate request (it is what "unconditional" means), so the fix is
that you must ask for it explicitly, not that you cannot have it. Without this test, a
later "fix" could make zeros behave like full context and destroy the capability while
still passing every other test in this file.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model import Aminx
from aminx.sampling.conditional_logits import (
  make_batched_conditional_logits_split_fn,
  make_conditional_logits_fn,
  make_encoding_conditional_logits_split_fn,
)
from aminx.utils.autoregression import full_context_ar_mask

NUM_RESIDUES = 12
CONTEXT_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def model() -> Aminx:
  built = Aminx(
    node_features=64,
    edge_features=64,
    hidden_features=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=8,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(2),
  )
  return eqx.tree_inference(built, value=True)


@pytest.fixture(scope="module")
def inputs():
  rng = np.random.default_rng(1)
  coordinates = jnp.asarray(rng.normal(size=(NUM_RESIDUES, 37, 3)).astype(np.float32))
  mask = jnp.ones((NUM_RESIDUES,), dtype=jnp.float32)
  residue_index = jnp.arange(NUM_RESIDUES, dtype=jnp.int32)
  chain_index = jnp.zeros((NUM_RESIDUES,), dtype=jnp.int32)
  native = jnp.asarray(rng.integers(1, 20, size=(NUM_RESIDUES,)).astype(np.int32))
  # Maximally different context: every position a different residue from the native one.
  alternative = jnp.zeros_like(native)
  return coordinates, mask, residue_index, chain_index, native, alternative


def _encoding(model, inputs):
  coordinates, mask, residue_index, chain_index, _, _ = inputs
  encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)
  encoding = encode_fn(
    coordinates,
    mask,
    residue_index,
    chain_index,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
    prng_key=jax.random.PRNGKey(0),
  )
  return encoding, decode_fn


def test_split_decode_default_reads_sequence_context(model, inputs):
  """#4222: the default was ZEROS, so this difference was exactly 0.0 for every input."""
  _, _, _, _, native, alternative = inputs
  encoding, decode_fn = _encoding(model, inputs)

  delta = float(jnp.max(jnp.abs(decode_fn(encoding, native) - decode_fn(encoding, alternative))))
  assert delta > CONTEXT_TOLERANCE, (
    f"changing the entire sequence moved the conditional logits by {delta:.3e}. The decoder "
    f"is not reading sequence context, so these are structure-only logits from a function "
    f"named for conditional ones -- the #4222 defect is back"
  )


def test_split_decode_default_matches_the_full_conditional_path(model, inputs):
  """The two public routes to a conditional must not disagree about what the default means.

  `make_conditional_logits_fn` goes through `build_inference_bundle(mode="score_conditional")`,
  which has always defaulted to `1 - I`. The split path defaulted to zeros. Both are
  documented as computing conditional logits, so a caller could reasonably pick either and
  get, silently, completely different objects.
  """
  coordinates, mask, residue_index, chain_index, native, _ = inputs
  encoding, decode_fn = _encoding(model, inputs)

  split = decode_fn(encoding, native)
  full = make_conditional_logits_fn(model)(
    jax.random.PRNGKey(0),
    coordinates,
    mask,
    residue_index,
    chain_index,
    native,
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  np.testing.assert_allclose(np.asarray(split), np.asarray(full), atol=1e-3, rtol=1e-3)


def test_explicit_zero_mask_is_still_unconditional(model, inputs):
  """The capability must survive the fix: zeros ON PURPOSE still means no sequence context.

  This is the guard against over-correcting. The defect was the DEFAULT, not the existence
  of an all-zero mask -- unconditional scoring is a real thing aminx supports elsewhere.
  """
  _, _, _, _, native, alternative = inputs
  encoding, decode_fn = _encoding(model, inputs)
  zeros = jnp.zeros((NUM_RESIDUES, NUM_RESIDUES), dtype=jnp.float32)

  delta = float(
    jnp.max(jnp.abs(decode_fn(encoding, native, zeros) - decode_fn(encoding, alternative, zeros)))
  )
  assert delta == 0.0, (
    f"an explicitly all-zero ar_mask leaked sequence information (delta {delta:.3e}); "
    f"unconditional scoring is a supported request and must stay exactly unconditional"
  )


def test_batched_decode_default_reads_sequence_context(model, inputs):
  """The runner's `conditional_logits` feature calls this with no ar_mask at all.

  Before #4222 that meant every candidate in a batch scored identically -- the batch axis
  existed but carried no information.
  """
  coordinates, mask, residue_index, chain_index, native, alternative = inputs
  batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(model)
  encodings = batched_encode_fn(
    coordinates,
    mask,
    residue_index,
    chain_index,
    jax.random.split(jax.random.PRNGKey(0), 1),
    backbone_noise=jnp.array(0.0, dtype=jnp.float32),
  )
  candidates = jnp.stack([native, alternative])

  logits = np.asarray(batched_decode_fn(encodings, candidates))
  assert logits.shape == (1, 2, NUM_RESIDUES, 21)
  delta = float(np.max(np.abs(logits[0, 0] - logits[0, 1])))
  assert delta > CONTEXT_TOLERANCE, (
    f"two different candidate sequences scored identically (delta {delta:.3e}); the "
    f"candidate axis is carrying no sequence information"
  )


def test_full_context_mask_is_ones_off_diagonal():
  """The mask itself: sees everything except self."""
  built = np.asarray(full_context_ar_mask(5))
  np.testing.assert_array_equal(np.diag(built), np.zeros(5))
  assert built.sum() == 5 * 5 - 5
  assert set(np.unique(built).tolist()) == {0.0, 1.0}
