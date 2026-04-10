"""Parity-focused tests for unconditional logits helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import pearsonr

from prxteinmpnn.model import PrxteinMPNN
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn


def _synthetic_structure(num_residues: int = 12) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Build deterministic synthetic structure inputs."""
  rng = np.random.default_rng(0)
  coordinates = jnp.array(rng.normal(size=(num_residues, 37, 3)).astype(np.float32))
  mask = jnp.ones((num_residues,), dtype=jnp.float32)
  residue_index = jnp.arange(num_residues, dtype=jnp.int32)
  chain_index = jnp.zeros((num_residues,), dtype=jnp.int32)
  return coordinates, mask, residue_index, chain_index


@pytest.mark.parity_fast
def test_make_unconditional_logits_fn_returns_callable() -> None:
  """Ensure the unconditional helper factory returns a callable."""
  model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=10,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(0),
  )
  helper = make_unconditional_logits_fn(eqx.tree_inference(model, value=True))
  assert callable(helper)


@pytest.mark.parity_fast
def test_unconditional_logits_helper_matches_model_branch() -> None:
  """Validate helper-vs-branch parity with allclose and correlation checks."""
  model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=10,
    dropout_rate=0.0,
    key=jax.random.PRNGKey(1),
  )
  model = eqx.tree_inference(model, value=True)
  logits_helper = make_unconditional_logits_fn(model)

  coordinates, mask, residue_index, chain_index = _synthetic_structure()
  ar_mask = jnp.tril(jnp.ones((mask.shape[0], mask.shape[0]), dtype=jnp.int32), k=-1)
  backbone_noise = jnp.array(0.0, dtype=jnp.float32)

  helper_logits = logits_helper(
    jax.random.PRNGKey(7),
    coordinates,
    mask,
    residue_index,
    chain_index,
    ar_mask=ar_mask,
    backbone_noise=backbone_noise,
  )
  _, direct_logits = model(
    coordinates,
    mask,
    residue_index,
    chain_index,
    decoding_approach="unconditional",
    ar_mask=ar_mask,
    backbone_noise=backbone_noise,
  )

  helper_logits_np = np.asarray(helper_logits)
  direct_logits_np = np.asarray(direct_logits)
  np.testing.assert_allclose(helper_logits_np, direct_logits_np, rtol=1e-6, atol=1e-6)
  corr = float(pearsonr(helper_logits_np.ravel(), direct_logits_np.ravel())[0])
  assert corr >= 0.999999


@pytest.mark.parity_fast
def test_unconditional_logits_helper_forwards_optional_arguments() -> None:
  """Ensure optional arguments are forwarded to the model call."""
  num_residues = 5
  mock_model = MagicMock(return_value=(None, jnp.ones((num_residues, 21), dtype=jnp.float32)))
  with patch("prxteinmpnn.sampling.unconditional_logits.jax.jit", new=lambda fn, *args, **kwargs: fn):
    logits_helper = make_unconditional_logits_fn(mock_model)

    coordinates, mask, residue_index, chain_index = _synthetic_structure(num_residues=num_residues)
    ar_mask = jnp.tril(jnp.ones((num_residues, num_residues), dtype=jnp.int32), k=-1)
    backbone_noise = jnp.array(0.1, dtype=jnp.float32)

    logits = logits_helper(
      jax.random.PRNGKey(0),
      coordinates,
      mask,
      residue_index,
      chain_index,
      ar_mask=ar_mask,
      backbone_noise=backbone_noise,
    )
    np.testing.assert_allclose(np.asarray(logits), np.ones((num_residues, 21), dtype=np.float32))
    mock_model.assert_called_once_with(
      coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="unconditional",
      ar_mask=ar_mask,
      backbone_noise=backbone_noise,
    )
