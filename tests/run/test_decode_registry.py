"""Tests for DecodeFnRegistry: registration, UID stability, resolve."""

import jax.numpy as jnp
import pytest

from prxteinmpnn.run.decode_registry import register_decode_fn, resolve_decode_fn


def _arithmetic_mean(logits_stack, state_index, state_weights):
  return jnp.mean(logits_stack, axis=0)


def test_register_returns_uid():
  uid = register_decode_fn(_arithmetic_mean, name="arithmetic_mean")
  assert isinstance(uid, str)
  assert len(uid) == 16


def test_register_idempotent():
  uid1 = register_decode_fn(_arithmetic_mean)
  uid2 = register_decode_fn(_arithmetic_mean)
  assert uid1 == uid2


def test_resolve_returns_fn():
  uid = register_decode_fn(_arithmetic_mean)
  fn = resolve_decode_fn(uid)
  result = fn(
    jnp.zeros((2, 4, 21)),
    jnp.array([0, 1]),
    jnp.array([0.5, 0.5]),
  )
  assert result.shape == (4, 21)


def test_resolve_unknown_uid_raises():
  with pytest.raises(KeyError, match="No decode fn registered"):
    resolve_decode_fn("doesnotexist")
