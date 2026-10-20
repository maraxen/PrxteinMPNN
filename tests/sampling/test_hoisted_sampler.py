"""Tests for aminx.sampling.hoisted, the encode-once sampler (spec 261020_hoisted-sampler).

A1 parity with the monolithic vmap, A3 varying-input refusal through SAMPLER_AXIS,
A4 parity probe, A5 precomputed_* warning, A7 shared stage compiles once across Python-float temperatures.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.inference import make_hoisted
from xtrax.inference.errors import HoistParityError, HoistVaryingInputError

import aminx.sampling.hoisted as hoisted_mod
from aminx.model import Aminx
from aminx.sampling import make_sample_sequences
from aminx.sampling.hoisted import SAMPLER_AXIS, HoistedSampler, make_hoisted_sampler

L = 12  # residues: small, and (L, K, E) = (12, 8, 32) is not produced elsewhere in the graph
K = 8  # neighbours
E = 32  # node / edge feature width
_KEY = jax.random.PRNGKey(3)


@pytest.fixture(scope="module")
def small_model() -> Aminx:
  """Tiny random-weight model, dropout off."""
  # dropout_rate=0: the encoder otherwise draws dropout masks keyed on the encode key, which
  # would make the hoisted encode (fixed key) differ from the monolithic one for reasons
  # unrelated to the hoist.
  return Aminx(
    dropout_rate=0.0,
    node_features=E,
    edge_features=E,
    hidden_features=E,
    num_encoder_layers=1,
    num_decoder_layers=2,
    k_neighbors=K,
    key=_KEY,
  )


def _structure(num_states: int = 1) -> dict[str, Any]:
  rng = np.random.default_rng(1)
  coords = jnp.asarray(rng.normal(size=(L, 4, 3)).astype(np.float32) * 5.0)
  mask = jnp.ones((L,), dtype=jnp.float32)
  residue_index = jnp.arange(L, dtype=jnp.int32)
  chain_index = jnp.zeros((L,), dtype=jnp.int32)
  if num_states == 1:
    return {
      "structure_coordinates": coords,
      "mask": mask,
      "residue_index": residue_index,
      "chain_index": chain_index,
    }
  stack = lambda x: jnp.stack([x] * num_states, axis=0)  # noqa: E731
  return {
    "structure_coordinates": stack(coords),
    "mask": stack(mask),
    "residue_index": stack(residue_index),
    "chain_index": stack(chain_index),
  }


def _case_kwargs(case: str) -> dict[str, Any]:
  if case == "single":
    return {**_structure(1), "temperature": 0.5}
  if case == "multistate":
    return {
      **_structure(2),
      "temperature": 0.5,
      "state_weights": jnp.array([0.5, 0.5], dtype=jnp.float32),
    }
  if case == "tied":
    tie = jnp.arange(L, dtype=jnp.int32).at[1].set(0).at[2].set(0).at[6].set(5)
    return {**_structure(1), "temperature": 0.5, "tie_group_map": tie}
  raise ValueError(case)


def _num_groups(case: str) -> int | None:
  if case == "tied":
    tie = jnp.arange(L, dtype=jnp.int32).at[1].set(0).at[2].set(0).at[6].set(5)
    return int(jnp.unique(tie).shape[0])
  return None


def _monolithic(model: Aminx, kwargs: dict[str, Any], keys: jax.Array, num_groups: int | None):
  sampler = make_sample_sequences(model, sampling_strategy="temperature")
  return jax.vmap(
    lambda k: sampler(k, num_groups=num_groups, **kwargs),
  )(keys)


# --- A1: parity with the monolithic vmap --------------------------------------------------


@pytest.mark.parametrize("case", ["single", "multistate", "tied"])
@pytest.mark.parametrize("n", [1, 5])
def test_a1_matches_monolithic_vmap(small_model: Aminx, case: str, n: int) -> None:
  """A1: hoisted sampler equals the monolithic vmap, sequences and orders exactly."""
  kwargs = _case_kwargs(case)
  num_groups = _num_groups(case)
  keys = jax.random.split(jax.random.PRNGKey(7), n)

  sampler = make_hoisted_sampler(small_model, num_groups=num_groups)
  seqs, logits, orders = sampler(keys, **kwargs)
  want_seqs, want_logits, want_orders = _monolithic(small_model, kwargs, keys, num_groups)

  np.testing.assert_array_equal(np.asarray(seqs), np.asarray(want_seqs))
  np.testing.assert_array_equal(np.asarray(orders), np.asarray(want_orders))
  np.testing.assert_allclose(np.asarray(logits), np.asarray(want_logits), rtol=1e-5, atol=1e-6)
  assert seqs.dtype == jnp.int8
  assert seqs.shape == (n, L)
  assert logits.shape == (n, L, 21)


# --- A3: varying-input refusal through the sampler axis -----------------------------------


def test_a3_keys_in_shared_args_is_refused() -> None:
  """A3: keys wired into shared args raise HoistVaryingInputError."""
  assert SAMPLER_AXIS.varying_inputs == ("keys",)
  keys = jax.random.split(jax.random.PRNGKey(0), 2)
  hoisted = make_hoisted(
    lambda keys: keys.sum(),
    lambda shared, key: shared + key.sum(),
    axis=SAMPLER_AXIS,
  )
  with pytest.raises(HoistVaryingInputError):
    hoisted({"keys": keys}, keys)


# --- A4: parity probe ----------------------------------------------------------------------


def test_a4_probe_passes_on_a1_inputs(small_model: Aminx) -> None:
  """A4: the parity probe passes on A1 inputs."""
  kwargs = _case_kwargs("multistate")
  keys = jax.random.split(jax.random.PRNGKey(7), 3)
  sampler = make_hoisted_sampler(small_model)
  seqs, logits, _ = sampler(keys, check_elements=2, **kwargs)
  assert seqs.shape == (3, L)
  assert logits.shape == (3, L, 21)


def test_a4_probe_raises_on_perturbed_reference(
  small_model: Aminx, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A4: a reference perturbed by 1e-3 in logits raises HoistParityError."""
  kwargs = _case_kwargs("single")
  keys = jax.random.split(jax.random.PRNGKey(7), 3)
  sampler = make_hoisted_sampler(small_model)
  real_reference = sampler._reference

  def perturbed(shared_args: dict[str, Any], key: jax.Array) -> tuple[jax.Array, ...]:
    """Reference with logits perturbed by 1e-3."""
    seq, logits, order = real_reference(shared_args, key)
    return seq, logits + 1e-3, order

  monkeypatch.setattr(sampler, "_reference", perturbed)
  with pytest.raises(HoistParityError):
    sampler(keys, check_elements=1, **kwargs)


# --- A5: precomputed_* kwargs warn and change nothing --------------------------------------


def test_a5_precomputed_kwargs_warn_and_are_ignored(small_model: Aminx) -> None:
  """A5: precomputed_* kwargs warn with FutureWarning and change nothing."""
  kwargs = _case_kwargs("single")
  key = jax.random.PRNGKey(7)
  sampler = make_sample_sequences(small_model, sampling_strategy="temperature")
  want = sampler(key, **kwargs)

  precomputed = jnp.zeros((L, E), dtype=jnp.float32)
  with pytest.warns(FutureWarning, match="precomputed_node_features"):
    got = sampler(key, precomputed_node_features=precomputed, **kwargs)

  np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
  np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[1]), rtol=0, atol=0)


# --- A7: one shared-stage compile across Python-float temperatures --------------------------


def test_a7_shared_stage_compiles_once_across_python_float_temperatures(
  small_model: Aminx, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A7: two Python-float temperatures compile the shared stage once."""
  traces = [0]
  real_encode = hoisted_mod.encode_for_sampling

  def counting_encode(*args: Any, **kwargs: Any) -> Any:
    """Count trace-time encoder calls."""
    traces[0] += 1  # runs at trace time, once per compilation of the shared stage
    return real_encode(*args, **kwargs)

  monkeypatch.setattr(hoisted_mod, "encode_for_sampling", counting_encode)
  base = _structure(1)
  keys = jax.random.split(jax.random.PRNGKey(7), 2)
  sampler = make_hoisted_sampler(small_model)

  sampler(keys, temperature=0.5, **base)
  sampler(keys, temperature=0.7, **base)
  assert traces[0] == 1
  assert sampler.stats()["shared_calls"] == 2
