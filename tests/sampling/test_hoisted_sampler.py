"""Tests for aminx.sampling.hoisted, the encode-once sampler (spec 261020_hoisted-sampler).

A1 parity with the monolithic vmap (single, multi-state, tied, bias, fixed positions,
inference_only, chunked batch), A2 encoder execution count, A3 keys in shared args refused by the
SAMPLER_AXIS declaration (via xtrax HoistedMap), A4 parity probe, A5 precomputed_* warning,
A7 shared stage compiles once across Python-float temperatures.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from xtrax.inference import make_hoisted
from xtrax.inference.errors import HoistParityError, HoistVaryingInputError

import aminx.inference.sample_autoregressive as sample_autoregressive
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
  # dropout_rate=0 keeps the encoder deterministic in its key. With dropout on, a freshly
  # constructed Aminx applies dropout inside the encoder even at inference (Aminx.__call__ does
  # not forward `inference` to it). The hoisted encode (fixed key) and the monolithic encode
  # (per-key) would then differ for reasons unrelated to the hoist. Pretrained checkpoints do not
  # show this (see tests/sampling/test_decoding_order_mask_consistency.py).
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


def _tie_map() -> jax.Array:
  return jnp.arange(L, dtype=jnp.int32).at[1].set(0).at[2].set(0).at[6].set(5)


# case name -> (structure/conditioning kwargs, sampler options). Options are constructor-level
# arguments of make_hoisted_sampler and static arguments of make_sample_sequences.
CASES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


def _register(name: str, kwargs: dict[str, Any], options: dict[str, Any] | None = None) -> None:
  CASES[name] = (kwargs, options or {})


_register("single", {**_structure(1), "temperature": 0.5})
_register(
  "multistate",
  {**_structure(2), "temperature": 0.5, "state_weights": jnp.array([0.5, 0.5], jnp.float32)},
)
_register(
  "multistate_product",
  {**_structure(2), "temperature": 0.5, "state_weights": jnp.array([0.5, 0.5], jnp.float32)},
  {"multi_state_strategy": "product"},
)
_register(
  "multistate_geometric",
  {**_structure(2), "temperature": 0.5, "state_weights": jnp.array([0.5, 0.5], jnp.float32)},
  {"multi_state_strategy": "geometric_mean"},
)
_register(
  "tied",
  {**_structure(1), "temperature": 0.5, "tie_group_map": _tie_map()},
  {"num_groups": 9},  # len(unique(_tie_map())): positions 1,2 -> 0 and 6 -> 5
)
_register(
  "bias",
  {
    **_structure(1),
    "temperature": 0.5,
    "bias": jnp.asarray(np.random.default_rng(5).normal(size=(L, 21)).astype(np.float32)),
  },
)
_register(
  "fixed",
  {
    **_structure(1),
    "temperature": 0.5,
    "fixed_mask": (jnp.arange(L) % 3 == 0).astype(jnp.float32),
    "fixed_tokens": (jnp.arange(L) % 20).astype(jnp.int32),
  },
)
_register("inference_only", {**_structure(1), "temperature": 0.5}, {"inference_only": True})
_register("chunked_batch2", {**_structure(1), "temperature": 0.5}, {"batch_size": 2})


def _monolithic(model: Aminx, case: str, keys: jax.Array) -> Any:
  kwargs, options = CASES[case]
  sampler = make_sample_sequences(model, sampling_strategy="temperature")
  passthrough = {k: options[k] for k in ("num_groups", "multi_state_strategy", "inference_only") if k in options}
  return jax.vmap(lambda k: sampler(k, **passthrough, **kwargs))(keys)


# --- A1: parity with the monolithic vmap --------------------------------------------------


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("n", [1, 5])
def test_a1_matches_monolithic_vmap(small_model: Aminx, case: str, n: int) -> None:
  """A1: hoisted sampler equals the monolithic vmap, sequences and orders exactly.

  n=5 with batch_size=2 is a ragged chunked_map (2, 2, 1). Logits are compared to
  rtol=1e-5, atol=1e-6.
  """
  kwargs, options = CASES[case]
  keys = jax.random.split(jax.random.PRNGKey(7), n)

  sampler = make_hoisted_sampler(small_model, **options)
  seqs, logits, orders = sampler(keys, **kwargs)
  want_seqs, want_logits, want_orders = _monolithic(small_model, case, keys)

  np.testing.assert_array_equal(np.asarray(seqs), np.asarray(want_seqs))
  np.testing.assert_array_equal(np.asarray(orders), np.asarray(want_orders))
  np.testing.assert_allclose(np.asarray(logits), np.asarray(want_logits), rtol=1e-5, atol=1e-6)
  assert seqs.dtype == jnp.int8
  assert seqs.shape == (n, L)
  assert logits.shape == (n, L, 21)


# --- A2: the encoder executes once per call (execution counter, not jaxpr shapes) ---------
#
# A jaxpr shape check cannot isolate the encoder: the AR decode also produces per-key
# (L, K, E) tensors. So the encoder is counted at execution time. The wrapper reports
# through jax.debug.callback. Under vmap the callback runs once with the batched value, so
# the count is the number of encoder outputs in the batch (one per key), and an unbatched
# shared encode counts 1.


def _encoder_execution_counter(
  monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
  """Patch encode_for_sampling in both modules to count encoder executions."""
  counter = [0]
  real = sample_autoregressive.encode_for_sampling

  def counting(model: Aminx, bundle: Any, k_enc: Any, config: Any) -> Any:
    enc = real(model, bundle, k_enc, config)

    def bump(neighbor_indices: Any) -> None:
      counter[0] += int(np.size(neighbor_indices)) // (L * K)

    jax.debug.callback(bump, enc.neighbor_indices)
    return enc

  monkeypatch.setattr(sample_autoregressive, "encode_for_sampling", counting)
  monkeypatch.setattr(hoisted_mod, "encode_for_sampling", counting)
  return counter


def test_a2_encoder_executes_once_per_hoisted_call(
  small_model: Aminx, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A2: monolithic vmap runs the encoder once per key; the hoisted sampler runs it once."""
  n = 5
  kwargs = CASES["single"][0]
  keys = jax.random.split(jax.random.PRNGKey(7), n)
  counter = _encoder_execution_counter(monkeypatch)

  mono = make_sample_sequences(small_model, sampling_strategy="temperature")
  out = jax.vmap(lambda k: mono(k, **kwargs))(keys)
  jax.block_until_ready(out)
  jax.effects_barrier()
  assert counter[0] == n

  counter[0] = 0
  sampler = make_hoisted_sampler(small_model)
  out = sampler(keys, **kwargs)
  jax.block_until_ready(out)
  jax.effects_barrier()
  assert counter[0] == 1


def test_a2_negative_control_mutant_encodes_per_element(
  small_model: Aminx, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A2 control: a per-element encode fires once per key, on top of the shared encode."""

  class _EncodePerElement(HoistedSampler):
    """Mutant: each element re-encodes with a key derived from its own key."""

    def _per_element(self, shared: Any, key: Any) -> Any:
      bundle, config, stage_set, _ = shared
      k_enc, _ = jax.random.split(jax.random.split(key)[1])
      enc = hoisted_mod.encode_for_sampling(self._model, bundle, k_enc, config)
      return super()._per_element((bundle, config, stage_set, enc), key)

  n = 5
  kwargs = CASES["single"][0]
  keys = jax.random.split(jax.random.PRNGKey(7), n)
  counter = _encoder_execution_counter(monkeypatch)
  mutant = _EncodePerElement(
    small_model,
    decoding_order_fn=hoisted_mod._DEFAULT_DECODING_ORDER_FN,
    num_groups=None,
    multi_state_strategy="arithmetic_mean",
    multi_state_temperature=1.0,
    inference_only=False,
    batch_size=None,
  )
  out = mutant(keys, **kwargs)
  jax.block_until_ready(out)
  jax.effects_barrier()
  # One shared encode (the mutant still runs the library shared stage) plus N per-element.
  assert counter[0] == n + 1


# --- A3: varying-input refusal through the sampler axis -----------------------------------


def test_a3_sampler_axis_refuses_keys_in_shared_args_via_xtrax_hoistedmap() -> None:
  """A3: SAMPLER_AXIS (varying_inputs=keys) makes xtrax HoistedMap refuse keys in shared args.

    HoistedSampler cannot wire keys into shared args, so this exercises the xtrax name check
    directly with the same axis declaration the sampler uses.
    """
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
  kwargs = CASES["multistate"][0]
  keys = jax.random.split(jax.random.PRNGKey(7), 3)
  sampler = make_hoisted_sampler(small_model)
  seqs, logits, _ = sampler(keys, check_elements=2, **kwargs)
  assert seqs.shape == (3, L)
  assert logits.shape == (3, L, 21)


def test_a4_probe_raises_on_perturbed_reference(
  small_model: Aminx, monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A4: a reference perturbed by 1e-3 in logits raises HoistParityError."""
  kwargs = CASES["single"][0]
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
  kwargs = CASES["single"][0]
  key = jax.random.PRNGKey(7)
  sampler = make_sample_sequences(small_model, sampling_strategy="temperature")
  want = sampler(key, **kwargs)

  precomputed = jnp.zeros((L, E), dtype=jnp.float32)
  with pytest.warns(FutureWarning, match="precomputed_node_features"):
    got = sampler(key, precomputed_node_features=precomputed, **kwargs)

  np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(want[0]))
  np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[1]), rtol=0, atol=0)


def test_a5_guard_preserves_jitted_sampler_attributes(small_model: Aminx) -> None:
  """A5 guard: the wrapped sampler still exposes the jitted object's attributes (.lower, ...)."""
  sampler = make_sample_sequences(small_model, sampling_strategy="temperature")
  assert callable(sampler.lower)
  assert callable(sampler.eval_shape)
  assert callable(sampler.__wrapped__)
  kwargs = CASES["single"][0]
  lowered = sampler.lower(jax.random.PRNGKey(0), **kwargs)
  assert lowered is not None


def test_a5_positional_precomputed_argument_warns(small_model: Aminx) -> None:
  """A5 guard: a precomputed_* passed positionally is bound against the signature and warns."""
  import inspect

  sampler = make_sample_sequences(small_model, sampling_strategy="temperature")
  names = list(inspect.signature(sampler).parameters)
  idx = names.index("precomputed_node_features")
  values: dict[str, Any] = {
    "prng_key": jax.random.PRNGKey(7),
    "temperature": 0.5,
    **{k: v for k, v in CASES["single"][0].items() if k != "temperature"},
  }
  params = inspect.signature(sampler).parameters
  args = [
    values[name] if name in values else params[name].default for name in names[:idx]
  ]
  with pytest.warns(FutureWarning, match="precomputed_node_features"):
    sampler(*args, jnp.zeros((L, E), dtype=jnp.float32))


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
