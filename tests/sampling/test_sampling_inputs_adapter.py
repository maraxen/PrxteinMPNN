"""Tests for make_sampling_inputs_from_spec and make_static_config_from_spec."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model_inputs import SamplingInputs, SamplingStaticConfig
from prxteinmpnn.run.decode_registry import DEFAULT_DECODE_FN_UID, register_decode_fn, resolve_decode_fn
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.sampling.sample import make_sampling_inputs_from_spec, make_static_config_from_spec


def _spec(**kw) -> SamplingSpecification:
  defaults = dict(inputs=[], num_samples=4, temperature=0.1)
  defaults.update(kw)
  return SamplingSpecification(**defaults)


def _backbone_arrays(L: int = 8):
  return (
    jnp.zeros((L, 4, 3), dtype=jnp.float32),  # coords
    jnp.ones((L,), dtype=jnp.float32),          # mask
    jnp.arange(L, dtype=jnp.int32),              # residue_index
    jnp.zeros((L,), dtype=jnp.int32),            # chain_index
  )


def test_make_static_config_defaults():
  spec = _spec()
  cfg = make_static_config_from_spec(spec)
  assert isinstance(cfg, SamplingStaticConfig)
  assert cfg.n_samples == 4
  assert cfg.temperature == pytest.approx(0.1)
  assert cfg.decode_fn_uid == DEFAULT_DECODE_FN_UID


def test_make_static_config_custom_fn():
  def my_fn(logits_stack, state_index, state_weights):
    return jnp.mean(logits_stack, axis=0)

  spec = _spec()
  cfg = make_static_config_from_spec(spec, decode_fn=my_fn)
  assert cfg.decode_fn_uid != DEFAULT_DECODE_FN_UID
  assert isinstance(cfg.decode_fn_uid, str)
  assert len(cfg.decode_fn_uid) == 16


def test_make_static_config_from_spec_decode_fn():
  def my_fn(logits_stack, state_index, state_weights):
    return jnp.sum(logits_stack, axis=0)

  spec = _spec(decode_fn=my_fn)
  cfg = make_static_config_from_spec(spec)
  assert cfg.decode_fn_uid != DEFAULT_DECODE_FN_UID


def test_make_sampling_inputs_trivial():
  spec = _spec()
  coords, mask, res_idx, chain_idx = _backbone_arrays(L=8)
  inputs = make_sampling_inputs_from_spec(spec, coords, mask, res_idx, chain_idx)
  assert isinstance(inputs, SamplingInputs)
  assert inputs.backbone.coords.shape == (8, 4, 3)
  assert inputs.state_stack.n_states == 1
  assert inputs.state_stack.state_index.shape == (1,)
  assert inputs.state_stack.state_embedding.shape == (1, 1)
  assert inputs.wave_parallel.wave_group_ids.shape[0] == 1
  assert inputs.conditioning.bias.shape == (8, 21)
  assert inputs.conditioning.ar_mask.shape == (8, 8)


def test_make_sampling_inputs_pytree_roundtrip():
  spec = _spec()
  coords, mask, res_idx, chain_idx = _backbone_arrays(L=6)
  inputs = make_sampling_inputs_from_spec(spec, coords, mask, res_idx, chain_idx)
  leaves, treedef = jax.tree_util.tree_flatten(inputs)
  restored = jax.tree_util.tree_unflatten(treedef, leaves)
  assert restored.backbone.mask.shape == (6,)


def test_make_sampling_inputs_no_retrace():
  spec = _spec()
  trace_count = 0

  def fn(inp: SamplingInputs) -> jax.Array:
    nonlocal trace_count
    trace_count += 1
    return inp.backbone.coords.sum()

  jitted = jax.jit(fn)
  coords, mask, res_idx, chain_idx = _backbone_arrays()
  a = make_sampling_inputs_from_spec(spec, coords, mask, res_idx, chain_idx)
  b = make_sampling_inputs_from_spec(spec, coords * 2.0, mask, res_idx, chain_idx)
  jitted(a)
  assert trace_count == 1
  jitted(b)
  assert trace_count == 1, "same structure must not retrace"


def test_state_index_and_embedding_filled():
  from prxteinmpnn.sampling.state_vmap_prep import multistate_stack_payload_from_loose_ar_host
  L, S = 6, 3
  rows = jnp.stack([jnp.arange(i * L, (i + 1) * L, dtype=jnp.int32) for i in range(S)])
  ms = multistate_stack_payload_from_loose_ar_host(
    coords_stack=jnp.zeros((S, L, 4, 3)),
    mask_stack=jnp.ones((S, L)),
    residue_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
    chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
    tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
    fixed_mask_stack=jnp.zeros((S, L)),
    fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
    state_flat_rows=rows,
    n_canonical=L,
  )
  assert ms.state_index.shape == (S,)
  assert jnp.array_equal(ms.state_index, jnp.arange(S, dtype=jnp.int32))
  assert ms.state_embedding.shape == (S, 1)


def test_make_static_config_batch_fn_uid_matches_registry():
  """batch_fn resolved from spec.decode_fn matches what's in the registry."""
  def my_fn(logits_stack, state_index, state_weights):
    return jnp.mean(logits_stack, axis=0)

  uid = register_decode_fn(my_fn, name="test_fn")
  resolved = resolve_decode_fn(uid)
  assert resolved is my_fn

  spec = _spec(decode_fn=my_fn)
  cfg = make_static_config_from_spec(spec)
  assert cfg.decode_fn_uid == uid
