"""Pytree structure and leaf-count tests for ModelInputs type hierarchy."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model_inputs import (
  BackboneGeometry,
  ConditioningFeatures,
  SamplingInputs,
  SamplingStaticConfig,
  ScoringInputs,
)
from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload


def _make_backbone(L: int = 8) -> BackboneGeometry:
  return BackboneGeometry(
    coords=jnp.zeros((L, 4, 3)),
    mask=jnp.ones((L,)),
    residue_index=jnp.arange(L, dtype=jnp.int32),
    chain_index=jnp.zeros((L,), dtype=jnp.int32),
  )


def _make_state_stack(n_states: int = 2, n_pad: int = 8) -> MultistateStackPayload:
  n_canonical = n_pad
  n_flat = n_states * n_canonical
  rows = jnp.stack([
    jnp.arange(s * n_canonical, (s + 1) * n_canonical, dtype=jnp.int32)
    for s in range(n_states)
  ])
  return MultistateStackPayload(
    coords_stack=jnp.zeros((n_states, n_pad, 4, 3)),
    mask_stack=jnp.ones((n_states, n_pad)),
    residue_index_stack=jnp.zeros((n_states, n_pad), dtype=jnp.int32),
    chain_index_stack=jnp.zeros((n_states, n_pad), dtype=jnp.int32),
    tie_group_map_stack=jnp.zeros((n_states, n_pad), dtype=jnp.int32),
    fixed_mask_stack=jnp.zeros((n_states, n_pad)),
    fixed_tokens_stack=jnp.zeros((n_states, n_pad), dtype=jnp.int32),
    state_flat_rows=rows,
    flat_row_offsets=jnp.arange(n_states + 1, dtype=jnp.int32) * n_canonical,
    state_index=jnp.arange(n_states, dtype=jnp.int32),
    state_embedding=jnp.zeros((n_states, 1)),
    n_states=n_states,
    n_canonical=n_canonical,
    n_flat=n_flat,
  )


def _make_wave_parallel(
  n_waves: int = 2,
  max_wave_size: int = 4,
  max_group_size: int = 2,
) -> WaveParallelPayload:
  return WaveParallelPayload(
    wave_group_ids=jnp.zeros((n_waves, max_wave_size), dtype=jnp.int32),
    wave_group_positions=jnp.zeros(
      (n_waves, max_wave_size, max_group_size), dtype=jnp.int32
    ),
    wave_group_valid=jnp.zeros((n_waves, max_wave_size), dtype=bool),
    wave_position_valid=jnp.zeros(
      (n_waves, max_wave_size, max_group_size), dtype=bool
    ),
  )


def _make_conditioning(L: int = 8, vocab: int = 21) -> ConditioningFeatures:
  return ConditioningFeatures(
    fixed_tokens=jnp.zeros((L,), dtype=jnp.int32),
    bias=jnp.zeros((L, vocab)),
    ar_mask=jnp.eye(L),
  )


def _make_sampling_inputs() -> SamplingInputs:
  return SamplingInputs(
    backbone=_make_backbone(),
    state_stack=_make_state_stack(),
    wave_parallel=_make_wave_parallel(),
    conditioning=_make_conditioning(),
  )


def test_backbone_geometry_is_pytree():
  bg = _make_backbone()
  leaves, treedef = jax.tree_util.tree_flatten(bg)
  assert len(leaves) == 4
  restored = jax.tree_util.tree_unflatten(treedef, leaves)
  assert restored.coords.shape == bg.coords.shape


def test_wave_parallel_payload_is_pytree():
  wp = _make_wave_parallel()
  leaves, treedef = jax.tree_util.tree_flatten(wp)
  assert len(leaves) == 4
  restored = jax.tree_util.tree_unflatten(treedef, leaves)
  assert restored.wave_group_ids.shape == wp.wave_group_ids.shape


def test_multistate_stack_payload_has_new_fields():
  ss = _make_state_stack(n_states=3)
  assert ss.state_index.shape == (3,)
  assert ss.state_embedding.shape == (3, 1)


def test_sampling_inputs_pytree_roundtrip():
  inputs = _make_sampling_inputs()
  leaves, treedef = jax.tree_util.tree_flatten(inputs)
  restored = jax.tree_util.tree_unflatten(treedef, leaves)
  assert restored.backbone.coords.shape == inputs.backbone.coords.shape
  assert restored.state_stack.n_states == inputs.state_stack.n_states


def test_sampling_inputs_no_retrace():
  """Identical pytree structure must not cause JIT recompilation."""
  trace_count = 0

  def _fn(inputs: SamplingInputs) -> jax.Array:
    nonlocal trace_count
    trace_count += 1
    return inputs.backbone.coords.sum()

  jitted = jax.jit(_fn)
  a = _make_sampling_inputs()
  b = _make_sampling_inputs()

  jitted(a)
  assert trace_count == 1, "first call should trace once"
  jitted(b)
  assert trace_count == 1, "second call with same structure must not retrace"


def test_scoring_inputs_pytree():
  si = ScoringInputs(
    backbone=_make_backbone(),
    sequences=jnp.zeros((4, 8), dtype=jnp.int32),
  )
  leaves, _ = jax.tree_util.tree_flatten(si)
  assert len(leaves) == 5  # 4 backbone + 1 sequences


def test_sampling_static_config_hashable():
  cfg = SamplingStaticConfig(
    decode_fn_uid="abc123", n_samples=10, temperature=1.0
  )
  assert hash(cfg) == hash(cfg)
  cfg2 = SamplingStaticConfig(
    decode_fn_uid="abc123", n_samples=10, temperature=1.0
  )
  assert cfg == cfg2
