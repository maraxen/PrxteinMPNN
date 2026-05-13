"""Host-callable ``state_vmap_exact`` logits via :class:`~prxteinmpnn.payloads.MultistateStackPayload`.

JIT factories :func:`~prxteinmpnn.sampling.unconditional_logits.make_unconditional_logits_state_vmap_fn`
and :func:`~prxteinmpnn.sampling.conditional_logits.make_conditional_logits_state_vmap_fn` keep a
tuple-first surface for historical call sites. These helpers delegate to
:meth:`prxteinmpnn.model.mpnn.PrxteinMPNN.score_unconditional_from_payload` /
:meth:`~prxteinmpnn.model.mpnn.PrxteinMPNN.score_conditional_from_payload`
(or ligand equivalents) for payload-first tooling.
"""

from __future__ import annotations

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack
from prxteinmpnn.payloads import LigandStack, MultistateStackPayload


def _inference_model(model: PrxteinMPNN | PrxteinLigandMPNN) -> PrxteinMPNN | PrxteinLigandMPNN:
  return cast(
    "PrxteinMPNN | PrxteinLigandMPNN",
    eqx.nn.inference_mode(model, value=True) if isinstance(model, eqx.Module) else model,
  )


def unconditional_state_vmap_logits_from_payload(
  model: PrxteinMPNN | PrxteinLigandMPNN,
  prng_key: jax.Array,
  stack: MultistateStackPayload,
  ligand: LigandStack | None,
  *,
  tie_group_map: jax.Array | None,
  multi_state_strategy_idx: jax.Array,
  state_weights: jax.Array | None,
  state_mapping: jax.Array | None,
  states_chunk_size: int | None = None,
) -> jax.Array:
  """Unconditional stacked logits using a :class:`MultistateStackPayload` (+ ligand stack when needed)."""
  m = _inference_model(model)
  if isinstance(m, PrxteinLigandMPNN):
    if ligand is None:
      msg = "PrxteinLigandMPNN requires ligand="
      raise TypeError(msg)
    extra: dict[str, object] = {}
    if states_chunk_size is not None:
      extra["states_chunk_size"] = states_chunk_size
    return m.score_unconditional_from_payload(
      prng_key,
      stack,
      ligand,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      **extra,
    )
  if ligand is not None:
    msg = "ligand= must be None for PrxteinMPNN"
    raise ValueError(msg)
  return m.score_unconditional_from_payload(
    prng_key,
    stack,
    tie_group_map=tie_group_map,
    multi_state_strategy_idx=multi_state_strategy_idx,
    state_weights=state_weights,
    state_mapping=state_mapping,
    inference=True,
  )


def conditional_state_vmap_logits_from_payload(
  model: PrxteinMPNN | PrxteinLigandMPNN,
  prng_key: jax.Array,
  sequence: jax.Array,
  stack: MultistateStackPayload,
  ligand: LigandStack | None,
  *,
  tie_group_map: jax.Array | None,
  multi_state_strategy_idx: jax.Array,
  state_weights: jax.Array | None,
  state_mapping: jax.Array | None,
  ar_mask_stack: jax.Array | None = None,
  bias_flat: jax.Array | None = None,
  states_chunk_size: int | None = None,
) -> jax.Array:
  """Conditional stacked logits (teacher-forced) using stack (+ ligand stack when needed)."""
  m = _inference_model(model)
  n_emb = int(m.w_s_embed.num_embeddings)
  oh = jax.nn.one_hot(sequence, n_emb) if sequence.ndim == 1 else sequence
  seq_stack = gather_flat_to_stack(oh, stack.state_flat_rows)
  s_dim, p_dim = stack.mask_stack.shape[0], stack.mask_stack.shape[1]
  arm = (
    jnp.zeros((s_dim, p_dim, p_dim), dtype=jnp.int32)
    if ar_mask_stack is None
    else ar_mask_stack
  )
  if isinstance(m, PrxteinLigandMPNN):
    if ligand is None:
      msg = "PrxteinLigandMPNN requires ligand="
      raise TypeError(msg)
    extra: dict[str, object] = {}
    if states_chunk_size is not None:
      extra["states_chunk_size"] = states_chunk_size
    return m.score_conditional_from_payload(
      prng_key,
      stack,
      ligand,
      seq_stack,
      arm,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      bias_flat=bias_flat,
      inference=True,
      **extra,
    )
  if ligand is not None:
    msg = "ligand= must be None for PrxteinMPNN"
    raise ValueError(msg)
  return m.score_conditional_from_payload(
    prng_key,
    stack,
    seq_stack,
    arm,
    tie_group_map=tie_group_map,
    multi_state_strategy_idx=multi_state_strategy_idx,
    state_weights=state_weights,
    state_mapping=state_mapping,
    bias_flat=bias_flat,
    inference=True,
  )
