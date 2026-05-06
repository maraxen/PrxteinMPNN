"""Factory for creating unconditional logits functions.

Unconditional logits are computed without providing a sequence input,
allowing the model to predict the most likely amino acids at each position
based solely on the structure.

Multistate parallel encode (``state_vmap_exact``): use
:func:`make_unconditional_logits_state_vmap_fn`, or call ``model(...,
decoding_approach=\"unconditional\", multistate_mode=\"state_vmap_exact\", ...)``
with ``coords_stack`` / ``mask_stack`` / ``state_flat_rows`` / ``n_flat`` (and
ligand ``y_*_stack`` when applicable).

For a :class:`~prxteinmpnn.payloads.MultistateStackPayload` carrier, prefer
:func:`prxteinmpnn.sampling.state_vmap_payload_logits.unconditional_state_vmap_logits_from_payload`
or :meth:`prxteinmpnn.model.mpnn.PrxteinMPNN.score_unconditional_state_vmap_exact_from_payload`
or :meth:`~prxteinmpnn.model.mpnn.PrxteinLigandMPNN.score_unconditional_state_vmap_exact_from_payload`.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.protocols import StateVmapExactLogitsFn, UnconditionalLogitsFn


def make_unconditional_logits_fn(
  model: PrxteinMPNN | PrxteinLigandMPNN,
) -> UnconditionalLogitsFn:
  """Return a JIT function for dense single-graph unconditional logits."""

  @partial(jax.jit)
  def unconditional_logits(
    prng_key: jax.Array,
    structure_coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    ar_mask: jax.Array | None = None,
    backbone_noise: jax.Array | None = None,
  ) -> jax.Array:
    del prng_key
    _, logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="unconditional",
      ar_mask=ar_mask,
      backbone_noise=backbone_noise,
    )
    return logits

  return cast("UnconditionalLogitsFn", unconditional_logits)


def make_unconditional_logits_state_vmap_fn(
  model: PrxteinMPNN | PrxteinLigandMPNN,
) -> StateVmapExactLogitsFn:
  """JIT ``score_unconditional_state_vmap_exact``: stacked encode → scattered flat logits + fuse."""
  from prxteinmpnn.model.mpnn import PrxteinLigandMPNN as _LM
  from prxteinmpnn.model.mpnn import PrxteinMPNN as _PM

  m = eqx.nn.inference_mode(model, value=True) if isinstance(model, eqx.Module) else model
  is_lig = isinstance(model, _LM)

  def strategy_idx(strategy: str) -> jax.Array:
    return jnp.int32({"arithmetic_mean": 0, "geometric_mean": 1, "product": 2}[strategy])

  if is_lig:

    def unconditional_stack(
      prng_key: jax.Array,
      coords_stack: jax.Array,
      mask_stack: jax.Array,
      residue_index_stack: jax.Array,
      chain_index_stack: jax.Array,
      y_stack: jax.Array,
      y_t_stack: jax.Array,
      y_m_stack: jax.Array,
      state_flat_rows: jax.Array,
      n_flat: int,
      tie_group_map: jax.Array | None,
      multi_state_strategy_idx: jax.Array,
      multi_state_temperature: jax.Array | float,
      state_weights: jax.Array | None,
      state_mapping: jax.Array | None,
      states_chunk_size: int | None = None,
    ) -> jax.Array:
      """No outer ``jax.jit``: host state-chunk loop inside ``score_unconditional_state_vmap_exact``."""
      kwargs: dict[str, object] = {}
      if states_chunk_size is not None:
        kwargs["states_chunk_size"] = states_chunk_size
      return m.score_unconditional_state_vmap_exact(  # type: ignore[union-attr]
        prng_key,
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
        y_stack,
        y_t_stack,
        y_m_stack,
        state_flat_rows,
        n_flat,
        tie_group_map=tie_group_map,
        multi_state_strategy_idx=multi_state_strategy_idx,
        multi_state_temperature=jnp.asarray(multi_state_temperature, jnp.float32),
        state_weights=state_weights,
        state_mapping=state_mapping,
        **kwargs,
      )

    unconditional_stack.strategy_idx_from_str = strategy_idx  # type: ignore[attr-defined]
    return cast("StateVmapExactLogitsFn", unconditional_stack)

  if not isinstance(model, _PM):
    raise TypeError("Expected PrxteinMPNN or PrxteinLigandMPNN")

  @partial(jax.jit, static_argnames=("n_flat",))
  def unconditional_stack_prot(
    prng_key: jax.Array,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    tie_group_map: jax.Array | None,
    multi_state_strategy_idx: jax.Array,
    multi_state_temperature: jax.Array | float,
    state_weights: jax.Array | None,
    state_mapping: jax.Array | None,
  ) -> jax.Array:
    return m.score_unconditional_state_vmap_exact(
      prng_key,
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      state_flat_rows,
      n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      multi_state_temperature=jnp.asarray(multi_state_temperature, jnp.float32),
      state_weights=state_weights,
      state_mapping=state_mapping,
      inference=True,
    )

  unconditional_stack_prot.strategy_idx_from_str = strategy_idx  # type: ignore[attr-defined]
  return cast("StateVmapExactLogitsFn", unconditional_stack_prot)
