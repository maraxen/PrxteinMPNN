"""Factory for creating unconditional logits functions.

Unconditional logits are computed without providing a sequence input,
allowing the model to predict the most likely amino acids at each position
based solely on the structure.

Dense single-graph unconditional logits: :func:`make_unconditional_logits_fn` accepts
:class:`~prxteinmpnn.model.mpnn.PrxteinMPNN` only (backbone); ligand single-graph paths use the
ligand ``model.__call__`` API directly.

Multistate parallel encode (``state_vmap_exact``): use
:func:`make_unconditional_logits_state_vmap_fn`, or call ``model(...,
decoding_approach=\"unconditional\", multistate_mode=\"state_vmap_exact\", ...)``
with ``coords_stack`` / ``mask_stack`` / ``state_flat_rows`` / ``n_flat`` (and
ligand ``y_*_stack`` when applicable).

For a :class:`~prxteinmpnn.bundles.ProteinBundle` carrier, prefer
:func:`prxteinmpnn.sampling.state_vmap_payload_logits.unconditional_state_vmap_logits_from_payload`
or :meth:`prxteinmpnn.model.mpnn.PrxteinMPNN.score_unconditional_from_payload`
or :meth:`~prxteinmpnn.model.mpnn.PrxteinLigandMPNN.score_unconditional_from_payload`.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.bundles import LigandBundle, ProteinBundle
from prxteinmpnn.protocols import ModelProtocol, StateVmapExactLogitsFn, UnconditionalLogitsFn  # noqa: TC001
from prxteinmpnn.sampling.state_vmap_payload_logits import (
  unconditional_state_vmap_logits_from_payload,
)


def _multistate_payload_for_unconditional_vmap(
  coords: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
) -> ProteinBundle:
  """Build a :class:`~prxteinmpnn.bundles.ProteinBundle` for unconditional vmap scoring.

  Fields not read by :meth:`~prxteinmpnn.model.mpnn.PrxteinMPNN.score_unconditional_from_payload`
  are zero-filled with broadcastable shapes; ``n_canonical`` is set to the padded width for static typing.
  """
  s_dim, p_dim = coords.shape[0], coords.shape[1]
  zeros_i = jnp.zeros((s_dim, p_dim), dtype=jnp.int32)
  zeros_f = jnp.zeros((s_dim, p_dim), dtype=jnp.float32)
  offs = jnp.zeros((s_dim + 1,), dtype=jnp.int32)
  return cast(
    "ProteinBundle",
    ProteinBundle(
      coords=coords,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      tie_group_map=zeros_i,
      fixed_mask=zeros_f,
      fixed_tokens=zeros_i,
      state_flat_rows=state_flat_rows,
      flat_row_offsets=offs,
      state_index=jnp.arange(s_dim, dtype=jnp.int32),
      state_embedding=jnp.zeros((s_dim, 1), dtype=jnp.float32),
      n_states=int(s_dim),
      n_canonical=int(p_dim),
      n_flat=int(n_flat),
    ),
  )


def make_unconditional_logits_fn(
  model: ModelProtocol,
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
    bn = jnp.asarray(0.0, dtype=jnp.float32) if backbone_noise is None else backbone_noise
    _, logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="unconditional",
      ar_mask=ar_mask,
      backbone_noise=bn,
    )
    return cast("jax.Array", logits)

  return cast("UnconditionalLogitsFn", unconditional_logits)


def make_unconditional_logits_state_vmap_fn(
  model: ModelProtocol,
) -> StateVmapExactLogitsFn:
  """JIT ``score_unconditional``: stacked encode → scattered flat logits + fuse."""
  m = eqx.nn.inference_mode(model, value=True) if isinstance(model, eqx.Module) else model
  is_lig = model.capabilities.is_ligand_model

  if is_lig:

    def unconditional_stack(
      prng_key: jax.Array,
      coords: jax.Array,
      mask: jax.Array,
      residue_index: jax.Array,
      chain_index: jax.Array,
      y: jax.Array,
      y_t: jax.Array,
      y_m: jax.Array,
      state_flat_rows: jax.Array,
      n_flat: int,
      tie_group_map: jax.Array | None,
      multi_state_strategy_idx: jax.Array,
      state_weights: jax.Array | None,
      state_mapping: jax.Array | None,
      states_chunk_size: int | None = None,
    ) -> jax.Array:
      """No outer ``jax.jit``: host state-chunk loop inside ``score_unconditional``."""
      stack = _multistate_payload_for_unconditional_vmap(
        coords,
        mask,
        residue_index,
        chain_index,
        state_flat_rows,
        n_flat,
      )
      lig = cast(
        "LigandBundle",
        LigandBundle(y=y, y_t=y_t, y_m=y_m),
      )
      return unconditional_state_vmap_logits_from_payload(
        m,
        prng_key,
        stack,
        lig,
        tie_group_map=tie_group_map,
        multi_state_strategy_idx=multi_state_strategy_idx,

        state_weights=state_weights,
        state_mapping=state_mapping,
        states_chunk_size=states_chunk_size,
      )

    return cast("StateVmapExactLogitsFn", unconditional_stack)

  if not isinstance(model, ModelProtocol):
    raise TypeError("Expected a ModelProtocol-conforming model")

  @partial(jax.jit, static_argnames=("n_flat",))
  def unconditional_stack_prot(
    prng_key: jax.Array,
    coords: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    tie_group_map: jax.Array | None,
    multi_state_strategy_idx: jax.Array,
    state_weights: jax.Array | None,
    state_mapping: jax.Array | None,
  ) -> jax.Array:
    stack = _multistate_payload_for_unconditional_vmap(
      coords,
      mask,
      residue_index,
      chain_index,
      state_flat_rows,
      n_flat,
    )
    return unconditional_state_vmap_logits_from_payload(
      m,
      prng_key,
      stack,
      None,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
    )

  return cast("StateVmapExactLogitsFn", unconditional_stack_prot)
