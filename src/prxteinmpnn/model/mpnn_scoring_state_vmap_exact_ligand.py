"""LigandMPNN ``state_vmap_exact`` scoring (unconditional + conditional stacks).

Extracted from :mod:`prxteinmpnn.model.ligand_mpnn` (Phase 5e-cont) to shrink ``ligand_mpnn.py`` without
changing JIT boundaries or public method names on the model class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from prxteinmpnn.model._shared import apply_multistate_to_all_logits
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack, scatter_stack_to_flat

if TYPE_CHECKING:
  from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
  from prxteinmpnn.utils.types import (
    Float,
    Int,
    Logits,
    PRNGKeyArray,
    TieGroupMap,
  )


def _ligand_slice_pad_state_batch(
  s0: int,
  states_chunk_size: int,
  s_tot: int,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  state_flat_rows: jax.Array,
) -> tuple[jax.Array, ...]:
  """Slice ``[s0, s0+cs)`` along state axis and pad to length ``states_chunk_size``.

  Padded rows: ``mask=0``, ``state_flat_rows=-1`` (scatter skips). Other tensors tile the
  last real row so ligand feature shapes stay valid under ``ligand_l_chunk`` tiling.
  """
  cs = int(states_chunk_size)
  s1 = min(s0 + cs, s_tot)
  n_real = s1 - s0

  def pad_first_axis(a: jax.Array) -> jax.Array:
    slab = a[s0:s1]
    if n_real >= cs:
      return slab
    pad_n = cs - n_real
    last = slab[-1:]
    tail = jnp.tile(last, (pad_n,) + (1,) * (slab.ndim - 1))
    return jnp.concatenate([slab, tail], axis=0)

  rows = state_flat_rows[s0:s1]
  if n_real < cs:
    rows = jnp.concatenate(
      [rows, jnp.full((cs - n_real, rows.shape[1]), -1, dtype=rows.dtype)],
      axis=0,
    )
  m = mask_stack[s0:s1]
  if n_real < cs:
    m = jnp.concatenate([m, jnp.zeros((cs - n_real, m.shape[1]), dtype=m.dtype)], axis=0)

  return (
    pad_first_axis(coords_stack),
    m,
    pad_first_axis(residue_index_stack),
    pad_first_axis(chain_index_stack),
    pad_first_axis(y_stack),
    pad_first_axis(y_t_stack),
    pad_first_axis(y_m_stack),
    rows,
  )


def _ligand_slice_pad_cond_batch(
  s0: int,
  states_chunk_size: int,
  s_tot: int,
  seq_oh_stack: jax.Array,
  ar_mask_stack: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  cs = int(states_chunk_size)
  s1 = min(s0 + cs, s_tot)
  n_real = s1 - s0

  def pad_first_axis(a: jax.Array) -> jax.Array:
    slab = a[s0:s1]
    if n_real >= cs:
      return slab
    pad_n = cs - n_real
    last = slab[-1:]
    tail = jnp.tile(last, (pad_n,) + (1,) * (slab.ndim - 1))
    return jnp.concatenate([slab, tail], axis=0)

  arm = ar_mask_stack[s0:s1]
  if n_real < cs:
    arm = jnp.concatenate(
      [arm, jnp.zeros((cs - n_real,) + arm.shape[1:], dtype=arm.dtype)],
      axis=0,
    )
  return pad_first_axis(seq_oh_stack), arm


def ligand_score_unconditional_state_vmap_one_chunk(
  model: PrxteinLigandMPNN,
  _k_enc: jax.Array,
  k_feat: jax.Array,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
  *,
  inference: bool = True,
) -> jax.Array:
  """Single fixed-size state batch (padded to leading dim of ``coords_stack``); scatter only, no fuse."""
  from prxteinmpnn.model.ligand_mpnn import ligand_encode_stack_row  # noqa: PLC0415

  def enc_one(coords, ma, ri, ci, yy, yt, ym):
    return ligand_encode_stack_row(
      model, coords, ma, ri, ci, yy, yt, ym, k_feat, inference=inference,
    )

  node_b, edge_b, nei_b = jax.vmap(enc_one)(
    coords_stack,
    mask_stack,
    residue_index_stack,
    chain_index_stack,
    y_stack,
    y_t_stack,
    y_m_stack,
  )

  def decode_one(nb, eb, nei, mk):
    return model.decoder(nb, eb, nei, mk, key=None)

  decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, mask_stack)
  logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
  return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)


def ligand_score_conditional_state_vmap_one_chunk(
  model: PrxteinLigandMPNN,
  k_enc: jax.Array,
  k_feat: jax.Array,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  seq_oh_stack: jax.Array,
  ar_mask_stack: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
  bias_flat: jax.Array | None,
  inference: bool,
) -> jax.Array:
  """Conditional path for one fixed-size state batch; scatter only, no fuse."""
  from prxteinmpnn.model.ligand_mpnn import ligand_encode_stack_row  # noqa: PLC0415

  def enc_row(coords, ma, ri, ci, yy, yt, ym):
    return ligand_encode_stack_row(
      model, coords, ma, ri, ci, yy, yt, ym, k_feat, inference=inference,
    )

  node_b, edge_b, nei_b = jax.vmap(enc_row)(
    coords_stack,
    mask_stack,
    residue_index_stack,
    chain_index_stack,
    y_stack,
    y_t_stack,
    y_m_stack,
  )

  def dec_one(nb, eb, nei, mk, arm, oh):
    return model.decoder.call_conditional(
      nb,
      eb,
      nei,
      mk,
      arm,
      oh,
      model.w_s_embed.weight,
      inference=inference,
      key=k_enc,
    )

  decoded = jax.vmap(dec_one)(
    node_b,
    edge_b,
    nei_b,
    mask_stack,
    ar_mask_stack,
    seq_oh_stack,
  )

  logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
  if bias_flat is not None:
    logits_s = logits_s + gather_flat_to_stack(bias_flat, state_flat_rows)
  return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)


def run_score_unconditional_state_vmap_exact_ligand(
  model: PrxteinLigandMPNN,
  prng_key: PRNGKeyArray,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
  *,
  tie_group_map: TieGroupMap | None,
  multi_state_strategy_idx: Int,
  multi_state_temperature: Float | float,
  state_weights: jnp.ndarray | None,
  state_mapping: jnp.ndarray | None,
  inference: bool = True,
  states_chunk_size: int | None = None,
) -> Logits:
  """LigandMPNN unconditional logits: per-row encode + decoder, scatter+fuse (``state_vmap_exact``)."""
  from prxteinmpnn.model.ligand_mpnn import ligand_encode_stack_row  # noqa: PLC0415

  k_enc, k_feat = jax.random.split(prng_key)
  s_tot = int(coords_stack.shape[0])
  scs = int(states_chunk_size) if states_chunk_size is not None else 0
  log_dim = int(model.w_out.out_features)
  log_dtype = model.w_out.weight.dtype

  def _one_shot() -> jax.Array:
    def enc_one(coords, ma, ri, ci, yy, yt, ym):
      return ligand_encode_stack_row(
        model, coords, ma, ri, ci, yy, yt, ym, k_feat, inference=inference,
      )

    node_b, edge_b, nei_b = jax.vmap(enc_one)(
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      y_stack,
      y_t_stack,
      y_m_stack,
    )

    def decode_one(nb, eb, nei, mk):
      # Match :meth:`PrxteinLigandMPNN._call_unconditional`: ``key=None`` disables decoder dropout
      # ( unconditional ``__call__`` passes ``None`` for the decoder PRNG slot ).
      return model.decoder(nb, eb, nei, mk, key=None)

    decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, mask_stack)
    logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
    return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)

  if scs <= 0 or scs >= s_tot:
    logits_flat = _one_shot()
  else:
    logits_flat = jnp.zeros((n_flat, log_dim), dtype=log_dtype)
    for s0 in range(0, s_tot, scs):
      c_coords, c_mask, c_ri, c_ci, c_y, c_yt, c_ym, c_rows = _ligand_slice_pad_state_batch(
        s0,
        scs,
        s_tot,
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
        y_stack,
        y_t_stack,
        y_m_stack,
        state_flat_rows,
      )
      logits_flat = logits_flat + ligand_score_unconditional_state_vmap_one_chunk(
        model,
        k_enc,
        k_feat,
        c_coords,
        c_mask,
        c_ri,
        c_ci,
        c_y,
        c_yt,
        c_ym,
        c_rows,
        n_flat,
        inference=inference,
      )

  if tie_group_map is not None:
    logits_flat = apply_multistate_to_all_logits(
      logits_flat,
      tie_group_map,
      multi_state_strategy_idx,
      jnp.asarray(multi_state_temperature, jnp.float32),
      state_weights,
      state_mapping,
    )
  return logits_flat


def run_score_conditional_state_vmap_exact_ligand(
  model: PrxteinLigandMPNN,
  prng_key: PRNGKeyArray,
  coords_stack: jax.Array,
  mask_stack: jax.Array,
  residue_index_stack: jax.Array,
  chain_index_stack: jax.Array,
  y_stack: jax.Array,
  y_t_stack: jax.Array,
  y_m_stack: jax.Array,
  seq_oh_stack: jax.Array,
  ar_mask_stack: jax.Array,
  state_flat_rows: jax.Array,
  n_flat: int,
  *,
  tie_group_map: TieGroupMap | None,
  multi_state_strategy_idx: Int,
  multi_state_temperature: Float | float,
  state_weights: jnp.ndarray | None,
  state_mapping: jnp.ndarray | None,
  bias_flat: jax.Array | None = None,
  inference: bool = True,
  states_chunk_size: int | None = None,
) -> Logits:
  """LigandMPNN stacked conditional logits; optional ``bias_flat`` added before fuse."""
  from prxteinmpnn.model.ligand_mpnn import ligand_encode_stack_row  # noqa: PLC0415

  k_enc, k_feat = jax.random.split(prng_key)
  s_tot = int(coords_stack.shape[0])
  scs = int(states_chunk_size) if states_chunk_size is not None else 0
  log_dim = int(model.w_out.out_features)
  log_dtype = model.w_out.weight.dtype

  def _one_shot() -> jax.Array:
    def enc_row(coords, ma, ri, ci, yy, yt, ym):
      return ligand_encode_stack_row(
        model, coords, ma, ri, ci, yy, yt, ym, k_feat, inference=inference,
      )

    node_b, edge_b, nei_b = jax.vmap(enc_row)(
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      y_stack,
      y_t_stack,
      y_m_stack,
    )

    def dec_one(nb, eb, nei, mk, arm, oh):
      return model.decoder.call_conditional(
        nb,
        eb,
        nei,
        mk,
        arm,
        oh,
        model.w_s_embed.weight,
        inference=inference,
        key=k_enc,
      )

    decoded = jax.vmap(dec_one)(
      node_b,
      edge_b,
      nei_b,
      mask_stack,
      ar_mask_stack,
      seq_oh_stack,
    )

    logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
    if bias_flat is not None:
      logits_s = logits_s + gather_flat_to_stack(bias_flat, state_flat_rows)

    return scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)

  if scs <= 0 or scs >= s_tot:
    logits_flat = _one_shot()
  else:
    logits_flat = jnp.zeros((n_flat, log_dim), dtype=log_dtype)
    for s0 in range(0, s_tot, scs):
      c_coords, c_mask, c_ri, c_ci, c_y, c_yt, c_ym, c_rows = _ligand_slice_pad_state_batch(
        s0,
        scs,
        s_tot,
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
        y_stack,
        y_t_stack,
        y_m_stack,
        state_flat_rows,
      )
      c_oh, c_arm = _ligand_slice_pad_cond_batch(s0, scs, s_tot, seq_oh_stack, ar_mask_stack)
      logits_flat = logits_flat + ligand_score_conditional_state_vmap_one_chunk(
        model,
        k_enc,
        k_feat,
        c_coords,
        c_mask,
        c_ri,
        c_ci,
        c_y,
        c_yt,
        c_ym,
        c_oh,
        c_arm,
        c_rows,
        n_flat,
        bias_flat,
        inference,
      )

  if tie_group_map is not None:
    logits_flat = apply_multistate_to_all_logits(
      logits_flat,
      tie_group_map,
      multi_state_strategy_idx,
      jnp.asarray(multi_state_temperature, jnp.float32),
      state_weights,
      state_mapping,
    )
  return logits_flat
