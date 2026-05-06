"""Host-side helpers to pack per-state tensors for ``state_vmap_exact`` sampling."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.payloads import MultistateStackPayload

MPNN_UNK_TOKEN = 20


def pad_length_bucket_128(n: int) -> int:
  return ((int(n) + 127) // 128) * 128


def state_segment_lengths(
  n_states: int,
  n_canonical: int,
  peptide_lengths: np.ndarray,
  with_ligand: bool,
) -> list[int]:
  """Residue count per state (canonical + optional peptide) before global bucketing."""
  lengths: list[int] = []
  for s in range(n_states):
    n_loc = int(n_canonical)
    if not with_ligand:
      n_loc += int(peptide_lengths[s])
    lengths.append(n_loc)
  return lengths


def state_flat_row_offsets(lengths: list[int]) -> np.ndarray:
  """Cumulative flat offsets (length S+1) for the unpadded concatenated multistate graph."""
  off = np.zeros(len(lengths) + 1, dtype=np.int64)
  for i, ln in enumerate(lengths):
    off[i + 1] = off[i] + int(ln)
  return off


def build_state_vmap_exact_stacks(
  *,
  n_states: int,
  n_canonical: int,
  peptide_lengths: np.ndarray,
  with_ligand: bool,
  ca_states_4: np.ndarray,
  peptide_bb: np.ndarray | None,
  base_fixed_mask: np.ndarray,
  aatype_states: np.ndarray,
  af_to_mpnn: Callable[[np.ndarray], np.ndarray],
) -> dict[str, np.ndarray]:
  """Pack per-state graphs to shape ``(S, n_pad, ...)`` matching the flat multistate layout.

  Returns numpy arrays suitable for ``jnp.array`` +
  :py:meth:`PrxteinMPNN.sample_autoregressive_state_vmap_exact` or, with ligand tensors, for
  :py:meth:`prxteinmpnn.model.mpnn.PrxteinLigandMPNN.sample_autoregressive_state_vmap_exact` after
  :py:func:`slice_flat_tensor_to_stack` on ``Y`` / ``Y_t`` / ``Y_m``.
  """
  lengths = state_segment_lengths(n_states, n_canonical, peptide_lengths, with_ligand)
  n_pad = pad_length_bucket_128(max(lengths))
  coords = np.zeros((n_states, n_pad, 4, 3), dtype=np.float32)
  mask = np.zeros((n_states, n_pad), dtype=np.float32)
  res_i = np.zeros((n_states, n_pad), dtype=np.int32)
  chain = np.zeros((n_states, n_pad), dtype=np.int32)
  tie = np.zeros((n_states, n_pad), dtype=np.int32)
  fixed_m = np.zeros((n_states, n_pad), dtype=np.float32)
  fixed_t = np.zeros((n_states, n_pad), dtype=np.int32)
  rows = np.full((n_states, n_pad), -1, dtype=np.int32)
  offsets = state_flat_row_offsets(lengths)

  for s in range(n_states):
    ln = lengths[s]
    rows[s, :ln] = np.arange(int(offsets[s]), int(offsets[s + 1]), dtype=np.int32)
    coords[s, :n_canonical] = ca_states_4[s]
    mask[s, :n_canonical] = 1.0
    res_i[s, :n_canonical] = np.arange(n_canonical, dtype=np.int32)
    chain[s, :n_canonical] = 0
    tie[s, :n_canonical] = np.arange(n_canonical, dtype=np.int32)
    fixed_m[s, :n_canonical] = base_fixed_mask.astype(np.float32)
    fixed_t[s, :n_canonical] = np.array(af_to_mpnn(aatype_states[s]), dtype=np.int32)
    pos = n_canonical
    if not with_ligand and peptide_lengths is not None:
      l_pep = int(peptide_lengths[s])
      if l_pep > 0 and peptide_bb is not None:
        coords[s, pos : pos + l_pep] = peptide_bb[s, :l_pep]
        mask[s, pos : pos + l_pep] = 1.0
        res_i[s, pos : pos + l_pep] = np.arange(l_pep, dtype=np.int32) + 500
        chain[s, pos : pos + l_pep] = 1
        tie[s, pos : pos + l_pep] = np.arange(l_pep, dtype=np.int32) + 1000 + s * 100
        fixed_m[s, pos : pos + l_pep] = 1.0
        fixed_t[s, pos : pos + l_pep] = MPNN_UNK_TOKEN

  return {
    "coords_stack": coords,
    "mask_stack": mask,
    "residue_index_stack": res_i,
    "chain_index_stack": chain,
    "tie_group_map_stack": tie,
    "fixed_mask_stack": fixed_m,
    "fixed_tokens_stack": fixed_t,
    "state_flat_rows": rows,
    "flat_row_offsets": offsets.astype(np.int64),
  }


def multistate_stack_payload_from_loose_ar_host(
  *,
  coords_stack: jnp.ndarray,
  mask_stack: jnp.ndarray,
  residue_index_stack: jnp.ndarray,
  chain_index_stack: jnp.ndarray,
  tie_group_map_stack: jnp.ndarray,
  fixed_mask_stack: jnp.ndarray,
  fixed_tokens_stack: jnp.ndarray,
  state_flat_rows: jnp.ndarray,
  n_canonical: int,
) -> MultistateStackPayload:
  """Build :class:`~prxteinmpnn.payloads.MultistateStackPayload` on host for AR ``state_vmap_exact``.

  Callers that pass loose ``coords_stack`` / ``state_flat_rows=`` kwargs into the jitted sampler
  should be normalized **before** ``jax.jit`` so static ``n_flat`` / ``flat_row_offsets`` match
  :func:`build_state_vmap_exact_stacks` invariants (lengths from consecutive flat spans per
  state row).

  Uses ``jax.device_get`` on ``state_flat_rows`` only (small int32 metadata).
  """
  rows_np = np.asarray(jax.device_get(state_flat_rows))
  n_states = int(rows_np.shape[0])
  lengths: list[int] = []
  for s in range(n_states):
    row = rows_np[s]
    v = row[row >= 0]
    lengths.append(int(v.max() - v.min() + 1) if v.size > 0 else 0)
  offsets = state_flat_row_offsets(lengths)
  n_flat = int(offsets[-1])
  return MultistateStackPayload(
    coords_stack=coords_stack,
    mask_stack=mask_stack,
    residue_index_stack=residue_index_stack,
    chain_index_stack=chain_index_stack,
    tie_group_map_stack=tie_group_map_stack,
    fixed_mask_stack=fixed_mask_stack,
    fixed_tokens_stack=fixed_tokens_stack,
    state_flat_rows=state_flat_rows,
    flat_row_offsets=jnp.asarray(offsets, dtype=jnp.int32),
    n_states=n_states,
    n_canonical=n_canonical,
    n_flat=n_flat,
  )


def multistate_stack_payload_from_prep_numpy(
  sv: Mapping[str, np.ndarray],
  *,
  n_states: int,
  n_canonical: int,
) -> MultistateStackPayload:
  """Convert :func:`build_state_vmap_exact_stacks` numpy output to a JAX :class:`MultistateStackPayload`."""
  n_flat = int(np.asarray(sv["flat_row_offsets"])[-1])
  return MultistateStackPayload(
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    tie_group_map_stack=jnp.asarray(sv["tie_group_map_stack"], dtype=jnp.int32),
    fixed_mask_stack=jnp.asarray(sv["fixed_mask_stack"], dtype=jnp.float32),
    fixed_tokens_stack=jnp.asarray(sv["fixed_tokens_stack"], dtype=jnp.int32),
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    flat_row_offsets=jnp.asarray(sv["flat_row_offsets"], dtype=jnp.int32),
    n_states=n_states,
    n_canonical=n_canonical,
    n_flat=n_flat,
  )


def slice_flat_bias_to_stack(
  bias_flat: np.ndarray | None,
  state_flat_rows: np.ndarray,
  n_states: int,
  n_pad: int,
) -> np.ndarray:
  """Map optional flat ``(N_flat, 21)`` bias to ``(S, n_pad, 21)`` using ``state_flat_rows``."""
  out = np.zeros((n_states, n_pad, int(bias_flat.shape[-1]) if bias_flat is not None else 21), dtype=np.float32)
  if bias_flat is None:
    return out
  n_dim = int(bias_flat.shape[-1])
  out = np.zeros((n_states, n_pad, n_dim), dtype=np.float32)
  for s in range(n_states):
    for j in range(n_pad):
      fr = int(state_flat_rows[s, j])
      if fr < 0:
        continue
      if fr < bias_flat.shape[0]:
        out[s, j] = bias_flat[fr]
  return out


def remap_wave_positions_flat_to_local(
  wave_pos: np.ndarray,
  wave_pv: np.ndarray,
  offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Map flat super-graph indices in ``wave_pos`` to **local** stacked-graph indices.

  ``offsets`` is length ``S+1`` from :py:func:`state_flat_row_offsets` (unpadded cumulative starts).
  """
  out = np.zeros_like(wave_pos, dtype=np.int32)
  out_v = np.array(wave_pv, dtype=bool, copy=True)
  n_w, m_w, g_w = wave_pos.shape
  for w in range(n_w):
    for m in range(m_w):
      locals_this: list[int] = []
      for g in range(g_w):
        if not wave_pv[w, m, g]:
          continue
        flat = int(wave_pos[w, m, g])
        s_idx = int(np.searchsorted(offsets, flat, side="right") - 1)
        if s_idx < 0 or s_idx >= len(offsets) - 1 or flat < int(offsets[s_idx]) or flat >= int(offsets[s_idx + 1]):
          out_v[w, m, g] = False
          continue
        local = int(flat - offsets[s_idx])
        out[w, m, g] = local
        locals_this.append(local)
      if len(locals_this) > 1 and len(set(locals_this)) > 1:
        msg = (
          "Wave slot has inconsistent local indices across members — "
          "state_vmap_exact stacked decode requires aligned per-state layouts "
          f"at wave=({w},{m}) locals={locals_this}"
        )
        raise ValueError(msg)
  return out, out_v


def slice_flat_tensor_to_stack(
  flat: np.ndarray,
  state_flat_rows: np.ndarray,
  n_states: int,
  n_pad: int,
) -> np.ndarray:
  """Map a **flat** layout ``(N, ...)`` to the stacked state layout ``(S, n_pad, ...)``.

  For each (state, local column) the flat row index is ``state_flat_rows[s, j]``;
  out-of-range or negative indices are left zero-filled (padding / invalid).
  """
  if flat.size == 0 or flat.shape[0] == 0:
    return np.zeros((n_states, n_pad, 0), dtype=flat.dtype)
  n_flat = int(flat.shape[0])
  tail = flat.shape[1:]
  out = np.zeros((n_states, n_pad) + tail, dtype=flat.dtype)
  for s in range(n_states):
    for j in range(n_pad):
      fr = int(state_flat_rows[s, j])
      if fr < 0 or fr >= n_flat:
        continue
      out[s, j, ...] = flat[fr]
  return out
