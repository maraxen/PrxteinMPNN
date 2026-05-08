"""Tests for ``state_vmap_exact`` host prep and JIT sampler wiring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.payloads import LigandStack, MultistateStackPayload
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.sampling.state_vmap_prep import (
  build_state_vmap_exact_stacks,
  multistate_stack_payload_from_prep_numpy,
  remap_wave_positions_flat_to_local,
  slice_flat_tensor_to_stack,
  state_flat_row_offsets,
  state_segment_lengths,
)
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import random_decoding_order
from prxteinmpnn.utils.wave_parallel import compute_wave_assignments


def _identity_af_to_mpnn(x: np.ndarray) -> np.ndarray:
  return np.asarray(x, dtype=np.int32)


def test_segment_lengths_and_offsets():
  lens = state_segment_lengths(2, 5, np.array([0, 2], dtype=np.int32), with_ligand=False)
  assert lens == [5, 7]
  off = state_flat_row_offsets(lens)
  assert np.array_equal(off, [0, 5, 12])


def test_remap_wave_flat_to_local_agrees_across_states():
  n_can, n_states = 4, 3
  lengths = [n_can] * n_states
  offsets = state_flat_row_offsets(lengths)
  num_groups = n_can
  max_group_size = n_states
  git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_group_size), dtype=bool)
  for g in range(n_can):
    for s in range(n_states):
      git[g, s] = int(offsets[s] + g)
      gvt[g, s] = True
  ca = np.zeros((n_can, 4, 3), dtype=np.float32)
  tie = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  _w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca, tie, git, gvt, k_neighbors=4, n_canonical=n_can)
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, offsets)
  assert w_loc.shape == w_pos.shape
  assert bool(np.all(w_pv2 == w_pv))
  assert np.all((w_loc >= 0) & (w_loc < n_can) | ~w_pv2)


def test_build_stacks_and_smoke_sample_autoregressive_state_vmap_exact():
  n_states, n_can = 2, 6
  key = jax.random.PRNGKey(0)
  model = PrxteinMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=key,
  )
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = jax.random.normal(key, (n_states, n_can, 4, 3)).astype(jnp.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=True,
    ca_states_4=np.asarray(ca),
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  n_pad = sv["coords_stack"].shape[1]
  num_groups = n_can
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  max_group_size = n_states
  git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_group_size), dtype=bool)
  for g in range(n_can):
    for s in range(n_states):
      git[g, s] = int(sv["flat_row_offsets"][s] + g)
      gvt[g, s] = True
  ca0 = np.asarray(ca[0], dtype=np.float32)
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=8, n_canonical=n_can)
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, sv["flat_row_offsets"])

  order_key, _ = jax.random.split(key)
  decoding_order, _ = random_decoding_order(order_key, n_states * n_can, None, None)
  ar_flat = generate_ar_mask(decoding_order, None, jnp.asarray(tie_flat), num_groups)

  def _ar_submatrix(rows: jnp.ndarray) -> jnp.ndarray:
    g = jnp.where(rows >= 0, rows, 0)
    sub = ar_flat[g[:, None], g[None, :]]
    valid = rows >= 0
    vm = valid[:, None] & valid[None, :]
    return jnp.where(vm, sub, jnp.zeros_like(sub))

  rows = jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32)
  ar_stack = jax.vmap(_ar_submatrix)(rows)

  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  seq, logits = model.sample_autoregressive_state_vmap_exact(
    jax.random.fold_in(key, 1),
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    ar_stack,
    jnp.asarray(sv["tie_group_map_stack"], dtype=jnp.int32),
    jnp.zeros((n_states, n_pad, 21), dtype=jnp.float32),
    jnp.array(1.0, dtype=jnp.float32),
    jnp.int32(0),
    sw,
    jnp.asarray(sv["fixed_mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["fixed_tokens_stack"], dtype=jnp.int32),
    jnp.asarray(w_id, dtype=jnp.int32),
    jnp.asarray(w_loc, dtype=jnp.int32),
    jnp.asarray(w_gv, dtype=bool),
    jnp.asarray(w_pv2, dtype=bool),
  )
  assert seq.shape[0] == n_states
  assert logits.shape == seq.shape


def test_make_sample_sequences_state_vmap_exact_smoke():
  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(2)
  model = PrxteinMPNN(
    node_features=24,
    edge_features=24,
    hidden_features=24,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    key=key,
  )
  sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = jax.random.normal(key, (n_states, n_can, 4, 3)).astype(jnp.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=True,
    ca_states_4=np.asarray(ca),
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  n_pad = sv["coords_stack"].shape[1]
  bucket = n_states * n_can
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  num_groups = n_can
  max_group_size = n_states
  git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_group_size), dtype=bool)
  for g in range(n_can):
    for s in range(n_states):
      git[g, s] = int(sv["flat_row_offsets"][s] + g)
      gvt[g, s] = True
  ca0 = np.asarray(ca[0], dtype=np.float32)
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=6, n_canonical=n_can)
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, sv["flat_row_offsets"])

  coords_flat = jnp.zeros((bucket, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((bucket,), dtype=jnp.float32)
  res_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  chain_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  sm_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(ca[s])
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  seq, logits, _ = sample_fn(
    jax.random.fold_in(key, 9),
    coords_flat,
    mask_flat,
    res_flat,
    chain_flat,
    tie_group_map=jnp.asarray(tie_flat, dtype=jnp.int32),
    num_groups=num_groups,
    multistate_mode="state_vmap_exact",
    multi_state_strategy="arithmetic_mean",
    structure_mapping=sm_flat,
    temperature=jnp.float32(0.2),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states),
    fixed_mask=jnp.ones((bucket,), dtype=jnp.float32),
    fixed_tokens=jnp.zeros((bucket,), dtype=jnp.int32),
    group_indices_table=jnp.asarray(git, dtype=jnp.int32),
    group_valid_table=jnp.asarray(gvt, dtype=bool),
    wave_group_ids=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions=jnp.asarray(w_pos, dtype=jnp.int32),
    wave_group_valid=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid=jnp.asarray(w_pv, dtype=bool),
    state_vmap_n_canonical=n_can,
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    tie_group_map_stack=jnp.asarray(sv["tie_group_map_stack"], dtype=jnp.int32),
    fixed_mask_stack=jnp.asarray(sv["fixed_mask_stack"], dtype=jnp.float32),
    fixed_tokens_stack=jnp.asarray(sv["fixed_tokens_stack"], dtype=jnp.int32),
    bias_stack=None,
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    wave_group_ids_local=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions_local=jnp.asarray(w_loc, dtype=jnp.int32),
    wave_group_valid_local=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid_local=jnp.asarray(w_pv2, dtype=bool),
  )
  assert seq.shape == (n_can,)
  assert logits.shape == (n_can, 21)


def test_sample_fn_state_vmap_exact_multistate_stack_payload_path():
  """``multistate_stack=`` matches unpacked stack kwargs (protein)."""
  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(44)
  model = PrxteinMPNN(
    node_features=24,
    edge_features=24,
    hidden_features=24,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    key=key,
  )
  sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = jax.random.normal(key, (n_states, n_can, 4, 3)).astype(jnp.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=True,
    ca_states_4=np.asarray(ca),
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  pay = multistate_stack_payload_from_prep_numpy(sv, n_states=n_states, n_canonical=n_can)
  assert isinstance(pay, MultistateStackPayload)
  bucket = n_states * n_can
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  num_groups = n_can
  max_group_size = n_states
  git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_group_size), dtype=bool)
  for g in range(n_can):
    for s in range(n_states):
      git[g, s] = int(sv["flat_row_offsets"][s] + g)
      gvt[g, s] = True
  ca0 = np.asarray(ca[0], dtype=np.float32)
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=6, n_canonical=n_can)
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, sv["flat_row_offsets"])

  coords_flat = jnp.zeros((bucket, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((bucket,), dtype=jnp.float32)
  res_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  chain_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  sm_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(ca[s])
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  common = dict(
    prng_key=jax.random.fold_in(key, 9),
    structure_coordinates=coords_flat,
    mask=mask_flat,
    residue_index=res_flat,
    chain_index=chain_flat,
    tie_group_map=jnp.asarray(tie_flat, dtype=jnp.int32),
    num_groups=num_groups,
    multistate_mode="state_vmap_exact",
    multi_state_strategy="arithmetic_mean",
    structure_mapping=sm_flat,
    temperature=jnp.float32(0.2),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states),
    fixed_mask=jnp.ones((bucket,), dtype=jnp.float32),
    fixed_tokens=jnp.zeros((bucket,), dtype=jnp.int32),
    group_indices_table=jnp.asarray(git, dtype=jnp.int32),
    group_valid_table=jnp.asarray(gvt, dtype=bool),
    wave_group_ids=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions=jnp.asarray(w_pos, dtype=jnp.int32),
    wave_group_valid=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid=jnp.asarray(w_pv, dtype=bool),
    state_vmap_n_canonical=n_can,
    wave_group_ids_local=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions_local=jnp.asarray(w_loc, dtype=jnp.int32),
    wave_group_valid_local=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid_local=jnp.asarray(w_pv2, dtype=bool),
  )
  seq_a, logits_a, _ = sample_fn(
    **common,
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    tie_group_map_stack=jnp.asarray(sv["tie_group_map_stack"], dtype=jnp.int32),
    fixed_mask_stack=jnp.asarray(sv["fixed_mask_stack"], dtype=jnp.float32),
    fixed_tokens_stack=jnp.asarray(sv["fixed_tokens_stack"], dtype=jnp.int32),
    bias_stack=None,
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
  )
  seq_b, logits_b, _ = sample_fn(
    **common,
    multistate_stack=pay,
    bias_stack=None,
  )
  np.testing.assert_array_equal(np.asarray(seq_a), np.asarray(seq_b))
  np.testing.assert_allclose(np.asarray(logits_a), np.asarray(logits_b), rtol=1e-5, atol=1e-5)


def test_sample_fn_state_vmap_exact_ligand_stack_payload_path():
  """``multistate_stack=`` + ``ligand_stack=`` matches unpacked stack kwargs."""
  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(45)
  rng = np.random.default_rng(3)
  model = PrxteinLigandMPNN(
    node_features=24,
    edge_features=24,
    hidden_features=24,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )
  sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=True,
    ca_states_4=ca,
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  pay = multistate_stack_payload_from_prep_numpy(sv, n_states=n_states, n_canonical=n_can)
  n_pad = int(sv["coords_stack"].shape[1])
  n_flat = int(sv["flat_row_offsets"][-1])
  n_atoms = 3
  yf = rng.normal(size=(n_flat, n_atoms, 3)).astype(np.float32)
  ytf = rng.integers(1, 20, size=(n_flat, n_atoms), dtype=np.int32)
  ymf = np.ones((n_flat, n_atoms), dtype=np.float32)
  y_st = slice_flat_tensor_to_stack(yf, sv["state_flat_rows"], n_states, n_pad)
  y_tst = slice_flat_tensor_to_stack(ytf, sv["state_flat_rows"], n_states, n_pad)
  y_mst = slice_flat_tensor_to_stack(ymf, sv["state_flat_rows"], n_states, n_pad)
  lig = LigandStack(
    y_stack=jnp.asarray(y_st, dtype=jnp.float32),
    y_t_stack=jnp.asarray(y_tst, dtype=jnp.int32),
    y_m_stack=jnp.asarray(y_mst, dtype=jnp.float32),
  )

  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  num_groups = n_can
  max_group_size = n_states
  git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_group_size), dtype=bool)
  for g in range(n_can):
    for s in range(n_states):
      git[g, s] = int(sv["flat_row_offsets"][s] + g)
      gvt[g, s] = True
  ca0 = ca[0]
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=6, n_canonical=n_can)
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, sv["flat_row_offsets"])
  bucket = n_flat
  coords_flat = jnp.zeros((bucket, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((bucket,), dtype=jnp.float32)
  res_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  chain_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  sm_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(jnp.asarray(ca[s]))
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  common = dict(
    prng_key=jax.random.fold_in(key, 12),
    structure_coordinates=coords_flat,
    mask=mask_flat,
    residue_index=res_flat,
    chain_index=chain_flat,
    tie_group_map=jnp.asarray(tie_flat, dtype=jnp.int32),
    num_groups=num_groups,
    multistate_mode="state_vmap_exact",
    multi_state_strategy="arithmetic_mean",
    structure_mapping=sm_flat,
    temperature=jnp.float32(0.2),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states),
    fixed_mask=jnp.ones((bucket,), dtype=jnp.float32),
    fixed_tokens=jnp.zeros((bucket,), dtype=jnp.int32),
    group_indices_table=jnp.asarray(git, dtype=jnp.int32),
    group_valid_table=jnp.asarray(gvt, dtype=bool),
    wave_group_ids=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions=jnp.asarray(w_pos, dtype=jnp.int32),
    wave_group_valid=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid=jnp.asarray(w_pv, dtype=bool),
    state_vmap_n_canonical=n_can,
    wave_group_ids_local=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions_local=jnp.asarray(w_loc, dtype=jnp.int32),
    wave_group_valid_local=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid_local=jnp.asarray(w_pv2, dtype=bool),
  )
  seq_a, logits_a, _ = sample_fn(
    **common,
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    tie_group_map_stack=jnp.asarray(sv["tie_group_map_stack"], dtype=jnp.int32),
    fixed_mask_stack=jnp.asarray(sv["fixed_mask_stack"], dtype=jnp.float32),
    fixed_tokens_stack=jnp.asarray(sv["fixed_tokens_stack"], dtype=jnp.int32),
    bias_stack=None,
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    y_stack=lig.y_stack,
    y_t_stack=lig.y_t_stack,
    y_m_stack=lig.y_m_stack,
  )
  seq_b, logits_b, _ = sample_fn(
    **common,
    multistate_stack=pay,
    ligand_stack=lig,
    bias_stack=None,
  )
  np.testing.assert_array_equal(np.asarray(seq_a), np.asarray(seq_b))
  np.testing.assert_allclose(np.asarray(logits_a), np.asarray(logits_b), rtol=1e-5, atol=1e-5)
  assert jnp.all(jnp.isfinite(logits_b))


def test_score_fn_state_vmap_exact_multistate_stack_payload():
  """``make_score_fn(..., state_vmap_exact)`` accepts ``multistate_stack=`` vs unpacked stacks."""
  from prxteinmpnn.scoring.score import make_score_fn

  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(77)
  model = PrxteinMPNN(
    node_features=24,
    edge_features=24,
    hidden_features=24,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    key=key,
  )
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  rng = np.random.default_rng(2)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=False,
    ca_states_4=ca,
    peptide_bb=None,
    base_fixed_mask=np.zeros(n_can, dtype=np.float32),
    aatype_states=np.zeros((n_states, n_can), dtype=np.int32),
    af_to_mpnn=lambda x: np.asarray(x, dtype=np.int32),
  )
  n_flat = int(sv["flat_row_offsets"][-1])
  pay = multistate_stack_payload_from_prep_numpy(sv, n_states=n_states, n_canonical=n_can)
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  off = 0
  for s in range(n_states):
    sm_flat = sm_flat.at[off : off + n_can].set(s)
    off += n_can
  seq_flat = jax.random.randint(jax.random.fold_in(key, 1), (n_flat,), 0, 20, dtype=jnp.int32)
  stack_kwargs = dict(
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat=n_flat,
    state_weights=sw,
  )
  score_fn = make_score_fn(model, multistate_mode="state_vmap_exact")
  sc_a, log_a, _ = score_fn(
    key,
    seq_flat,
    jnp.zeros((1, 4, 3)),
    jnp.zeros((n_flat,)),
    jnp.zeros((n_flat,), dtype=jnp.int32),
    jnp.zeros((n_flat,), dtype=jnp.int32),
    structure_mapping=sm_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy="arithmetic_mean",
    multi_state_temperature=jnp.float32(1.0),
    **stack_kwargs,
  )
  sc_b, log_b, _ = score_fn(
    key,
    seq_flat,
    jnp.zeros((1, 4, 3)),
    jnp.zeros((n_flat,)),
    jnp.zeros((n_flat,), dtype=jnp.int32),
    jnp.zeros((n_flat,), dtype=jnp.int32),
    structure_mapping=sm_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy="arithmetic_mean",
    multi_state_temperature=jnp.float32(1.0),
    multistate_stack=pay,
    state_weights=sw,
  )
  assert float(np.asarray(sc_a)) == pytest.approx(float(np.asarray(sc_b)), rel=1e-5, abs=1e-5)
  np.testing.assert_allclose(np.asarray(log_a), np.asarray(log_b), rtol=1e-5, atol=1e-5)


def test_ligand_state_vmap_exact_smoke():
  """``PrxteinLigandMPNN`` + ``state_vmap_exact`` (stacked Y) completes without error."""
  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(11)
  rng = np.random.default_rng(3)
  model = PrxteinLigandMPNN(
    node_features=24,
    edge_features=24,
    hidden_features=24,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )
  sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=True,
    ca_states_4=ca,
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  n_pad = int(sv["coords_stack"].shape[1])
  n_flat = int(sv["flat_row_offsets"][-1])
  n_atoms = 3
  yf = rng.normal(size=(n_flat, n_atoms, 3)).astype(np.float32)
  ytf = rng.integers(1, 20, size=(n_flat, n_atoms), dtype=np.int32)
  ymf = np.ones((n_flat, n_atoms), dtype=np.float32)
  y_st = slice_flat_tensor_to_stack(yf, sv["state_flat_rows"], n_states, n_pad)
  y_tst = slice_flat_tensor_to_stack(ytf, sv["state_flat_rows"], n_states, n_pad)
  y_mst = slice_flat_tensor_to_stack(ymf, sv["state_flat_rows"], n_states, n_pad)
  assert y_st.shape == (n_states, n_pad, n_atoms, 3)
  assert y_tst.shape == (n_states, n_pad, n_atoms)
  assert y_mst.shape == (n_states, n_pad, n_atoms)

  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  num_groups = n_can
  max_group_size = n_states
  git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
  gvt = np.zeros((num_groups, max_group_size), dtype=bool)
  for g in range(n_can):
    for s in range(n_states):
      git[g, s] = int(sv["flat_row_offsets"][s] + g)
      gvt[g, s] = True
  ca0 = ca[0]
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=6, n_canonical=n_can)
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, sv["flat_row_offsets"])
  bucket = n_flat
  coords_flat = jnp.zeros((bucket, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((bucket,), dtype=jnp.float32)
  res_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  chain_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  sm_flat = jnp.zeros((bucket,), dtype=jnp.int32)
  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(jnp.asarray(ca[s]))
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)
  seq, logits, _ = sample_fn(
    jax.random.fold_in(key, 12),
    coords_flat,
    mask_flat,
    res_flat,
    chain_flat,
    tie_group_map=jnp.asarray(tie_flat, dtype=jnp.int32),
    num_groups=num_groups,
    multistate_mode="state_vmap_exact",
    multi_state_strategy="arithmetic_mean",
    structure_mapping=sm_flat,
    temperature=jnp.float32(0.2),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states),
    fixed_mask=jnp.ones((bucket,), dtype=jnp.float32),
    fixed_tokens=jnp.zeros((bucket,), dtype=jnp.int32),
    group_indices_table=jnp.asarray(git, dtype=jnp.int32),
    group_valid_table=jnp.asarray(gvt, dtype=bool),
    wave_group_ids=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions=jnp.asarray(w_pos, dtype=jnp.int32),
    wave_group_valid=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid=jnp.asarray(w_pv, dtype=bool),
    state_vmap_n_canonical=n_can,
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    tie_group_map_stack=jnp.asarray(sv["tie_group_map_stack"], dtype=jnp.int32),
    fixed_mask_stack=jnp.asarray(sv["fixed_mask_stack"], dtype=jnp.float32),
    fixed_tokens_stack=jnp.asarray(sv["fixed_tokens_stack"], dtype=jnp.int32),
    bias_stack=None,
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    wave_group_ids_local=jnp.asarray(w_id, dtype=jnp.int32),
    wave_group_positions_local=jnp.asarray(w_loc, dtype=jnp.int32),
    wave_group_valid_local=jnp.asarray(w_gv, dtype=bool),
    wave_position_valid_local=jnp.asarray(w_pv2, dtype=bool),
    y_stack=jnp.asarray(y_st, dtype=jnp.float32),
    y_t_stack=jnp.asarray(y_tst, dtype=jnp.int32),
    y_m_stack=jnp.asarray(y_mst, dtype=jnp.float32),
  )
  assert seq.shape == (n_can,)
  assert logits.shape == (n_can, 21)
  assert jnp.all(jnp.isfinite(logits))
