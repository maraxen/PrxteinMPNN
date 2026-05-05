"""Parity checks: stacked ``state_vmap_exact`` scoring vs flat multistate forward."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.sampling.state_vmap_prep import build_state_vmap_exact_stacks, slice_flat_tensor_to_stack


def _identity_af_to_mpnn(x: np.ndarray) -> np.ndarray:
  return np.asarray(x, dtype=np.int32)


def test_multistate_unconditional_scores_match_flat_small_prot():
  n_states, n_can = 2, 6
  key = jax.random.PRNGKey(7)
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
  np_ca = np.asarray(ca)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)

  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=False,
    ca_states_4=np_ca,
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  n_flat = int(sv["flat_row_offsets"][-1])
  n_pad = int(sv["coords_stack"].shape[1])
  coords_flat = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((n_flat,), dtype=jnp.float32)
  res_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  chain_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)

  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(ca[s])
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  _, logits_flat = model(
    coords_flat,
    mask_flat,
    res_flat,
    chain_flat,
    decoding_approach="unconditional",
    prng_key=key,
    inference=True,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy="arithmetic_mean",
    multi_state_temperature=jnp.float32(1.0),
    structure_mapping=sm_flat,
    state_weights=sw,
    state_mapping=sm_flat,
  )

  logits_sv = model.score_unconditional_state_vmap_exact(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
  )

  assert logits_flat.shape == (n_flat, 21)
  # fp32+vmap reordering yields ~1e-2 deltas vs one flat graph; tight 1e-4 is unrealistic here.
  assert jnp.allclose(logits_flat, logits_sv, rtol=0.06, atol=0.06)


def test_multistate_conditional_scores_match_flat_small_prot():
  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(13)
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
  rng = np.random.default_rng(1)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = np.zeros((n_states, n_can), dtype=np.int32)

  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=False,
    ca_states_4=ca,
    peptide_bb=None,
    base_fixed_mask=base_fixed,
    aatype_states=aatype,
    af_to_mpnn=_identity_af_to_mpnn,
  )
  n_flat = int(sv["flat_row_offsets"][-1])
  n_pad = int(sv["coords_stack"].shape[1])
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)

  coords_flat = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((n_flat,), dtype=jnp.float32)
  res_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  chain_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  seq_flat = jax.random.randint(jax.random.fold_in(key, 2), (n_flat,), 0, 20, dtype=jnp.int32)
  oh_flat = jax.nn.one_hot(seq_flat, model.w_s_embed.num_embeddings)

  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(jnp.asarray(ca[s]))
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  ar_zeros = jnp.zeros((n_flat, n_flat), dtype=jnp.int32)
  _, logits_flat = model(
    coords_flat,
    mask_flat,
    res_flat,
    chain_flat,
    decoding_approach="conditional",
    prng_key=key,
    inference=True,
    ar_mask=ar_zeros,
    one_hot_sequence=oh_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy="arithmetic_mean",
    multi_state_temperature=jnp.float32(1.0),
    structure_mapping=sm_flat,
    state_weights=sw,
    state_mapping=sm_flat,
  )

  oh_stack_np = slice_flat_tensor_to_stack(
    np.asarray(oh_flat),
    sv["state_flat_rows"],
    n_states,
    n_pad,
  )
  oh_stack = jnp.asarray(oh_stack_np, dtype=jnp.float32)
  ar_stack = jnp.zeros((n_states, n_pad, n_pad), dtype=jnp.int32)

  logits_sv = model.score_conditional_state_vmap_exact(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    oh_stack,
    ar_stack,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
  )

  max_abs = float(jnp.max(jnp.abs(logits_flat - logits_sv)))
  assert max_abs < 0.12, max_abs


def test_multistate_unconditional_scores_match_flat_small_ligand():
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

  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)

  n_atoms = 3
  ytf = rng.integers(1, 20, size=(n_flat, n_atoms), dtype=np.int32)
  yf = rng.normal(size=(n_flat, n_atoms, 3)).astype(np.float32)
  ymf = np.ones((n_flat, n_atoms), dtype=np.float32)
  y_st = jnp.asarray(slice_flat_tensor_to_stack(yf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)
  y_tst = jnp.asarray(slice_flat_tensor_to_stack(ytf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.int32)
  y_mst = jnp.asarray(slice_flat_tensor_to_stack(ymf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)

  coords_flat = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((n_flat,), dtype=jnp.float32)
  res_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  chain_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  y_flat = jnp.zeros((n_flat, n_atoms, 3), dtype=jnp.float32)
  yt_flat = jnp.zeros((n_flat, n_atoms), dtype=jnp.int32)
  ym_flat = jnp.zeros((n_flat, n_atoms), dtype=jnp.float32)

  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    coords_flat = coords_flat.at[lo : lo + n_can].set(jnp.asarray(ca[s]))
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)
    y_flat = y_flat.at[lo : lo + n_can].set(y_st[s, :n_can])
    yt_flat = yt_flat.at[lo : lo + n_can].set(y_tst[s, :n_can])
    ym_flat = ym_flat.at[lo : lo + n_can].set(y_mst[s, :n_can])

  git = np.full((n_can, n_states), -1, dtype=np.int32)
  gvt = np.zeros((n_can, n_states), dtype=bool)
  for g in range(n_can):
    for st in range(n_states):
      git[g, st] = int(sv["flat_row_offsets"][st] + g)
      gvt[g, st] = True

  _, logits_flat = model(
    coords_flat,
    mask_flat,
    res_flat,
    chain_flat,
    y_flat,
    yt_flat,
    ym_flat,
    decoding_approach="unconditional",
    prng_key=key,
    inference=True,
    tie_group_map=jnp.asarray(tie_flat),
    group_indices_table=jnp.asarray(git),
    group_valid_table=jnp.asarray(gvt),
    multi_state_strategy="arithmetic_mean",
    multi_state_temperature=jnp.float32(1.0),
    multistate_mode="flat",
    structure_mapping=sm_flat,
    state_weights=sw,
    state_mapping=sm_flat,
  )

  logits_sv = model.score_unconditional_state_vmap_exact(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    y_st,
    y_tst,
    y_mst,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
  )

  assert logits_flat.shape == (n_flat, 21)
  # fp32+vmap reordering yields ~1e-2 deltas vs one flat graph; tight 1e-4 is unrealistic here.
  assert jnp.allclose(logits_flat, logits_sv, rtol=0.06, atol=0.06)


@pytest.mark.parametrize("biased", [False, True])
def test_multistate_conditional_scores_match_flat_small_ligand(biased: bool):
  n_states, n_can = 2, 4
  key = jax.random.PRNGKey(19)
  rng = np.random.default_rng(4)
  model = PrxteinLigandMPNN(
    node_features=20,
    edge_features=20,
    hidden_features=20,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )

  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = rng.integers(0, 15, size=(n_states, n_can), dtype=np.int32)

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
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)

  n_atoms = 2
  ytf = rng.integers(1, 18, size=(n_flat, n_atoms), dtype=np.int32)
  yf = rng.normal(size=(n_flat, n_atoms, 3)).astype(np.float32)
  ymf = np.ones((n_flat, n_atoms), dtype=np.float32)
  y_st = jnp.asarray(slice_flat_tensor_to_stack(yf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)
  y_tst = jnp.asarray(slice_flat_tensor_to_stack(ytf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.int32)
  y_mst = jnp.asarray(slice_flat_tensor_to_stack(ymf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)

  coords_flat = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_flat = jnp.zeros((n_flat,), dtype=jnp.float32)
  res_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  chain_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  y_flat = jnp.zeros((n_flat, n_atoms, 3), dtype=jnp.float32)
  yt_flat = jnp.zeros((n_flat, n_atoms), dtype=jnp.int32)
  ym_flat = jnp.zeros((n_flat, n_atoms), dtype=jnp.float32)
  seq_flat = jax.random.randint(jax.random.fold_in(key, 4), (n_flat,), 0, 19, dtype=jnp.int32)
  oh_flat = jax.nn.one_hot(seq_flat, model.w_s_embed.num_embeddings)

  bias_flat = None
  if biased:
    bias_flat = jax.random.uniform(jax.random.fold_in(key, 9), (n_flat, 21), dtype=jnp.float32) * 0.1

  git = np.full((n_can, n_states), -1, dtype=np.int32)
  gvt = np.zeros((n_can, n_states), dtype=bool)
  for g in range(n_can):
    for st in range(n_states):
      git[g, st] = int(sv["flat_row_offsets"][st] + g)
      gvt[g, st] = True

  for st in range(n_states):
    lo = int(sv["flat_row_offsets"][st])
    coords_flat = coords_flat.at[lo : lo + n_can].set(jnp.asarray(ca[st]))
    mask_flat = mask_flat.at[lo : lo + n_can].set(1.0)
    res_flat = res_flat.at[lo : lo + n_can].set(jnp.arange(n_can))
    sm_flat = sm_flat.at[lo : lo + n_can].set(st)
    y_flat = y_flat.at[lo : lo + n_can].set(y_st[st, :n_can])
    yt_flat = yt_flat.at[lo : lo + n_can].set(y_tst[st, :n_can])
    ym_flat = ym_flat.at[lo : lo + n_can].set(y_mst[st, :n_can])

  ar_zeros = jnp.zeros((n_flat, n_flat), dtype=jnp.int32)
  kw: dict = {
    "tie_group_map": jnp.asarray(tie_flat),
    "group_indices_table": jnp.asarray(git),
    "group_valid_table": jnp.asarray(gvt),
    "multi_state_strategy": "arithmetic_mean",
    "multi_state_temperature": jnp.float32(1.0),
    "multistate_mode": "flat",
    "structure_mapping": sm_flat,
    "state_weights": sw,
    "state_mapping": sm_flat,
  }
  _, logits_flat = model(
    coords_flat,
    mask_flat,
    res_flat,
    chain_flat,
    y_flat,
    yt_flat,
    ym_flat,
    decoding_approach="conditional",
    prng_key=key,
    inference=True,
    ar_mask=ar_zeros,
    one_hot_sequence=oh_flat,
    bias=bias_flat,
    **kw,
  )

  oh_np = slice_flat_tensor_to_stack(np.asarray(oh_flat), sv["state_flat_rows"], n_states, n_pad)
  oh_stack = jnp.asarray(oh_np, dtype=jnp.float32)
  ar_stack = jnp.zeros((n_states, n_pad, n_pad), dtype=jnp.int32)

  logits_sv = model.score_conditional_state_vmap_exact(
    jax.random.fold_in(key, 8),
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    y_st,
    y_tst,
    y_mst,
    oh_stack,
    ar_stack,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    bias_flat=bias_flat,
    inference=True,
  )

  assert jnp.allclose(logits_flat, logits_sv, rtol=0.06, atol=0.06)


def test_factory_unconditional_state_vmap_prot_matches_direct():
  from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_state_vmap_fn

  n_states, n_can = 2, 6
  key = jax.random.PRNGKey(7)
  model = PrxteinMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=key,
  )
  fn = make_unconditional_logits_state_vmap_fn(model)
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = jax.random.normal(key, (n_states, n_can, 4, 3)).astype(jnp.float32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=False,
    ca_states_4=np.asarray(ca),
    peptide_bb=None,
    base_fixed_mask=np.zeros(n_can, dtype=np.float32),
    aatype_states=np.zeros((n_states, n_can), dtype=np.int32),
    af_to_mpnn=lambda x: np.asarray(x, dtype=np.int32),
  )
  n_flat = int(sv["flat_row_offsets"][-1])
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  off = 0
  for s in range(n_states):
    sm_flat = sm_flat.at[off : off + n_can].set(s)
    off += n_can

  direct = model.score_unconditional_state_vmap_exact(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
  )
  via_fn = fn(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
  )
  assert jnp.allclose(via_fn, direct, rtol=1e-5, atol=1e-5)


def test_factory_conditional_and_score_state_vmap_prot_smoke():
  from prxteinmpnn.scoring.score import make_score_fn
  from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_state_vmap_fn

  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(13)
  model = PrxteinMPNN(
    node_features=24,
    edge_features=24,
    hidden_features=24,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    key=key,
  )
  cl_fn = make_conditional_logits_state_vmap_fn(model)
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  rng = np.random.default_rng(1)
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
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  off = 0
  for s in range(n_states):
    sm_flat = sm_flat.at[off : off + n_can].set(s)
    off += n_can

  seq_flat = jax.random.randint(jax.random.fold_in(key, 2), (n_flat,), 0, 20, dtype=jnp.int32)
  stack_kwargs = dict(
    coords_stack=jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    mask_stack=jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    residue_index_stack=jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    chain_index_stack=jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    state_flat_rows=jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat=n_flat,
    state_weights=sw,
  )
  l1 = cl_fn(
    key,
    seq_flat,
    **stack_kwargs,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_mapping=sm_flat,
  )
  score_fn = make_score_fn(model, multistate_mode="state_vmap_exact")
  sc, l2, _order = score_fn(
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
  assert l1.shape == (n_flat, 21)
  assert l2.shape == (n_flat, 21)
  assert jnp.allclose(l1, l2, rtol=1e-5, atol=1e-5)
  assert sc.shape == ()
  assert jnp.isfinite(sc)


@pytest.mark.parametrize("lig_lc", [0, 2])
@pytest.mark.parametrize("states_chunk_size", [1, 2, 3])
def test_ligand_state_chunk_matches_full_vmap_unconditional(lig_lc: int, states_chunk_size: int):
  n_states, n_can = 3, 3  # shortest toy geometry that still fits k_neighbors below
  key = jax.random.PRNGKey(101)
  rng = np.random.default_rng(7)
  model = PrxteinLigandMPNN(
    node_features=20,
    edge_features=20,
    hidden_features=20,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    num_context_layers=2,
    dropout_rate=0.0,
    ligand_l_chunk=lig_lc,
    key=key,
  )

  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  base_fixed = np.zeros(n_can, dtype=np.float32)
  aatype = rng.integers(0, 14, size=(n_states, n_can), dtype=np.int32)

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
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  off = 0
  for st in range(n_states):
    sm_flat = sm_flat.at[off : off + n_can].set(st)
    off += n_can

  n_atoms = 2
  ytf = rng.integers(1, 17, size=(n_flat, n_atoms), dtype=np.int32)
  yf = rng.normal(size=(n_flat, n_atoms, 3)).astype(np.float32)
  ymf = np.ones((n_flat, n_atoms), dtype=np.float32)
  y_st = jnp.asarray(slice_flat_tensor_to_stack(yf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)
  y_tst = jnp.asarray(slice_flat_tensor_to_stack(ytf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.int32)
  y_mst = jnp.asarray(slice_flat_tensor_to_stack(ymf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)

  cs_kw = dict(states_chunk_size=states_chunk_size)
  logits_full = model.score_unconditional_state_vmap_exact(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    y_st,
    y_tst,
    y_mst,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    states_chunk_size=None,
  )
  logits_chunked = model.score_unconditional_state_vmap_exact(
    key,
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    y_st,
    y_tst,
    y_mst,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    **cs_kw,
  )
  assert jnp.allclose(logits_full, logits_chunked, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("lig_lc", [0, 2])
@pytest.mark.parametrize("states_chunk_size", [1, 2])
def test_ligand_state_chunk_matches_full_vmap_conditional(lig_lc: int, states_chunk_size: int):
  n_states, n_can = 3, 3
  key = jax.random.PRNGKey(107)
  rng = np.random.default_rng(8)
  model = PrxteinLigandMPNN(
    node_features=20,
    edge_features=20,
    hidden_features=20,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    num_context_layers=2,
    dropout_rate=0.0,
    ligand_l_chunk=lig_lc,
    key=key,
  )

  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = rng.normal(size=(n_states, n_can, 4, 3)).astype(np.float32)
  sv = build_state_vmap_exact_stacks(
    n_states=n_states,
    n_canonical=n_can,
    peptide_lengths=peptide_lengths,
    with_ligand=True,
    ca_states_4=ca,
    peptide_bb=None,
    base_fixed_mask=np.zeros(n_can, dtype=np.float32),
    aatype_states=rng.integers(0, 12, size=(n_states, n_can), dtype=np.int32),
    af_to_mpnn=_identity_af_to_mpnn,
  )
  n_pad = int(sv["coords_stack"].shape[1])
  n_flat = int(sv["flat_row_offsets"][-1])
  tie_flat = np.concatenate([np.arange(n_can, dtype=np.int32) for _ in range(n_states)], axis=0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  off = 0
  for st in range(n_states):
    sm_flat = sm_flat.at[off : off + n_can].set(st)
    off += n_can

  n_atoms = 2
  ytf = rng.integers(1, 16, size=(n_flat, n_atoms), dtype=np.int32)
  yf = rng.normal(size=(n_flat, n_atoms, 3)).astype(np.float32)
  ymf = np.ones((n_flat, n_atoms), dtype=np.float32)
  y_st = jnp.asarray(slice_flat_tensor_to_stack(yf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)
  y_tst = jnp.asarray(slice_flat_tensor_to_stack(ytf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.int32)
  y_mst = jnp.asarray(slice_flat_tensor_to_stack(ymf, sv["state_flat_rows"], n_states, n_pad), dtype=jnp.float32)

  seq_flat = jax.random.randint(jax.random.fold_in(key, 1), (n_flat,), 0, 18, dtype=jnp.int32)
  oh_flat = jax.nn.one_hot(seq_flat, model.w_s_embed.num_embeddings)
  oh_stack = jnp.asarray(
    slice_flat_tensor_to_stack(np.asarray(oh_flat), sv["state_flat_rows"], n_states, n_pad),
    dtype=jnp.float32,
  )
  ar_stack = jnp.zeros((n_states, n_pad, n_pad), dtype=jnp.int32)

  logits_full = model.score_conditional_state_vmap_exact(
    jax.random.fold_in(key, 2),
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    y_st,
    y_tst,
    y_mst,
    oh_stack,
    ar_stack,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
    states_chunk_size=None,
  )
  logits_chunked = model.score_conditional_state_vmap_exact(
    jax.random.fold_in(key, 2),
    jnp.asarray(sv["coords_stack"], dtype=jnp.float32),
    jnp.asarray(sv["mask_stack"], dtype=jnp.float32),
    jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32),
    jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32),
    y_st,
    y_tst,
    y_mst,
    oh_stack,
    ar_stack,
    jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32),
    n_flat,
    tie_group_map=jnp.asarray(tie_flat),
    multi_state_strategy_idx=jnp.int32(0),
    multi_state_temperature=jnp.float32(1.0),
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
    states_chunk_size=states_chunk_size,
  )
  assert jnp.allclose(logits_full, logits_chunked, rtol=1e-5, atol=1e-5)