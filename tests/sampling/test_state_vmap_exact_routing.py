"""Phase 4 routing guards for ``state_vmap_exact`` (roadmap §347–348).

**(a)** ``PrxteinMPNN`` / ``PrxteinLigandMPNN.__call__`` with
``multistate_mode=\"state_vmap_exact\"`` must match the **preserved direct**
stacked score methods (regression vs the implementation bodies dispatch routes
to).

**(b)** With Phase 0a / §13.2 **GO** recorded for unconditional **ProteinMPNN**,
logits must also match an explicit ``jax.vmap`` single-state encode/decode
reference (same construction as ``tests/sampling/spikes/test_state_vmap_exact_spike.py``).

Factories (:func:`make_unconditional_logits_state_vmap_fn`, etc.) are covered in
``test_multistate_exact_call_logits.py``; this module pins **descriptor routing →
direct score APIs** and the **GO vmap reference** slice.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack, scatter_stack_to_flat
from prxteinmpnn.sampling.state_vmap_prep import build_state_vmap_exact_stacks, slice_flat_tensor_to_stack
from prxteinmpnn.utils.testing import get_tolerances


def _identity_af_to_mpnn(x: np.ndarray) -> np.ndarray:
  return np.asarray(x, dtype=np.int32)


def _protein_sv_and_dummy_flat(*, key: jax.Array, n_states: int, n_can: int) -> tuple[dict[str, np.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = np.asarray(jax.random.normal(key, (n_states, n_can, 4, 3)), dtype=np.float32)
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
  n_flat = int(sv["flat_row_offsets"][-1])
  coords_dummy = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_dummy = jnp.ones((n_flat,), dtype=jnp.float32)
  res_dummy = jnp.arange(n_flat, dtype=jnp.int32)
  chain_dummy = jnp.zeros((n_flat,), dtype=jnp.int32)
  return sv, coords_dummy, mask_dummy, res_dummy, chain_dummy, n_flat


@pytest.mark.parity_fast
def test_prxtein_mpnn_unconditional_call_matches_direct_score() -> None:
  """§347(a): stacked-path ``__call__`` agrees with :meth:`score_unconditional`."""
  key = jax.random.PRNGKey(501)
  model = PrxteinMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=key,
  )
  model = eqx.tree_inference(model, value=True)

  sv, coords_d, mask_d, res_d, chain_d, n_flat = _protein_sv_and_dummy_flat(key=key, n_states=2, n_can=6)
  cs = jnp.asarray(sv["coords_stack"], dtype=jnp.float32)
  ms = jnp.asarray(sv["mask_stack"], dtype=jnp.float32)
  ris = jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32)
  cis = jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32)
  rows = jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32)
  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((2,), dtype=jnp.float32) / jnp.float32(2)
  pk = jax.random.fold_in(key, 9)

  logits_direct = model.score_unconditional(
    pk,
    cs,
    ms,
    ris,
    cis,
    rows,
    n_flat,
    tie_group_map=None,
    multi_state_strategy_idx=strat_idx,
    state_weights=sw,
    state_mapping=None,
    inference=True,
  )
  _, logits_call = model(
    coords_d,
    mask_d,
    res_d,
    chain_d,
    "unconditional",
    prng_key=pk,
    multistate_mode="state_vmap_exact",
    coords_stack=cs,
    mask_stack=ms,
    residue_index_stack=ris,
    chain_index_stack=cis,
    state_flat_rows=rows,
    n_flat=n_flat,
    tie_group_map=None,
    multi_state_strategy="arithmetic_mean",
    state_weights=sw,
    state_mapping=None,
  )
  rtol, atol = get_tolerances(jnp.float32)
  np.testing.assert_allclose(np.asarray(logits_call), np.asarray(logits_direct), rtol=float(rtol), atol=float(atol))


@pytest.mark.parity_fast
def test_prxtein_mpnn_conditional_call_matches_direct_score() -> None:
  """§347(a): conditional stacked ``__call__`` agrees with :meth:`score_conditional`."""
  key = jax.random.PRNGKey(502)
  rng = np.random.default_rng(0)
  model = PrxteinMPNN(
    node_features=30,
    edge_features=30,
    hidden_features=30,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=key,
  )
  model = eqx.tree_inference(model, value=True)

  sv, coords_d, mask_d, res_d, chain_d, n_flat = _protein_sv_and_dummy_flat(key=key, n_states=2, n_can=5)
  cs = jnp.asarray(sv["coords_stack"], dtype=jnp.float32)
  ms = jnp.asarray(sv["mask_stack"], dtype=jnp.float32)
  ris = jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32)
  cis = jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32)
  rows = jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32)
  s_dim, p_dim = ms.shape[0], ms.shape[1]
  arm = jnp.zeros((s_dim, p_dim, p_dim), dtype=jnp.int32)
  strat_idx = jnp.int32(1)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((2,), dtype=jnp.float32) / jnp.float32(2)
  pk = jax.random.fold_in(key, 3)
  seq_flat = rng.integers(0, 20, size=(n_flat,), dtype=np.int32)
  oh_in = jax.nn.one_hot(jnp.asarray(seq_flat, dtype=jnp.int32), model.w_s_embed.num_embeddings)
  seq_stack = gather_flat_to_stack(oh_in, rows)

  logits_direct = model.score_conditional(
    pk,
    cs,
    ms,
    ris,
    cis,
    seq_stack,
    arm,
    rows,
    n_flat,
    tie_group_map=None,
    multi_state_strategy_idx=strat_idx,
    state_weights=sw,
    state_mapping=None,
    bias_flat=None,
    inference=True,
  )
  _, logits_call = model(
    coords_d,
    mask_d,
    res_d,
    chain_d,
    "conditional",
    prng_key=pk,
    one_hot_sequence=jnp.asarray(seq_flat, dtype=jnp.int32),
    multistate_mode="state_vmap_exact",
    coords_stack=cs,
    mask_stack=ms,
    residue_index_stack=ris,
    chain_index_stack=cis,
    state_flat_rows=rows,
    n_flat=n_flat,
    ar_mask_stack=arm,
    tie_group_map=None,
    multi_state_strategy="geometric_mean",
    state_weights=sw,
    state_mapping=None,
  )
  rtol, atol = get_tolerances(jnp.float32)
  np.testing.assert_allclose(np.asarray(logits_call), np.asarray(logits_direct), rtol=float(rtol), atol=float(atol))


@pytest.mark.parity_fast
def test_prxtein_mpnn_unconditional_matches_explicit_vmap_go_branch() -> None:
  """§347(b) under §13.2 GO: unconditional logits match explicit ``jax.vmap`` stack reference."""
  key = jax.random.PRNGKey(503)
  model = PrxteinMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=8,
    key=key,
  )
  model = eqx.tree_inference(model, value=True)

  sv, _, _, _, _, n_flat = _protein_sv_and_dummy_flat(key=key, n_states=2, n_can=6)
  cs = jnp.asarray(sv["coords_stack"], dtype=jnp.float32)
  ms = jnp.asarray(sv["mask_stack"], dtype=jnp.float32)
  ris = jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32)
  cis = jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32)
  rows = jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32)
  strat_idx = jnp.int32(0)
  sw = jnp.ones((2,), dtype=jnp.float32) / jnp.float32(2)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  n_can = 6
  for s in range(2):
    lo = int(sv["flat_row_offsets"][s])
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  logits_sv = model.score_unconditional(
    key,
    cs,
    ms,
    ris,
    cis,
    rows,
    n_flat,
    tie_group_map=None,
    multi_state_strategy_idx=strat_idx,
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
  )

  def run_ref(pk: jax.Array) -> jax.Array:
    k_enc, k_feat = jax.random.split(pk)

    def encode_one(coords: jax.Array, ma: jax.Array, ri: jax.Array, ci: jax.Array):
      ef, nei, nf, _ = model.features(
        k_feat,
        coords,
        ma,
        ri,
        ci,
        jnp.asarray(0.0, jnp.float32),
        structure_mapping=None,
        initial_node_features=None,
        rbf_features=None,
        neighbor_indices=None,
      )
      nf2, ef2 = model.encoder(
        ef, nei, ma, initial_node_features=nf, inference=True, key=k_enc
      )
      return nf2, ef2, nei.astype(jnp.int32)

    node_b, edge_b, nei_b = jax.vmap(encode_one)(cs, ms, ris, cis)

    def decode_one(nb: jax.Array, eb: jax.Array, nei: jax.Array, mk: jax.Array):
      return model.decoder(nb, eb, nei, mk, key=k_enc)

    decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, ms)
    logits_s = jax.vmap(jax.vmap(model.w_out))(decoded)
    return scatter_stack_to_flat(logits_s, rows, n_flat)

  logits_ref = run_ref(key)
  rtol, atol = get_tolerances(jnp.float32)
  np.testing.assert_allclose(np.asarray(logits_sv), np.asarray(logits_ref), rtol=float(rtol), atol=float(atol))


@pytest.mark.parametrize("kind", ["unconditional", "conditional"])
@pytest.mark.parity_fast
def test_prxtein_ligand_mpnn_call_matches_direct_score(kind: str) -> None:
  """§347(a): LigandMPNN stacked ``__call__`` agrees with direct stacked score methods."""
  key = jax.random.PRNGKey(504)
  rng = np.random.default_rng(2)
  model = PrxteinLigandMPNN(
    node_features=28,
    edge_features=28,
    hidden_features=28,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=6,
    num_context_layers=2,
    dropout_rate=0.0,
    key=key,
  )
  model = eqx.tree_inference(model, value=True)

  n_states, n_can = 2, 5
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
  rows_np = sv["state_flat_rows"]
  y_st = slice_flat_tensor_to_stack(yf, rows_np, n_states, n_pad)
  y_tst = slice_flat_tensor_to_stack(ytf, rows_np, n_states, n_pad)
  y_mst = slice_flat_tensor_to_stack(ymf, rows_np, n_states, n_pad)

  cs = jnp.asarray(sv["coords_stack"], dtype=jnp.float32)
  ms = jnp.asarray(sv["mask_stack"], dtype=jnp.float32)
  ris = jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32)
  cis = jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32)
  rows = jnp.asarray(rows_np, dtype=jnp.int32)
  ys = jnp.asarray(y_st, dtype=jnp.float32)
  yts = jnp.asarray(y_tst, dtype=jnp.int32)
  yms = jnp.asarray(y_mst, dtype=jnp.float32)
  s_dim, p_dim = ms.shape[0], ms.shape[1]
  arm = jnp.zeros((s_dim, p_dim, p_dim), dtype=jnp.int32)

  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  pk = jax.random.fold_in(key, 11)

  coords_dummy = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_dummy = jnp.ones((n_flat,), dtype=jnp.float32)
  res_dummy = jnp.arange(n_flat, dtype=jnp.int32)
  chain_dummy = jnp.zeros((n_flat,), dtype=jnp.int32)
  y_dummy = jnp.zeros((n_flat, n_atoms, 3), dtype=jnp.float32)
  yt_dummy = jnp.ones((n_flat, n_atoms), dtype=jnp.int32)
  ym_dummy = jnp.ones((n_flat, n_atoms), dtype=jnp.float32)

  if kind == "unconditional":
    logits_direct = model.score_unconditional(
      pk,
      cs,
      ms,
      ris,
      cis,
      ys,
      yts,
      yms,
      rows,
      n_flat,
      tie_group_map=None,
      multi_state_strategy_idx=strat_idx,
      state_weights=sw,
      state_mapping=None,
    )
    _, logits_call = model(
      coords_dummy,
      mask_dummy,
      res_dummy,
      chain_dummy,
      y_dummy,
      yt_dummy,
      ym_dummy,
      "unconditional",
      prng_key=pk,
      multistate_mode="state_vmap_exact",
      coords_stack=cs,
      mask_stack=ms,
      residue_index_stack=ris,
      chain_index_stack=cis,
      y_stack=ys,
      y_t_stack=yts,
      y_m_stack=yms,
      state_flat_rows=rows,
      n_flat=n_flat,
      tie_group_map=None,
      multi_state_strategy="arithmetic_mean",
      state_weights=sw,
      state_mapping=None,
    )
  else:
    seq_flat = rng.integers(0, 20, size=(n_flat,), dtype=np.int32)
    oh_in = jax.nn.one_hot(jnp.asarray(seq_flat, dtype=jnp.int32), model.w_s_embed.num_embeddings)
    seq_stack = gather_flat_to_stack(oh_in, rows)
    logits_direct = model.score_conditional(
      pk,
      cs,
      ms,
      ris,
      cis,
      ys,
      yts,
      yms,
      seq_stack,
      arm,
      rows,
      n_flat,
      tie_group_map=None,
      multi_state_strategy_idx=strat_idx,
      state_weights=sw,
      state_mapping=None,
      bias_flat=None,
      inference=True,
    )
    _, logits_call = model(
      coords_dummy,
      mask_dummy,
      res_dummy,
      chain_dummy,
      y_dummy,
      yt_dummy,
      ym_dummy,
      "conditional",
      prng_key=pk,
      one_hot_sequence=jnp.asarray(seq_flat, dtype=jnp.int32),
      multistate_mode="state_vmap_exact",
      coords_stack=cs,
      mask_stack=ms,
      residue_index_stack=ris,
      chain_index_stack=cis,
      y_stack=ys,
      y_t_stack=yts,
      y_m_stack=yms,
      state_flat_rows=rows,
      n_flat=n_flat,
      ar_mask_stack=arm,
      tie_group_map=None,
      multi_state_strategy="arithmetic_mean",
      state_weights=sw,
      state_mapping=None,
    )

  rtol, atol = get_tolerances(jnp.float32)
  np.testing.assert_allclose(np.asarray(logits_call), np.asarray(logits_direct), rtol=float(rtol), atol=float(atol))
