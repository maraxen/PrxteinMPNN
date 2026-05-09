"""``PrxteinMPNN`` / ``PrxteinLigandMPNN`` ``__call__`` parity vs logits factories (stacked exact)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_state_vmap_fn
from prxteinmpnn.sampling.state_vmap_prep import (
  build_state_vmap_exact_stacks,
  slice_flat_tensor_to_stack,
)
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_state_vmap_fn


def _identity_af_to_mpnn(x: np.ndarray) -> np.ndarray:
  return np.asarray(x, dtype=np.int32)


@pytest.mark.parametrize("kind", ["unconditional", "conditional"])
def test_prxteinmpnn_call_matches_state_vmap_factories(kind: str) -> None:
  n_states, n_can = 2, 6
  key = jax.random.PRNGKey(101)
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
  cs = jnp.asarray(sv["coords_stack"], dtype=jnp.float32)
  ms = jnp.asarray(sv["mask_stack"], dtype=jnp.float32)
  ris = jnp.asarray(sv["residue_index_stack"], dtype=jnp.int32)
  cis = jnp.asarray(sv["chain_index_stack"], dtype=jnp.int32)
  rows = jnp.asarray(sv["state_flat_rows"], dtype=jnp.int32)
  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  pk = jax.random.fold_in(key, 7)

  coords_dummy = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_dummy = jnp.ones((n_flat,), dtype=jnp.float32)
  res_dummy = jnp.arange(n_flat, dtype=jnp.int32)
  chain_dummy = jnp.zeros((n_flat,), dtype=jnp.int32)

  if kind == "unconditional":
    uf = make_unconditional_logits_state_vmap_fn(model)
    lf_f = uf(pk, cs, ms, ris, cis, rows, n_flat, None, strat_idx, sw, None)
    _, lf_c = model(
      coords_dummy,
      mask_dummy,
      res_dummy,
      chain_dummy,
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
  else:
    rng = np.random.default_rng(42)
    seq_flat = rng.integers(0, 20, size=(n_flat,), dtype=np.int32)
    cf = make_conditional_logits_state_vmap_fn(model)
    lf_f = cf(
      pk,
      jnp.asarray(seq_flat, dtype=jnp.int32),
      cs,
      ms,
      ris,
      cis,
      rows,
      n_flat,
      None,
      strat_idx,
      sw,
      None,
    )
    _, lf_c = model(
      coords_dummy,
      mask_dummy,
      res_dummy,
      chain_dummy,
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
      tie_group_map=None,
      multi_state_strategy="arithmetic_mean",
      state_weights=sw,
      state_mapping=None,
    )

  np.testing.assert_allclose(np.asarray(lf_f), np.asarray(lf_c), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ["unconditional", "conditional"])
def test_prxteinligandmpnn_call_matches_state_vmap_factories(kind: str) -> None:
  n_states, n_can = 2, 5
  key = jax.random.PRNGKey(202)
  rng = np.random.default_rng(1)
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

  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  pk = jax.random.fold_in(key, 13)

  coords_dummy = jnp.zeros((n_flat, 4, 3), dtype=jnp.float32)
  mask_dummy = jnp.ones((n_flat,), dtype=jnp.float32)
  res_dummy = jnp.arange(n_flat, dtype=jnp.int32)
  chain_dummy = jnp.zeros((n_flat,), dtype=jnp.int32)
  y_dummy = jnp.zeros((n_flat, n_atoms, 3), dtype=jnp.float32)
  yt_dummy = jnp.ones((n_flat, n_atoms), dtype=jnp.int32)
  ym_dummy = jnp.ones((n_flat, n_atoms), dtype=jnp.float32)

  if kind == "unconditional":
    uf = make_unconditional_logits_state_vmap_fn(model)
    lf_f = uf(pk, cs, ms, ris, cis, ys, yts, yms, rows, n_flat, None, strat_idx, ms_temp, sw, None)
    _, lf_c = model(
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
    cf = make_conditional_logits_state_vmap_fn(model)
    lf_f = cf(
      pk,
      jnp.asarray(seq_flat, dtype=jnp.int32),
      cs,
      ms,
      ris,
      cis,
      ys,
      yts,
      yms,
      rows,
      n_flat,
      None,
      strat_idx,
      ms_temp,
      sw,
      None,
    )
    _, lf_c = model(
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
      tie_group_map=None,
      multi_state_strategy="arithmetic_mean",
      state_weights=sw,
      state_mapping=None,
    )

  np.testing.assert_allclose(np.asarray(lf_f), np.asarray(lf_c), rtol=1e-5, atol=1e-5)


def test_state_vmap_exact_autoreg_via_call_errors() -> None:
  key = jax.random.PRNGKey(3)
  m = PrxteinMPNN(16, 16, 16, 1, 1, 4, key=key)
  n_flat = 4
  with pytest.raises(ValueError, match="state_vmap_exact autoregressive"):
    m(
      jnp.zeros((n_flat, 4, 3)),
      jnp.ones((n_flat,)),
      jnp.arange(n_flat),
      jnp.zeros((n_flat,), jnp.int32),
      "autoregressive",
      prng_key=key,
      multistate_mode="state_vmap_exact",
      coords_stack=jnp.ones((2, n_flat, 4, 3)),
      mask_stack=jnp.ones((2, n_flat)),
      residue_index_stack=jnp.zeros((2, n_flat), jnp.int32),
      chain_index_stack=jnp.zeros((2, n_flat), jnp.int32),
      state_flat_rows=jnp.zeros((2, n_flat), jnp.int32),
      n_flat=n_flat,
    )
