"""Parity: payload-first state_vmap_exact logits helpers vs tuple-first factories."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN, PrxteinMPNN
from prxteinmpnn.payloads import LigandStack, MultistateStackPayload
from prxteinmpnn.sampling.conditional_logits import make_conditional_logits_state_vmap_fn
from prxteinmpnn.sampling.state_vmap_payload_logits import (
  conditional_state_vmap_logits_from_payload,
  unconditional_state_vmap_logits_from_payload,
)
from prxteinmpnn.sampling.state_vmap_prep import (
  build_state_vmap_exact_stacks,
  multistate_stack_payload_from_prep_numpy,
  slice_flat_tensor_to_stack,
)
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_state_vmap_fn
from prxteinmpnn.utils.testing import get_tolerances


def _identity_af_to_mpnn(x: np.ndarray) -> np.ndarray:
  return np.asarray(x, dtype=np.int32)


def _tiny_stack_payload(*, key: jax.Array) -> tuple[jnp.ndarray, MultistateStackPayload]:
  n_states, n_can = 2, 6
  peptide_lengths = np.zeros(n_states, dtype=np.int32)
  ca = np.asarray(jax.random.normal(key, (n_states, n_can, 4, 3)), dtype=np.float32)
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
  pay = multistate_stack_payload_from_prep_numpy(sv, n_states=n_states, n_canonical=n_can)
  n_flat = int(sv["flat_row_offsets"][-1])
  return jnp.asarray(np.arange(n_flat, dtype=np.int32)), pay


@pytest.mark.parity_fast
def test_unconditional_payload_logits_matches_factory() -> None:
  key = jax.random.PRNGKey(303)
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
  _, stack = _tiny_stack_payload(key=key)
  pk = jax.random.fold_in(key, 1)
  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((stack.n_states,), dtype=jnp.float32) / jnp.float32(stack.n_states)

  uf = make_unconditional_logits_state_vmap_fn(model)
  ref = uf(
    pk,
    stack.coords_stack,
    stack.mask_stack,
    stack.residue_index_stack,
    stack.chain_index_stack,
    stack.state_flat_rows,
    stack.n_flat,
    None,
    strat_idx,
    ms_temp,
    sw,
    None,
  )
  got = unconditional_state_vmap_logits_from_payload(
    model,
    pk,
    stack,
    None,
    tie_group_map=None,
    multi_state_strategy_idx=strat_idx,
    multi_state_temperature=ms_temp,
    state_weights=sw,
    state_mapping=None,
  )
  rtol, atol = get_tolerances(jnp.float32)
  np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=float(rtol), atol=float(atol))


@pytest.mark.parity_fast
def test_conditional_payload_logits_matches_factory() -> None:
  key = jax.random.PRNGKey(404)
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
  seq_flat, stack = _tiny_stack_payload(key=key)
  pk = jax.random.fold_in(key, 2)
  strat_idx = jnp.int32(1)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((stack.n_states,), dtype=jnp.float32) / jnp.float32(stack.n_states)

  cf = make_conditional_logits_state_vmap_fn(model)
  ref = cf(
    pk,
    seq_flat,
    stack.coords_stack,
    stack.mask_stack,
    stack.residue_index_stack,
    stack.chain_index_stack,
    stack.state_flat_rows,
    stack.n_flat,
    None,
    strat_idx,
    ms_temp,
    sw,
    None,
  )
  got = conditional_state_vmap_logits_from_payload(
    model,
    pk,
    seq_flat,
    stack,
    None,
    tie_group_map=None,
    multi_state_strategy_idx=strat_idx,
    multi_state_temperature=ms_temp,
    state_weights=sw,
    state_mapping=None,
    ar_mask_stack=None,
    bias_flat=None,
  )
  rtol, atol = get_tolerances(jnp.float32)
  np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=float(rtol), atol=float(atol))


def _ligand_stack_and_payload(
  *, key: jax.Array, rng: np.random.Generator
) -> tuple[jnp.ndarray, MultistateStackPayload, LigandStack]:
  n_states, n_can = 2, 5
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
  pay = multistate_stack_payload_from_prep_numpy(sv, n_states=n_states, n_canonical=n_can)
  lig = LigandStack(
    y_stack=jnp.asarray(y_st, dtype=jnp.float32),
    y_t_stack=jnp.asarray(y_tst, dtype=jnp.int32),
    y_m_stack=jnp.asarray(y_mst, dtype=jnp.float32),
  )
  seq_flat = jnp.asarray(np.arange(n_flat, dtype=np.int32))
  return seq_flat, pay, lig


@pytest.mark.parity_fast
@pytest.mark.parametrize("kind", ["unconditional", "conditional"])
def test_ligand_payload_logits_matches_factory(kind: str) -> None:
  key = jax.random.PRNGKey(707)
  rng = np.random.default_rng(707)
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
  seq_flat, stack, lig = _ligand_stack_and_payload(key=key, rng=rng)
  pk = jax.random.fold_in(key, 1)
  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)
  sw = jnp.ones((stack.n_states,), dtype=jnp.float32) / jnp.float32(stack.n_states)
  rtol, atol = get_tolerances(jnp.float32)

  if kind == "unconditional":
    uf = make_unconditional_logits_state_vmap_fn(model)
    ref = uf(
      pk,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      lig.y_stack,
      lig.y_t_stack,
      lig.y_m_stack,
      stack.state_flat_rows,
      stack.n_flat,
      None,
      strat_idx,
      ms_temp,
      sw,
      None,
    )
    got = unconditional_state_vmap_logits_from_payload(
      model,
      pk,
      stack,
      lig,
      tie_group_map=None,
      multi_state_strategy_idx=strat_idx,
      multi_state_temperature=ms_temp,
      state_weights=sw,
      state_mapping=None,
    )
  else:
    cf = make_conditional_logits_state_vmap_fn(model)
    ref = cf(
      pk,
      seq_flat,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      lig.y_stack,
      lig.y_t_stack,
      lig.y_m_stack,
      stack.state_flat_rows,
      stack.n_flat,
      None,
      strat_idx,
      ms_temp,
      sw,
      None,
    )
    got = conditional_state_vmap_logits_from_payload(
      model,
      pk,
      seq_flat,
      stack,
      lig,
      tie_group_map=None,
      multi_state_strategy_idx=strat_idx,
      multi_state_temperature=ms_temp,
      state_weights=sw,
      state_mapping=None,
      ar_mask_stack=None,
      bias_flat=None,
    )
  np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=float(rtol), atol=float(atol))
