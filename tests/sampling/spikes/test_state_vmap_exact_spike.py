"""Phase 0a spike: ``state_vmap_exact`` unconditional path vs explicit ``jax.vmap`` stack (roadmap §227).

Records **go / no-go** for Phase 4 in the PR that merges substantive changes; this test is the
numeric gate for the unconditional **ProteinMPNN** branch (``tie_group_map=None``).

HLO byte counts for PR narrative: the test emits a ``UserWarning`` body
``spike_hlo_bytes_state_vmap_exact=...`` (visible under ``pytest -W default``).
"""

from __future__ import annotations

import os
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.multistate_stack import scatter_stack_to_flat
from prxteinmpnn.profiling.hlo_tools import export_hlo
from prxteinmpnn.sampling.state_vmap_prep import build_state_vmap_exact_stacks
from prxteinmpnn.utils.testing import get_tolerances


def _identity_af_to_mpnn(x: np.ndarray) -> np.ndarray:
  return np.asarray(x, dtype=np.int32)


@pytest.mark.parity_fast
def test_state_vmap_exact_unconditional_matches_explicit_vmap_stack() -> None:
  """``score_unconditional_state_vmap_exact`` matches a literal ``jax.vmap`` encode/decode."""
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
    with_ligand=False,
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
  sw = jnp.ones((n_states,), dtype=jnp.float32) / jnp.float32(n_states)
  sm_flat = jnp.zeros((n_flat,), dtype=jnp.int32)
  for s in range(n_states):
    lo = int(sv["flat_row_offsets"][s])
    sm_flat = sm_flat.at[lo : lo + n_can].set(s)

  strat_idx = jnp.int32(0)
  ms_temp = jnp.float32(1.0)

  logits_sv = model.score_unconditional_state_vmap_exact(
    key,
    cs,
    ms,
    ris,
    cis,
    rows,
    n_flat,
    tie_group_map=None,
    multi_state_strategy_idx=strat_idx,
    multi_state_temperature=ms_temp,
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
  )

  k_enc, k_feat = jax.random.split(key)

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
  logits_ref = scatter_stack_to_flat(logits_s, rows, n_flat)

  rtol, atol = get_tolerances(jnp.float32)
  assert jnp.allclose(logits_sv, logits_ref, rtol=rtol, atol=atol)

  def run_sv(pk: jax.Array) -> jax.Array:
    return model.score_unconditional_state_vmap_exact(
      pk,
      cs,
      ms,
      ris,
      cis,
      rows,
      n_flat,
      tie_group_map=None,
      multi_state_strategy_idx=strat_idx,
      multi_state_temperature=ms_temp,
      state_weights=sw,
      state_mapping=sm_flat,
      inference=True,
    )

  hlo_sv = export_hlo(run_sv, key)
  warnings.warn(
    f"spike_hlo_bytes_state_vmap_exact={len(hlo_sv.encode('utf-8'))}",
    UserWarning,
    stacklevel=1,
  )


@pytest.mark.parity_heavy
def test_state_vmap_exact_spike_placeholder_heavy_when_reference_assets() -> None:
  """Optional parity_heavy segment when ``REFERENCE_PATH`` is set (roadmap §228)."""
  if not os.environ.get("REFERENCE_PATH"):
    pytest.skip("REFERENCE_PATH not set; run locally for parity_heavy segment.")
  pytest.skip(
    "Heavy spike extension: add reference-backed batch + HLO narrative when assets are wired."
  )
