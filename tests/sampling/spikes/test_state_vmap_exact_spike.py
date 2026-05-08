"""Phase 0a spike: ``state_vmap_exact`` unconditional path vs explicit ``jax.vmap`` stack (roadmap §227).

**Roadmap §11 item 10 (split):** this module satisfies the **spike evidence** slice (numeric
agreement + HLO stats + recorded go/no-go for Phase 4 *entry*). The clause *matching Phase 4
implementation* in §11 #10 is **not** claimed here — that belongs to Phase 4 registry/unify PRs.
See ``.agents/TECHNICAL_DEBT.md`` (Phase 0a spike vs §11 #10) and sprint
``.agents/SPRINT_refactor-phase3b-20260506.md`` PR1.

This test is the numeric gate for the unconditional **ProteinMPNN** branch (``tie_group_map=None``).

**Payloads:** markers ``parity_fast`` / ``parity_heavy`` exercise **synthetic** multi-state stacks
built in-process (``jax.random`` CA coords, zero masks). They are **not** wired to
``tests/parity`` LigandMPNN fixtures; ``parity_heavy`` only enlarges ``n_states`` / ``n_can`` when
``REFERENCE_PATH`` is set so cluster/local opt-in jobs can pay the cost without skipping in CI.

HLO narrative for PR visibility: both code paths emit ``UserWarning`` bodies including UTF-8 byte
length, newline count, and a case-insensitive count of the substring ``custom-call`` (cheap proxy
for custom-call ops in the IR text). This is **advisory** evidence for PR go/no-go; CI does not
assert on HLO text (see ``tests/profiling/hlo_allowlist.toml`` for unrelated callable thresholds).
Use ``pytest -W default`` to surface warnings.
"""

from __future__ import annotations

import os
import warnings
from typing import NamedTuple

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


class _HloSummary(NamedTuple):
  utf8_bytes: int
  newline_count: int
  custom_call_markers: int


def _summarize_hlo_text(hlo: str) -> _HloSummary:
  """Cheap IR stats: UTF-8 size, raw newline count, ``custom-call`` substring hits (lowered)."""
  data = hlo.encode("utf-8")
  lowered = hlo.lower()
  return _HloSummary(
    utf8_bytes=len(data),
    newline_count=hlo.count("\n"),
    custom_call_markers=lowered.count("custom-call"),
  )


def _emit_spike_hlo_warnings(hlo: str, *, path_tag: str) -> None:
  """Emit one advisory ``UserWarning`` per lowered path (tag disambiguates PR logs)."""
  s = _summarize_hlo_text(hlo)
  warnings.warn(
    f"spike_hlo_{path_tag} "
    f"bytes={s.utf8_bytes} newlines={s.newline_count} custom_call_markers={s.custom_call_markers}",
    UserWarning,
    stacklevel=2,
  )


def _run_state_vmap_exact_unconditional_spike(*, n_states: int, n_can: int, key: jax.Array) -> None:
  """Shared numeric + HLO export body for fast and heavy spike tests."""
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
    state_weights=sw,
    state_mapping=sm_flat,
    inference=True,
  )

  def run_ref(pk: jax.Array) -> jax.Array:
    """Explicit ``jax.vmap`` encode → decode → ``w_out`` stack; ``pk`` feeds feature/encoder keys."""
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
  assert jnp.allclose(
    logits_sv,
    logits_ref,
    rtol=rtol,
    atol=atol,
  ), (
    "Phase 0a: score_unconditional_state_vmap_exact logits differ from explicit jax.vmap stack "
    f"(rtol={rtol}, atol={atol}). See roadmap Phase 0a / Q6."
  )

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
      state_weights=sw,
      state_mapping=sm_flat,
      inference=True,
    )

  hlo_sv = export_hlo(run_sv, key)
  hlo_ref = export_hlo(run_ref, key)
  _emit_spike_hlo_warnings(hlo_sv, path_tag="state_vmap_exact")
  _emit_spike_hlo_warnings(hlo_ref, path_tag="explicit_vmap_ref")


@pytest.mark.phase0a_spike
@pytest.mark.parity_fast
def test_state_vmap_exact_unconditional_matches_explicit_vmap_stack() -> None:
  """``score_unconditional_state_vmap_exact`` matches a literal ``jax.vmap`` encode/decode."""
  key = jax.random.PRNGKey(101)
  _run_state_vmap_exact_unconditional_spike(n_states=2, n_can=6, key=key)


@pytest.mark.phase0a_spike
@pytest.mark.parity_heavy
def test_state_vmap_exact_spike_extended_when_reference_assets() -> None:
  """Larger synthetic stack when ``REFERENCE_PATH`` points at a real directory (local / CI opt-in)."""
  ref = os.environ.get("REFERENCE_PATH")
  if not ref or not os.path.isdir(ref):
    pytest.skip("REFERENCE_PATH unset or not a directory; heavy spike is opt-in.")

  key = jax.random.PRNGKey(707)
  _run_state_vmap_exact_unconditional_spike(n_states=3, n_can=10, key=key)
