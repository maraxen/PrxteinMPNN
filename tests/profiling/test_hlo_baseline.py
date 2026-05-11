"""HLO export smoke + zero-copy wiring (Phase 0; roadmap §4 Phase 0).

Raw ``baseline_hlo/*.txt`` files are **review-only** artifacts. CI does **not**
fail on HLO text drift; we only check export succeeds and optional byte ceilings
from ``hlo_allowlist.toml`` (§13 Q8).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload
from prxteinmpnn.profiling.hlo_tools import assert_zero_copy_overhead, export_hlo

_ROOT = Path(__file__).resolve().parent


def _allowlist() -> dict:
  with (_ROOT / "hlo_allowlist.toml").open("rb") as f:
    return tomllib.load(f)


def test_export_hlo_model_call_under_allowlist(tiny_model: PrxteinMPNN) -> None:
  n = 4
  coords = jnp.zeros((n, 4, 3), jnp.float32)
  mask = jnp.ones((n,), jnp.float32)
  ri = jnp.arange(n, dtype=jnp.int32)
  ci = jnp.zeros((n,), jnp.int32)
  pk = jax.random.PRNGKey(1)

  def f(pk: jax.Array) -> jax.Array:
    return tiny_model(coords, mask, ri, ci, "unconditional", prng_key=pk)[1]

  hlo = export_hlo(f, pk)
  max_b = int(_allowlist()["model_call"]["max_hlo_bytes"])
  assert len(hlo.encode("utf-8")) <= max_b


def test_assert_zero_copy_overhead_self_check(tiny_model: PrxteinMPNN) -> None:
  n = 4
  coords = jnp.zeros((n, 4, 3), jnp.float32)
  mask = jnp.ones((n,), jnp.float32)
  ri = jnp.arange(n, dtype=jnp.int32)
  ci = jnp.zeros((n,), jnp.int32)
  pk = jax.random.PRNGKey(2)

  def f(pk: jax.Array) -> jax.Array:
    return tiny_model(coords, mask, ri, ci, "unconditional", prng_key=pk)[1]

  assert_zero_copy_overhead(f, f, pk)


def test_baseline_hlo_review_artifacts_exist() -> None:
  names = ("model_call", "score", "sample", "logits")
  for name in names:
    p = _ROOT / "baseline_hlo" / f"{name}.txt"
    assert p.is_file(), f"missing review artifact {p}"
    assert p.stat().st_size > 0, f"empty review artifact {p}"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_hlo_score_unconditional_payload(
    tiny_model: PrxteinMPNN,
    tiny_stack: MultistateStackPayload,
) -> None:
  def f(pk: jax.Array) -> jax.Array:
    return tiny_model.score_unconditional_from_payload(
        pk,
        tiny_stack,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

  hlo = export_hlo(f, jax.random.PRNGKey(0))
  max_b = int(_allowlist()["score_unconditional_payload"]["max_hlo_bytes"])
  assert len(hlo.encode("utf-8")) <= max_b


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_hlo_score_conditional_payload(
    tiny_model: PrxteinMPNN,
    tiny_stack: MultistateStackPayload,
) -> None:
  S, L, V = tiny_stack.n_states, tiny_stack.n_canonical, 21
  seq_oh = jnp.zeros((S, L, V))
  ar_mask = jnp.eye(L, dtype=jnp.float32)[None].repeat(S, axis=0)

  def f(pk: jax.Array) -> jax.Array:
    return tiny_model.score_conditional_from_payload(
        pk,
        tiny_stack,
        seq_oh,
        ar_mask,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        state_weights=None,
        state_mapping=None,
        inference=True,
    )

  hlo = export_hlo(f, jax.random.PRNGKey(0))
  max_b = int(_allowlist()["score_conditional"]["max_hlo_bytes"])
  assert len(hlo.encode("utf-8")) <= max_b


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_export_hlo_sample_autoregressive_payload(
    tiny_model: PrxteinMPNN,
    tiny_stack: MultistateStackPayload,
) -> None:
  import numpy as np
  from prxteinmpnn.utils.wave_parallel import compute_wave_assignments
  from prxteinmpnn.sampling.state_vmap_prep import remap_wave_positions_flat_to_local

  S, L = tiny_stack.n_states, tiny_stack.n_canonical
  tie_flat = np.tile(np.arange(L, dtype=np.int32), S)
  flat_offsets = np.array([s * L for s in range(S + 1)], dtype=np.int32)
  git = np.full((L, S), -1, dtype=np.int32)
  gvt = np.zeros((L, S), dtype=bool)
  for g in range(L):
    for s in range(S):
      git[g, s] = int(flat_offsets[s] + g)
      gvt[g, s] = True
  ca0 = np.zeros((L, 4, 3), dtype=np.float32)
  w_id, w_pos, w_gv, w_pv = compute_wave_assignments(
      ca0, tie_flat, git, gvt, k_neighbors=4, n_canonical=L
  )
  w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, flat_offsets)
  ar_mask = jnp.zeros((S, L, L), dtype=jnp.float32)
  bias = jnp.zeros((S, L, 21), dtype=jnp.float32)

  def f(pk: jax.Array) -> jax.Array:
    seqs, _ = tiny_model.sample_autoregressive_from_payload(
        pk,
        tiny_stack,
        ar_mask,
        bias,
        1.0,
        0,
        None,
        jnp.asarray(w_id, dtype=jnp.int32),
        jnp.asarray(w_loc, dtype=jnp.int32),
        jnp.asarray(w_gv),
        jnp.asarray(w_pv2),
    )
    return seqs

  hlo = export_hlo(f, jax.random.PRNGKey(0))
  max_b = int(_allowlist()["sample_autoregressive"]["max_hlo_bytes"])
  assert len(hlo.encode("utf-8")) <= max_b
