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
from prxteinmpnn.profiling.hlo_tools import assert_zero_copy_overhead, export_hlo

_ROOT = Path(__file__).resolve().parent


def _allowlist() -> dict:
  with (_ROOT / "hlo_allowlist.toml").open("rb") as f:
    return tomllib.load(f)


@pytest.fixture
def tiny_model() -> PrxteinMPNN:
  key = jax.random.PRNGKey(0)
  m = PrxteinMPNN(
    node_features=16,
    edge_features=16,
    hidden_features=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    k_neighbors=4,
    key=key,
  )
  return eqx.tree_inference(m, value=True)


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
