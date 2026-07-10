"""End-to-end tests for the three E4 dispatch entry points (backlog node E4).

Each test builds a small, real ``ProteinEBMModel`` (same fixture shape as
``tests/ebm/test_model.py``) and checks the dispatched result against a plain
Python loop over ``model.energy`` -- proving the axis-tiled dispatch produces
identical numerics to the un-tiled reference, for both the Vmap and SafeMap
branches BatchPlanner can select.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from aminx.ebm.dispatch import (
  score_decoy_batch,
  score_mutant_ensemble,
  score_state_difference,
)
from aminx.ebm.model import ProteinEBMModel

TOKEN_S = 16
TOKEN_Z = 8
DEPTH = 2
HEADS = 2
N = 6


def _make_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=DEPTH,
    transformer_heads=HEADS,
    key=key,
  )


class TestScoreDecoyBatch:
  def test_vmap_path_matches_manual_loop(self) -> None:
    model = _make_model(jax.random.PRNGKey(0))
    k_coords, k_aatype = jax.random.split(jax.random.PRNGKey(1))
    d = 3
    coords = jax.random.normal(k_coords, (d, N, 3)) * 0.1
    aatype = jax.random.randint(k_aatype, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.2)

    # cardinality=3 <= default_batch_size=8 -> Vmap
    energies = score_decoy_batch(model, coords, aatype, t, mask, default_batch_size=8)
    assert energies.shape == (d,)

    expected = jnp.stack([model.energy(coords[i], aatype, t, mask) for i in range(d)])
    assert jnp.allclose(energies, expected, atol=1e-5)

  def test_safemap_path_matches_manual_loop(self) -> None:
    model = _make_model(jax.random.PRNGKey(2))
    k_coords, k_aatype = jax.random.split(jax.random.PRNGKey(3))
    d = 4
    coords = jax.random.normal(k_coords, (d, N, 3)) * 0.1
    aatype = jax.random.randint(k_aatype, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.2)

    # cardinality=4 > default_batch_size=2, divisible -> SafeMap
    energies = score_decoy_batch(model, coords, aatype, t, mask, default_batch_size=2)
    assert energies.shape == (d,)

    expected = jnp.stack([model.energy(coords[i], aatype, t, mask) for i in range(d)])
    assert jnp.allclose(energies, expected, atol=1e-5)


class TestScoreMutantEnsemble:
  def test_raw_per_mutant_energy_no_corrections(self) -> None:
    model = _make_model(jax.random.PRNGKey(4))
    k_coords, k_mut = jax.random.split(jax.random.PRNGKey(5))
    m = 3
    coords = jax.random.normal(k_coords, (N, 3)) * 0.1
    mutant_aatype = jax.random.randint(k_mut, (m, N), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.1)

    energies = score_mutant_ensemble(model, coords, mutant_aatype, t, mask, default_batch_size=8)
    expected = jnp.stack([model.energy(coords, mutant_aatype[i], t, mask) for i in range(m)])
    assert jnp.allclose(energies, expected, atol=1e-5)

  def test_wildtype_relative_ddg(self) -> None:
    model = _make_model(jax.random.PRNGKey(6))
    k_coords, k_mut, k_wt = jax.random.split(jax.random.PRNGKey(7), 3)
    m = 3
    coords = jax.random.normal(k_coords, (N, 3)) * 0.1
    mutant_aatype = jax.random.randint(k_mut, (m, N), 0, 21)
    wildtype_aatype = jax.random.randint(k_wt, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.1)

    ddg = score_mutant_ensemble(
      model,
      coords,
      mutant_aatype,
      t,
      mask,
      wildtype_aatype=wildtype_aatype,
      default_batch_size=8,
    )
    wt_energy = model.energy(coords, wildtype_aatype, t, mask)
    expected = jnp.stack(
      [model.energy(coords, mutant_aatype[i], t, mask) - wt_energy for i in range(m)],
    )
    assert jnp.allclose(ddg, expected, atol=1e-5)

  def test_full_ddg_with_unfolded_ensemble_mean_correction(self) -> None:
    model = _make_model(jax.random.PRNGKey(8))
    k_coords, k_mut, k_wt, k_ens = jax.random.split(jax.random.PRNGKey(9), 4)
    m, u = 2, 4
    coords = jax.random.normal(k_coords, (N, 3)) * 0.1
    mutant_aatype = jax.random.randint(k_mut, (m, N), 0, 21)
    wildtype_aatype = jax.random.randint(k_wt, (N,), 0, 21)
    ensemble_aatype = jax.random.randint(k_ens, (u, N), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.1)

    ddg = score_mutant_ensemble(
      model,
      coords,
      mutant_aatype,
      t,
      mask,
      wildtype_aatype=wildtype_aatype,
      unfolded_ensemble_aatype=ensemble_aatype,
      default_batch_size=8,
    )
    wt_energy = model.energy(coords, wildtype_aatype, t, mask)
    unfolded_mean = jnp.mean(
      jnp.stack([model.energy(coords, ensemble_aatype[i], t, mask) for i in range(u)]),
    )
    expected = jnp.stack(
      [
        model.energy(coords, mutant_aatype[i], t, mask) - wt_energy - unfolded_mean
        for i in range(m)
      ],
    )
    assert jnp.allclose(ddg, expected, atol=1e-5)


class TestScoreStateDifference:
  def test_two_state_difference_matches_manual_computation(self) -> None:
    model = _make_model(jax.random.PRNGKey(10))
    k_coords, k_aatype = jax.random.split(jax.random.PRNGKey(11))
    s = 2
    coords_states = jax.random.normal(k_coords, (s, N, 3)) * 0.1
    aatype = jax.random.randint(k_aatype, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.15)

    gap = score_state_difference(model, coords_states, aatype, t, mask, default_batch_size=4)
    expected = model.energy(coords_states[0], aatype, t, mask) - model.energy(
      coords_states[1], aatype, t, mask,
    )
    assert jnp.allclose(gap, expected, atol=1e-5)
