"""Additional coverage for the real ``aminx.ebm.readout`` API (backlog node E3).

Complements ``test_readout_invariants.py`` (the un-xfailed E0-authored
invariant tests) with coverage specific to this implementation: masked-residue
exclusion for both energy and score, ``AuxScoreReadout`` shape/no-grad-required
sanity, jit/vmap compatibility, and an end-to-end composition with a REAL
(not toy) small ``DiffusionTransformer`` from ``aminx.ebm.trunk`` (E1).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.ebm.readout import AuxScoreReadout, EnergyReadout, ScoreReadout
from aminx.ebm.trunk import DiffusionTransformer


def test_masked_residues_contribute_zero_energy() -> None:
  key = jax.random.PRNGKey(7)
  init_key, data_key = jax.random.split(key)
  trunk_dim = 6
  n = 5
  readout = EnergyReadout(trunk_dim=trunk_dim, key=init_key)
  trunk_out = jax.random.normal(data_key, (n, trunk_dim))
  mask = jnp.array([True, True, True, False, False])

  per_residue = readout.per_residue_energy(trunk_out, mask)
  assert jnp.allclose(per_residue[3:], 0.0)
  assert not jnp.allclose(per_residue[:3], 0.0)

  # Masking must also zero the *total* energy contribution: an all-False
  # mask over the same trunk_out gives exactly zero, whereas the partial
  # mask above is strictly positive (energies are sums of squares, so a
  # nonzero unmasked residue keeps the sum > 0).
  assert jnp.allclose(readout(trunk_out, jnp.zeros((n,), dtype=bool)), 0.0)
  assert readout(trunk_out, mask) > 0.0


def test_masked_residues_contribute_zero_score() -> None:
  key = jax.random.PRNGKey(8)
  init_key, data_key = jax.random.split(key)
  trunk_dim = 6
  n = 5
  readout = EnergyReadout(trunk_dim=trunk_dim, key=init_key)
  score_readout = ScoreReadout(readout)
  mask = jnp.array([True, True, True, False, False])

  def trunk_fn(coords: jax.Array, _mask: jax.Array) -> jax.Array:
    # Per-residue-only (no cross-residue mixing) trunk stand-in: pads
    # coords to trunk_dim, so any nonzero score at a masked residue can only
    # come from the energy readout's own masking, not cross terms leaking
    # in from unmasked residues via some (absent, here) attention mechanism.
    pad_width = trunk_dim - coords.shape[-1]
    return jnp.pad(coords, ((0, 0), (0, pad_width)))

  coords = jax.random.normal(data_key, (n, 3))
  score = score_readout(coords, mask, trunk_fn)
  assert jnp.allclose(score[3:], 0.0)
  assert not jnp.allclose(score[:3], 0.0)


def test_aux_score_readout_shape_and_direct_forward_no_grad_needed() -> None:
  key = jax.random.PRNGKey(9)
  init_key, data_key = jax.random.split(key)
  trunk_dim = 8
  n = 5
  aux = AuxScoreReadout(trunk_dim=trunk_dim, key=init_key)
  trunk_out = jax.random.normal(data_key, (n, trunk_dim))

  # Plain forward call -- no jax.grad wrapper required (contrast ScoreReadout).
  aux_score = aux(trunk_out)
  assert aux_score.shape == (n, 3)
  assert jnp.all(jnp.isfinite(aux_score))


def test_energy_readout_jit_and_vmap_compatible() -> None:
  key = jax.random.PRNGKey(42)
  trunk_dim = 8
  n = 5
  batch = 4
  readout = EnergyReadout(trunk_dim=trunk_dim, key=key)
  batch_trunk_out = jax.random.normal(jax.random.PRNGKey(43), (batch, n, trunk_dim))
  batch_mask = jnp.ones((batch, n), dtype=bool)

  jitted = eqx.filter_jit(readout.__call__)
  energies = jax.vmap(jitted)(batch_trunk_out, batch_mask)
  assert energies.shape == (batch,)
  assert jnp.all(jnp.isfinite(energies))
  assert jnp.all(energies >= 0.0)


def test_score_readout_jit_and_vmap_compatible() -> None:
  key = jax.random.PRNGKey(44)
  init_key, data_key = jax.random.split(key)
  trunk_dim = 8
  n = 5
  batch = 3
  energy_readout = EnergyReadout(trunk_dim=trunk_dim, key=init_key)
  score_readout = ScoreReadout(energy_readout)
  batch_coords = jax.random.normal(data_key, (batch, n, 3))
  mask = jnp.ones((n,), dtype=bool)

  def trunk_fn(coords: jax.Array, _mask: jax.Array) -> jax.Array:
    pad_width = trunk_dim - coords.shape[-1]
    return jnp.pad(coords, ((0, 0), (0, pad_width)))

  def score_one(coords: jax.Array) -> jax.Array:
    return score_readout(coords, mask, trunk_fn)

  jitted = eqx.filter_jit(score_one)
  scores = jax.vmap(jitted)(batch_coords)
  assert scores.shape == (batch, n, 3)
  assert jnp.all(jnp.isfinite(scores))


def test_end_to_end_with_real_diffusion_transformer_trunk() -> None:
  """EnergyReadout+ScoreReadout composed with a REAL (not toy) small trunk.

  Wires coords -> (fixed linear embed) -> a real E1 ``DiffusionTransformer``
  forward pass -> ``EnergyReadout`` -> scalar energy, then differentiates
  through the *entire* composition (embed + attention + AdaLN + transitions)
  w.r.t. ``coords`` to get the conservative score. Asserts finite,
  correctly-shaped output at both stages.
  """
  key = jax.random.PRNGKey(100)
  embed_key, trunk_key, energy_key, coord_key = jax.random.split(key, 4)

  dim = 8
  dim_pairwise = 4
  n = 5

  embed = eqx.nn.Linear(3, dim, key=embed_key)
  trunk = DiffusionTransformer(depth=1, heads=2, dim=dim, dim_pairwise=dim_pairwise, key=trunk_key)
  energy_readout = EnergyReadout(trunk_dim=dim, key=energy_key)
  score_readout = ScoreReadout(energy_readout)

  def trunk_fn(coords: jax.Array, mask: jax.Array) -> jax.Array:
    a0 = jax.vmap(embed)(coords)
    z = jnp.zeros((n, n, dim_pairwise))
    return trunk(a0, a0, z, mask)

  coords = jax.random.normal(coord_key, (n, 3))
  mask = jnp.ones((n,), dtype=bool)

  energy = score_readout.energy(coords, mask, trunk_fn)
  assert energy.shape == ()
  assert jnp.isfinite(energy)
  assert energy >= 0.0

  score = score_readout(coords, mask, trunk_fn)
  assert score.shape == (n, 3)
  assert jnp.all(jnp.isfinite(score))
