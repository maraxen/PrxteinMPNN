"""E3 readout invariants (EnergyReadout/ScoreReadout), backlog node E0 test harness.

The real ``EnergyReadout``/``ScoreReadout`` land in backlog node E3 (design
spec §3.1: ``EnergyReadout(a, mask) -> E``; ``ScoreReadout -> -jax.grad(E)``).
These tests were originally written speculatively against that intended API
(before E3 existed) and marked ``xfail(reason="E3 not implemented",
strict=False)``. Now that ``aminx.ebm.readout`` is implemented, the xfail
markers are removed. The real ``ScoreReadout`` takes its ``trunk_fn`` (the
composition seam to E1's ``DiffusionTransformer``) as a **call-time**
argument rather than a stored constructor field (see
``aminx.ebm.readout.ScoreReadout``'s docstring for why) -- tests 3 and 4
below were adjusted accordingly, preserving the exact invariant each was
written to check (score = -grad(energy); the outer training grad through
that nested grad is finite).

The ``TestToyEnergyInvariants``/``TestSecondOrderTrainingGradOnToy`` classes
prove the *same* invariants against an inline toy energy
``E(x) = sum_i ||W @ x_i||**2`` (``toy_energy_fn`` fixture, ``conftest.py``)
so the harness itself is validated *now*, independent of the pending model
port -- the BATHOS "verify the measurement pipeline before trusting any
research conclusion" discipline (``~/.claude/rules/BATHOS.md``).

Invariant (d) is new per design spec §1 Fork 2 / EPIC risk MAJOR-6: the
conservative score is itself a nested ``jax.grad`` (``score = -grad_x E``),
and E8's training loss differentiates *through* that score
(reverse-over-reverse AD). That second-order path has zero existing
precedent in aminx and must be gated by a synthetic finite-difference check
before any real 2nd-order training grad is trusted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# ---------------------------------------------------------------------------
# (a)-(d) against the real, not-yet-implemented E3 API.
# ---------------------------------------------------------------------------


def test_energy_readout_zero_when_head_output_zero() -> None:
  from aminx.ebm.readout import EnergyReadout  # noqa: PLC0415

  key = jax.random.PRNGKey(0)
  readout = EnergyReadout(trunk_dim=8, key=key)
  trunk_out = jnp.zeros((5, 8))
  mask = jnp.ones((5,), dtype=bool)

  energy = readout(trunk_out, mask)
  assert jnp.allclose(energy, 0.0)


def test_energy_readout_is_nonnegative() -> None:
  from aminx.ebm.readout import EnergyReadout  # noqa: PLC0415

  key = jax.random.PRNGKey(1)
  init_key, data_key = jax.random.split(key)
  readout = EnergyReadout(trunk_dim=8, key=init_key)
  trunk_out = jax.random.normal(data_key, (5, 8))
  mask = jnp.ones((5,), dtype=bool)

  energy = readout(trunk_out, mask)
  assert energy >= 0.0


def test_score_readout_is_negative_grad_of_energy() -> None:
  from aminx.ebm.readout import EnergyReadout, ScoreReadout  # noqa: PLC0415

  key = jax.random.PRNGKey(2)
  init_key, data_key = jax.random.split(key)
  energy_readout = EnergyReadout(trunk_dim=8, key=init_key)
  score_readout = ScoreReadout(energy_readout)
  coords = jax.random.normal(data_key, (5, 3))
  mask = jnp.ones((5,), dtype=bool)

  def trunk_fn(x: jax.Array, _mask: jax.Array) -> jax.Array:
    return jnp.zeros((5, 8)) + x.sum()  # placeholder trunk stand-in

  def energy_of_coords(x: jax.Array) -> jax.Array:
    return energy_readout(trunk_fn(x, mask), mask)

  expected_score = -jax.grad(energy_of_coords)(coords)
  actual_score = score_readout(coords, mask, trunk_fn)
  assert jnp.allclose(actual_score, expected_score)


def test_second_order_training_grad_is_finite_on_real_model() -> None:
  import equinox as eqx  # noqa: PLC0415

  from aminx.ebm.readout import EnergyReadout, ScoreReadout  # noqa: PLC0415

  key = jax.random.PRNGKey(3)
  init_key, data_key = jax.random.split(key)
  energy_readout = EnergyReadout(trunk_dim=8, key=init_key)
  coords = jax.random.normal(data_key, (5, 3))
  mask = jnp.ones((5,), dtype=bool)
  target = jax.random.normal(jax.random.PRNGKey(4), (5, 3))

  def trunk_fn(x: jax.Array, _mask: jax.Array) -> jax.Array:
    return jnp.pad(x, ((0, 0), (0, 5)))  # placeholder trunk stand-in, (5,3) -> (5,8)

  def loss_fn(model: EnergyReadout) -> jax.Array:
    score_readout = ScoreReadout(model)
    predicted = score_readout(coords, mask, trunk_fn)
    return jnp.sum((predicted - target) ** 2)

  grads = eqx.filter_grad(loss_fn)(energy_readout)
  leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
  assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


# ---------------------------------------------------------------------------
# Same invariants, proven now on the inline toy energy (harness validation).
# ---------------------------------------------------------------------------


class TestToyEnergyInvariants:
  """(a)-(c) proven against ``E(x) = sum_i ||W @ x_i||**2``."""

  def test_zero_energy_when_head_output_zero(self, toy_energy_fn) -> None:
    w = jax.random.normal(jax.random.PRNGKey(10), (3, 3))
    x = jnp.zeros((6, 3))  # zero input -> r_i = W @ x_i == 0 for all i
    mask = jnp.ones((6,), dtype=bool)

    energy = toy_energy_fn(w, x, mask)
    assert jnp.allclose(energy, 0.0)

  @pytest.mark.parametrize("seed", [0, 1, 2, 3])
  def test_energy_is_always_nonnegative(self, toy_energy_fn, seed: int) -> None:
    key = jax.random.PRNGKey(seed)
    k_w, k_x = jax.random.split(key)
    w = jax.random.normal(k_w, (4, 3))
    x = jax.random.normal(k_x, (6, 3)) * 5.0  # wide range, incl. negative components
    mask = jnp.ones((6,), dtype=bool)

    energy = toy_energy_fn(w, x, mask)
    assert energy >= 0.0

  def test_score_equals_negative_grad_of_toy_energy(self, toy_energy_fn) -> None:
    key = jax.random.PRNGKey(20)
    k_w, k_x = jax.random.split(key)
    w = jax.random.normal(k_w, (3, 3))
    x = jax.random.normal(k_x, (6, 3))
    mask = jnp.ones((6,), dtype=bool)

    score = -jax.grad(lambda xx: toy_energy_fn(w, xx, mask))(x)

    # Analytic: dE/dx_i = 2 * mask_i * (W^T W) @ x_i => score_i = -2 * mask_i * (W^T W) @ x_i
    wtw = w.T @ w
    analytic_score = -2.0 * jnp.einsum("de,ne->nd", wtw, x) * mask.astype(x.dtype)[:, None]

    assert jnp.allclose(score, analytic_score, atol=1e-5, rtol=1e-5)

  def test_masked_residues_do_not_contribute_to_score(self, toy_energy_fn) -> None:
    key = jax.random.PRNGKey(21)
    k_w, k_x = jax.random.split(key)
    w = jax.random.normal(k_w, (3, 3))
    x = jax.random.normal(k_x, (6, 3))
    mask = jnp.array([True, True, True, False, False, False])

    score = -jax.grad(lambda xx: toy_energy_fn(w, xx, mask))(x)
    assert jnp.allclose(score[3:], 0.0)
    assert not jnp.allclose(score[:3], 0.0)


class TestSecondOrderTrainingGradOnToy:
  """(d) NEW 2nd-order gate (Fork 2).

  An outer training grad through a nested ``jax.grad(jax.grad(...))``-style
  score -- the exact pattern E8's training loop needs (``score = -grad(E)``;
  the loss differentiates *through* that score w.r.t. model parameters,
  i.e. reverse-over-reverse AD) -- must be finite/non-NaN and must match a
  finite-difference approximation of the same loss.
  """

  def test_outer_grad_is_finite_and_matches_finite_difference(
    self,
    toy_energy_fn,
    enable_x64,
  ) -> None:
    key = jax.random.PRNGKey(30)
    k_w, k_x, k_t = jax.random.split(key, 3)
    w0 = jax.random.normal(k_w, (3, 3), dtype=jnp.float64) * 0.5
    x = jax.random.normal(k_x, (4, 3), dtype=jnp.float64)
    mask = jnp.ones((4,), dtype=bool)
    target = jax.random.normal(k_t, (4, 3), dtype=jnp.float64)

    def score_fn(w: jax.Array) -> jax.Array:
      # -grad_x E, then differentiated again w.r.t. w in loss_fn below:
      # reverse-over-reverse, the exact 2nd-order AD pattern Fork 2 gates.
      return -jax.grad(lambda xx: toy_energy_fn(w, xx, mask))(x)

    def loss_fn(w: jax.Array) -> jax.Array:
      predicted = score_fn(w)
      return jnp.sum((predicted - target) ** 2)

    outer_grad = jax.grad(loss_fn)(w0)
    assert jnp.all(jnp.isfinite(outer_grad))

    eps = 1e-4
    flat_w0 = w0.reshape(-1)
    numerical_flat = []
    for i in range(flat_w0.shape[0]):
      bump = jnp.zeros_like(flat_w0).at[i].set(eps)
      w_plus = (flat_w0 + bump).reshape(w0.shape)
      w_minus = (flat_w0 - bump).reshape(w0.shape)
      numerical_flat.append((loss_fn(w_plus) - loss_fn(w_minus)) / (2 * eps))
    numerical_grad = jnp.array(numerical_flat).reshape(w0.shape)

    assert jnp.allclose(outer_grad, numerical_grad, atol=1e-3, rtol=1e-3)

  def test_outer_grad_is_nan_free_near_t_like_degenerate_input(
    self,
    toy_energy_fn,
    enable_x64,
  ) -> None:
    # Degenerate-ish input (near-zero x) is the toy analog of the t->0
    # singularity the diffusion invariants gate; the 2nd-order path must
    # not silently produce NaN there either.
    key = jax.random.PRNGKey(31)
    k_w, k_t = jax.random.split(key)
    w0 = jax.random.normal(k_w, (3, 3), dtype=jnp.float64) * 0.5
    x = jnp.full((4, 3), 1e-8, dtype=jnp.float64)
    mask = jnp.ones((4,), dtype=bool)
    target = jax.random.normal(k_t, (4, 3), dtype=jnp.float64)

    def loss_fn(w: jax.Array) -> jax.Array:
      predicted = -jax.grad(lambda xx: toy_energy_fn(w, xx, mask))(x)
      return jnp.sum((predicted - target) ** 2)

    outer_grad = jax.grad(loss_fn)(w0)
    assert jnp.all(jnp.isfinite(outer_grad))
