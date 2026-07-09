"""Synthetic ground-truth invariants for ``aminx.ebm.diffusion`` (VP-SDE math), backlog node E0.

Per ``~/.claude/rules/BATHOS.md`` ("verify your measurement pipeline before
trusting research conclusions"): these are the 30-second sanity checks that
must pass *now*, before any parity claim about the ported energy model is
trusted downstream (E3.5 weight-port gate, E5/E6/E7 Spearman gates). They
pin the exact VP-SDE sign/scaling convention against
``~/repos/ProteinEBM/protein_ebm/model/r3_diffuser.py`` (design spec §4:
"feed the VP-SDE score closed form a known (x_0, t) and assert it matches
-(x_t - sqrt(alpha_bar) x_0)/(1 - alpha_bar)").
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.ebm.diffusion import (
  DEFAULT_COORDINATE_SCALING,
  DEFAULT_MAX_B,
  DEFAULT_MIN_B,
  calc_trans_0,
  conditional_var,
  forward_marginal,
  marginal_b_t,
  score_target,
)

# t values bounded away from 0, where the `conditional_var` epsilon guard
# (`_VAR_EPS = 1e-6`) would otherwise perturb exact-recovery/analytic-match
# assertions. t=0 NaN-safety is checked separately below.
_NONDEGENERATE_TS = (0.05, 0.2, 0.5, 0.8, 1.0)


def _mean_and_var(x0: jax.Array, t: float) -> tuple[jax.Array, jax.Array]:
  """Reference closed form, computed independently of the module under test."""
  mbt = t * DEFAULT_MIN_B + 0.5 * (t**2) * (DEFAULT_MAX_B - DEFAULT_MIN_B)
  mean = jnp.exp(-0.5 * mbt) * x0
  var = 1.0 - jnp.exp(-mbt)
  return mean, var


class TestMarginalScheduleFormulas:
  """Pin the VP-SDE (not VE) schedule formulas directly against hand-computed values."""

  @pytest.mark.parametrize("t", _NONDEGENERATE_TS)
  def test_marginal_b_t_matches_hand_computed_integral(self, t: float) -> None:
    expected = t * DEFAULT_MIN_B + 0.5 * (t**2) * (DEFAULT_MAX_B - DEFAULT_MIN_B)
    actual = marginal_b_t(jnp.array(t))
    assert jnp.allclose(actual, expected, atol=1e-7)

  @pytest.mark.parametrize("t", _NONDEGENERATE_TS)
  def test_conditional_var_is_one_minus_exp_neg_marginal(self, t: float) -> None:
    mbt = marginal_b_t(jnp.array(t))
    expected = 1.0 - jnp.exp(-mbt)
    actual = conditional_var(jnp.array(t))
    assert jnp.allclose(actual, expected, atol=1e-7)

  def test_conditional_var_is_zero_at_t_zero(self) -> None:
    # Sharpest possible check that this is VP (bounded variance schedule
    # starting at 0), not VE (which would diverge/be unbounded at t=0).
    assert jnp.allclose(conditional_var(jnp.array(0.0)), 0.0, atol=1e-9)

  def test_conditional_var_is_bounded_in_zero_one(self) -> None:
    ts = jnp.linspace(0.0, 1.0, 50)
    var = jax.vmap(conditional_var)(ts)
    assert jnp.all(var >= -1e-9)
    assert jnp.all(var <= 1.0 + 1e-6)


class TestScoreTargetInvariants:
  """Invariant (a): score_target equals the analytic Gaussian score."""

  @pytest.mark.parametrize("t", _NONDEGENERATE_TS)
  def test_matches_analytic_gaussian_score(self, t: float) -> None:
    key = jax.random.PRNGKey(0)
    k_x0, k_xt = jax.random.split(key)
    x0 = jax.random.normal(k_x0, (7, 3))
    x_t = jax.random.normal(k_xt, (7, 3))

    mean, var = _mean_and_var(x0, t)
    expected = -(x_t - mean) / var

    actual = score_target(x_t, x0, jnp.array(t))
    assert jnp.allclose(actual, expected, atol=1e-6, rtol=1e-6)

  def test_score_target_is_finite_and_nan_safe_at_t_zero(self) -> None:
    # NaN-safety gate: conditional_var(0) == 0 exactly, so the reciprocal in
    # score_target must be epsilon-guarded rather than producing NaN/Inf.
    key = jax.random.PRNGKey(1)
    k_x0, k_xt = jax.random.split(key)
    x0 = jax.random.normal(k_x0, (7, 3))
    x_t = jax.random.normal(k_xt, (7, 3))

    actual = score_target(x_t, x0, jnp.array(0.0))
    assert jnp.all(jnp.isfinite(actual))


class TestCalcTrans0Invariants:
  """Invariant (b): calc_trans_0(score_target(x_t, x0, t), x_t, t) recovers x0."""

  @pytest.mark.parametrize("t", _NONDEGENERATE_TS)
  def test_recovers_x0_to_tolerance(self, t: float) -> None:
    key = jax.random.PRNGKey(2)
    k_x0, k_xt = jax.random.split(key)
    x0 = jax.random.normal(k_x0, (11, 3))
    x_t = jax.random.normal(k_xt, (11, 3))

    score = score_target(x_t, x0, jnp.array(t))
    recovered = calc_trans_0(score, x_t, jnp.array(t))

    # float32-safe tolerance: the algebraic identity is exact, but chained
    # exp/divide ops accumulate ~1e-5 rounding error at the larger t values.
    assert jnp.allclose(recovered, x0, atol=1e-4, rtol=1e-4)

  def test_recovers_x0_across_a_batch_of_structures(self) -> None:
    key = jax.random.PRNGKey(3)
    k_x0, k_xt = jax.random.split(key)
    x0 = jax.random.normal(k_x0, (4, 20, 3))
    x_t = jax.random.normal(k_xt, (4, 20, 3))
    t = jnp.full((4,), 0.3)

    score = jax.vmap(score_target)(x_t, x0, t)
    recovered = jax.vmap(calc_trans_0)(score, x_t, t)

    assert jnp.allclose(recovered, x0, atol=1e-4, rtol=1e-4)


class TestForwardMarginalInvariants:
  """Invariant (c): forward_marginal is deterministic under a fixed key and

  matches the closed-form mean/variance empirically over many samples.
  Invariant (d): forward_marginal (and the pure schedule fns) are vmappable
  over a batch of t.
  """

  def test_deterministic_under_fixed_key(self) -> None:
    key = jax.random.PRNGKey(42)
    x0 = jax.random.normal(jax.random.PRNGKey(7), (9, 3))
    t = jnp.array(0.4)

    x_t_1, score_1 = forward_marginal(x0, t, key)
    x_t_2, score_2 = forward_marginal(x0, t, key)

    assert jnp.array_equal(x_t_1, x_t_2)
    assert jnp.array_equal(score_1, score_2)

  def test_different_keys_give_different_samples(self) -> None:
    x0 = jax.random.normal(jax.random.PRNGKey(7), (9, 3))
    t = jnp.array(0.4)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(11))

    x_t_a, _ = forward_marginal(x0, t, key_a)
    x_t_b, _ = forward_marginal(x0, t, key_b)

    assert not jnp.array_equal(x_t_a, x_t_b)

  @pytest.mark.parametrize("t", [0.1, 0.5, 0.9])
  def test_empirical_mean_and_var_match_closed_form(self, t: float) -> None:
    num_samples = 20_000
    x0 = jnp.array([[1.0, -2.0, 0.5]])  # single residue, easy to reason about
    keys = jax.random.split(jax.random.PRNGKey(123), num_samples)

    x_t_samples, _ = jax.vmap(lambda k: forward_marginal(x0, jnp.array(t), k))(keys)
    # x_t_samples: [num_samples, 1, 3]

    empirical_mean = jnp.mean(x_t_samples, axis=0)
    empirical_var = jnp.var(x_t_samples, axis=0)

    x0_scaled = x0 * DEFAULT_COORDINATE_SCALING
    expected_mean, expected_var = _mean_and_var(x0_scaled, t)

    assert jnp.allclose(empirical_mean, expected_mean, atol=0.05)
    assert jnp.allclose(empirical_var, expected_var, atol=0.05)

  def test_coordinate_scaling_applied_exactly_once(self) -> None:
    # The mean of x_t must reflect x0 * coordinate_scaling, not x0 directly --
    # pins the single-scaling-boundary design decision (spec §10 MINOR
    # finding on coordinate_scaling double-application).
    key = jax.random.PRNGKey(99)
    x0 = jnp.array([[10.0, 0.0, 0.0]])
    t = jnp.array(1e-6)  # ~0 variance -> x_t approx == mean

    x_t, _ = forward_marginal(x0, t, key, coordinate_scaling=0.1)

    # At t -> 0, mean -> coordinate_scaling * x0 (exp(-0.5*mbt) -> 1).
    assert jnp.allclose(x_t, x0 * 0.1, atol=1e-3)

  def test_vmap_over_batch_of_t(self) -> None:
    x0 = jax.random.normal(jax.random.PRNGKey(5), (6, 3))
    ts = jnp.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    keys = jax.random.split(jax.random.PRNGKey(6), ts.shape[0])

    x_t_batch, score_batch = jax.vmap(
      forward_marginal,
      in_axes=(None, 0, 0),
    )(x0, ts, keys)

    assert x_t_batch.shape == (ts.shape[0], *x0.shape)
    assert score_batch.shape == (ts.shape[0], *x0.shape)
    assert jnp.all(jnp.isfinite(x_t_batch))
    assert jnp.all(jnp.isfinite(score_batch))

    # Cross-check against sequential (non-vmapped) calls for a couple of
    # indices -- vmap must not silently change the math.
    for i in (0, 3, 5):
      x_t_i, score_i = forward_marginal(x0, ts[i], keys[i])
      assert jnp.allclose(x_t_batch[i], x_t_i, atol=1e-6)
      assert jnp.allclose(score_batch[i], score_i, atol=1e-6)

  def test_jit_compatible(self) -> None:
    jitted = jax.jit(forward_marginal)
    x0 = jax.random.normal(jax.random.PRNGKey(8), (5, 3))
    t = jnp.array(0.3)
    key = jax.random.PRNGKey(9)

    x_t_eager, score_eager = forward_marginal(x0, t, key)
    x_t_jit, score_jit = jitted(x0, t, key)

    assert jnp.allclose(x_t_eager, x_t_jit, atol=1e-6)
    assert jnp.allclose(score_eager, score_jit, atol=1e-6)
