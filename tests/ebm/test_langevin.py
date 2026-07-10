"""Synthetic-invariant + jit/vmap tests for ``aminx.ebm.langevin`` (backlog node **E9**, CORE).

Per ``~/.claude/rules/BATHOS.md`` ("verify your measurement pipeline before
trusting any research conclusion"): these are non-vacuous ground-truth checks
on a toy energy landscape, not just shape/smoke tests. Scope matches the
dispatch: a single, fixed noise level ``t`` against a single (real or toy)
model; the outer noise-schedule/model-swap loop is out of scope and not
tested here.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.ebm.diffusion import beta_t, drift_coef
from aminx.ebm.diffusion import diffusion_coef as module_diffusion_coef
from aminx.ebm.langevin import langevin_step, metropolis_hastings_step, run_langevin_equilibration
from aminx.ebm.model import ProteinEBMModel

N = 5
AATYPE = jnp.zeros((N,), dtype=jnp.int32)
MASK = jnp.ones((N,), dtype=bool)


class _ToyCenteredModel:
  """Toy quadratic energy, translation-invariant by construction.

  ``energy(x) = 0.5 * k * ||center(x) - target_centered||**2`` where
  ``center(x) = x - mean(x)``. This is deliberately made invariant to any
  *rigid* translation of ``x`` -- necessary because :func:`langevin_step`
  always re-centers **and translates** (``center_random_augmentation``'s
  default ``augmentation=True``, only ``rotate=False``) after every step;
  a translation-*variant* toy energy (bare ``||x - target||**2``) would be
  dominated by that per-step random rigid shift and could never show a
  meaningful convergence trend. A real ``ProteinEBMModel`` energy is itself
  effectively translation-invariant for the same structural reason (its
  featurization has no absolute-position channel), so this toy is a faithful
  stand-in, not a special-cased workaround.

  ``aux_score`` is ``-jax.grad(energy)`` computed on this toy via exact
  autodiff (affordable for a toy quadratic; this is the "synthetic ground
  truth" construction the task calls for, not a claim about the real
  ``AuxScoreReadout``, which is trained, not differentiated at inference).
  """

  def __init__(self, target_centered: jax.Array, k: float = 1.0) -> None:
    self.target_centered = target_centered
    self.k = k

  def _centered_energy(self, coords: jax.Array) -> jax.Array:
    mean = jnp.mean(coords, axis=0)
    centered = coords - mean
    diff = centered - self.target_centered
    return 0.5 * self.k * jnp.sum(diff**2)

  def energy(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return self._centered_energy(coords)

  def aux_score(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return -jax.grad(self._centered_energy)(coords)


class _ToyFlatModel:
  """Degenerate toy model: energy identically zero everywhere -> score identically zero.

  Used for the "always accept" Metropolis-Hastings sanity check: with a flat
  energy landscape, ``energy_ratio == 1`` and ``kernel_ratio == 1`` exactly
  (not merely approximately) -- see ``TestMetropolisHastingsDetailedBalance``
  for the closed-form derivation.
  """

  def energy(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return jnp.zeros(()) * jnp.sum(coords)  # zero, but keeps coords in the trace

  def aux_score(self, coords: jax.Array, aatype: jax.Array, t: jax.Array, mask: jax.Array) -> jax.Array:
    del aatype, t, mask
    return jnp.zeros_like(coords)


class TestDriftDiffusionCoefSanity:
  """(1) shape/value sanity for the new diffusion.py additions vs. the already-tested beta_t."""

  @pytest.mark.parametrize("t_val", [0.0, 0.05, 0.3, 0.7, 1.0])
  def test_diffusion_coef_is_sqrt_beta_t(self, t_val: float) -> None:
    t = jnp.array(t_val)
    assert jnp.allclose(module_diffusion_coef(t), jnp.sqrt(beta_t(t)), atol=1e-6)

  @pytest.mark.parametrize("t_val", [0.05, 0.3, 0.7])
  def test_drift_coef_matches_hand_computed_formula(self, t_val: float) -> None:
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (N, 3))
    t = jnp.array(t_val)
    expected = -0.5 * beta_t(t) * x
    assert jnp.allclose(drift_coef(x, t), expected, atol=1e-6)

  def test_drift_coef_shape_matches_input(self) -> None:
    x = jnp.ones((N, 3))
    out = drift_coef(x, jnp.array(0.2))
    assert out.shape == x.shape

  def test_diffusion_coef_is_scalar_and_nonnegative(self) -> None:
    out = module_diffusion_coef(jnp.array(0.4))
    assert out.shape == ()
    assert out >= 0.0


class TestLangevinStepConvergence:
  """(2) Non-vacuous convergence sanity on a toy quadratic energy landscape.

  Parameters (t=0.01, dt=2e-4, effective_temp_scaling=3.0, k=20.0, n=300)
  were tuned empirically (see dispatch notes) specifically so the
  score-following drift dominates the per-step diffusive noise (the
  ``coordinate_scaling`` division amplifies injected noise ~10x, so this
  is not a free "just run it and it converges" default) over the fixed
  seed below. This is a real, non-monotonic-per-step but clearly
  trending-downward trajectory -- not a smoke test.
  """

  def test_energy_and_distance_decrease_over_trajectory(self) -> None:
    t_val, dt, temp_scale, n_steps, k = 0.01, 2e-4, 3.0, 300, 20.0
    key = jax.random.PRNGKey(0)
    k_target, k_start, k_run = jax.random.split(key, 3)
    target = jax.random.normal(k_target, (N, 3)) * 0.02
    target_centered = target - jnp.mean(target, axis=0)
    model = _ToyCenteredModel(target_centered, k=k)

    coords0 = jax.random.normal(k_start, (N, 3)) * 0.5
    t = jnp.array(t_val)

    def dist(coords: jax.Array) -> jax.Array:
      return jnp.linalg.norm((coords - jnp.mean(coords, axis=0)) - target_centered)

    e0 = float(model.energy(coords0, AATYPE, t, MASK))
    d0 = float(dist(coords0))

    coords = coords0
    energies = []
    keys = jax.random.split(k_run, n_steps)
    for step_key in keys:
      coords = langevin_step(model, coords, AATYPE, t, dt, MASK, step_key, effective_temp_scaling=temp_scale)
      energies.append(model.energy(coords, AATYPE, t, MASK))
    energies = jnp.stack(energies)

    e_final = float(energies[-1])
    d_final = float(dist(coords))
    q = n_steps // 10
    first_decile_mean = float(jnp.mean(energies[:q]))
    last_decile_mean = float(jnp.mean(energies[-q:]))

    # Report the actual numbers (see final dispatch report).
    print(
      f"E0={e0:.4f} Ef={e_final:.4f} d0={d0:.4f} df={d_final:.4f} "
      f"first_decile_mean={first_decile_mean:.4f} last_decile_mean={last_decile_mean:.4f}"
    )

    assert jnp.all(jnp.isfinite(energies))
    # Non-vacuous: energy at the end of the run is substantially lower than
    # at the start (not just "didn't blow up").
    assert e_final < 0.5 * e0
    assert d_final < 0.5 * d0
    # Trend, not just endpoints: late-trajectory energy is clearly lower
    # than early-trajectory energy on average.
    assert last_decile_mean < 0.7 * first_decile_mean


class TestMetropolisHastingsDetailedBalance:
  """(3) MH accept-ratio sanity: independent recomputation, not "it runs without erroring"."""

  def test_flat_energy_gives_exact_unit_accept_ratio_and_always_accepts(self) -> None:
    """Closed-form check: a flat (zero everywhere) energy landscape forces accept_ratio == 1 exactly.

    Derivation (matches ``metropolis_hastings_step``'s docstring formula):
    with ``score == 0`` everywhere, ``next_mean == pos_t`` (no drift), so
    ``pos_t - proposed_next_mean == pos_t - pos_t_proposed == -next_noise``,
    i.e. ``(pos_t - proposed_next_mean)**2.sum() == next_noise**2.sum()``
    *exactly* -> ``kernel_ratio == exp(0) == 1``. And
    ``energy_proposed - energy_current == 0`` exactly -> ``energy_ratio ==
    1``. So ``accept_ratio == 1`` for *every* draw, and since
    ``jax.random.uniform`` samples strictly on ``[0, 1)``, ``accept`` must be
    ``True`` on every single draw -- not just "statistically likely".
    """
    model = _ToyFlatModel()
    t = jnp.array(0.3)
    dt = 0.01
    coords = jax.random.normal(jax.random.PRNGKey(1), (N, 3)) * 0.4

    keys = jax.random.split(jax.random.PRNGKey(2), 500)

    def run_one(key: jax.Array) -> jax.Array:
      _coords_new, _energy_new, _score_new, accepted = metropolis_hastings_step(model, coords, AATYPE, t, dt, MASK, key)
      return accepted

    accepted_all = jax.vmap(run_one)(keys)
    accept_fraction = float(jnp.mean(accepted_all.astype(jnp.float32)))
    print(f"flat-energy accept_fraction={accept_fraction} (expected exactly 1.0)")
    assert accept_fraction == 1.0
    assert bool(jnp.all(accepted_all))

  def test_engineered_mostly_reject_scenario_matches_independent_recomputation(self) -> None:
    """Steep toy quadratic + moderate dt -> partial acceptance; cross-check against a hand-rewritten ratio.

    The independent recomputation below is written from scratch in this test
    (not by importing/calling into ``metropolis_hastings_step``'s internals)
    -- it re-derives ``accept_ratio`` from the same closed-form formula given
    in the module docstring, using the *same* PRNG key-splitting order
    (``key_noise, key_accept = jax.random.split(key)``) so that, for a given
    key, both computations see identical Gaussian/uniform draws. This is a
    reproducibility+correctness cross-check on the formula transcription
    (catches sign/factor/summation-axis bugs), not an independent
    closed-form derivation of the MALA acceptance probability from first
    principles (that would require a separate, larger derivation for a
    general energy landscape) -- see the dispatch report for this scoping
    note.
    """
    k, dt, t_val, temp_scale, coordinate_scaling = 0.2, 1e-3, 0.5, 1.0, 0.1
    t = jnp.array(t_val)
    target_centered = jnp.zeros((N, 3))
    model = _ToyCenteredModel(target_centered, k=k)
    coords = jax.random.normal(jax.random.PRNGKey(3), (N, 3)) * 0.3

    def independent_ratio_and_accept(key: jax.Array) -> tuple[jax.Array, jax.Array]:
      key_noise, key_accept = jax.random.split(key)
      g = module_diffusion_coef(t)
      score_current = model.aux_score(coords, AATYPE, t, MASK)
      energy_current = model.energy(coords, AATYPE, t, MASK)
      next_mean = coords + (g**2) * score_current * temp_scale / coordinate_scaling * dt
      noise = jax.random.normal(key_noise, shape=coords.shape)
      next_noise = jnp.sqrt(2.0) * g * jnp.sqrt(dt) * noise / coordinate_scaling
      proposed = next_mean + next_noise
      score_proposed = model.aux_score(proposed, AATYPE, t, MASK)
      energy_proposed = model.energy(proposed, AATYPE, t, MASK)
      proposed_next_mean = proposed + (g**2) * score_proposed * temp_scale / coordinate_scaling * dt
      energy_ratio = jnp.exp(-(energy_proposed - energy_current) * temp_scale)
      num = jnp.sum((coords - proposed_next_mean) ** 2) - jnp.sum(next_noise**2)
      denom = 2.0 * (g**2) * dt / (coordinate_scaling**2)
      kernel_ratio = jnp.exp(-num / denom / 2.0)
      ratio = energy_ratio * kernel_ratio
      u = jax.random.uniform(key_accept, shape=())
      return ratio, u < ratio

    keys = jax.random.split(jax.random.PRNGKey(4), 4000)
    ratios_independent, accept_independent = jax.vmap(independent_ratio_and_accept)(keys)

    def run_actual(key: jax.Array) -> jax.Array:
      _c, _e, _s, accepted = metropolis_hastings_step(
        model,
        coords,
        AATYPE,
        t,
        dt,
        MASK,
        key,
        effective_temp_scaling=temp_scale,
        coordinate_scaling=coordinate_scaling,
      )
      return accepted

    accept_actual = jax.vmap(run_actual)(keys)

    # (a) Per-draw exact reproducibility: identical keys -> identical
    # accept/reject decisions between the module and the independently
    # rewritten formula.
    assert bool(jnp.array_equal(accept_actual, accept_independent))

    empirical_fraction = float(jnp.mean(accept_actual.astype(jnp.float32)))
    expected_fraction = float(jnp.mean(jnp.minimum(1.0, ratios_independent)))
    print(
      f"mostly-reject scenario: empirical_accept_fraction={empirical_fraction:.4f} "
      f"expected(E[min(1,ratio)])={expected_fraction:.4f}"
    )

    # (b) Non-degenerate: this is a genuinely *partial*-acceptance regime,
    # not an accidental always-accept/always-reject case.
    assert 0.05 < empirical_fraction < 0.6
    # (c) Statistical consistency between the empirical acceptance fraction
    # and the analytically-expected-value estimate (mean of min(1, ratio)),
    # over many independent draws. `accept` is itself a further Bernoulli
    # draw from `ratio` (via the uniform comparison), so the two quantities
    # are expected to differ by O(binomial sampling noise), not be bitwise
    # equal -- with n=4000 draws and p~0.16, the binomial std is
    # sqrt(p*(1-p)/n) ~= 0.0058, so a tolerance of 5 std (~0.03) is a
    # genuine statistical-consistency bound, not a rubber-stamp.
    binomial_std = (expected_fraction * (1 - expected_fraction) / len(keys)) ** 0.5
    assert abs(empirical_fraction - expected_fraction) < 5 * binomial_std


class TestWhileLoopVsScanEquivalence:
  """(4) lax.while_loop-based run_langevin_equilibration vs. a test-only lax.scan reference."""

  @staticmethod
  def _scan_reference(
    model: _ToyCenteredModel,
    coords0: jax.Array,
    t: jax.Array,
    dt: float,
    n_steps: int,
    key0: jax.Array,
    *,
    use_metropolis: bool,
  ) -> jax.Array:
    """Throwaway lax.scan reimplementation of the same n_steps recurrence, test-only."""

    def step(carry: tuple[jax.Array, jax.Array], _xs: None) -> tuple[tuple[jax.Array, jax.Array], None]:
      coords, key = carry
      key, step_key = jax.random.split(key)
      if use_metropolis:
        coords_next, _e, _s, _a = metropolis_hastings_step(model, coords, AATYPE, t, dt, MASK, step_key)
      else:
        coords_next = langevin_step(model, coords, AATYPE, t, dt, MASK, step_key)
      return (coords_next, key), None

    (final_coords, _final_key), _ = jax.lax.scan(step, (coords0, key0), xs=None, length=n_steps)
    return final_coords

  @pytest.mark.parametrize("use_metropolis", [False, True])
  def test_while_loop_matches_scan_reference(self, use_metropolis: bool) -> None:
    target_centered = jnp.zeros((N, 3))
    model = _ToyCenteredModel(target_centered, k=2.0)
    t = jnp.array(0.1)
    dt = 1e-3
    n_steps = 40
    coords0 = jax.random.normal(jax.random.PRNGKey(5), (N, 3)) * 0.3
    key0 = jax.random.PRNGKey(6)

    while_loop_result = run_langevin_equilibration(
      model, coords0, AATYPE, t, MASK, n_steps, dt, key0, use_metropolis=use_metropolis
    )
    scan_result = self._scan_reference(model, coords0, t, dt, n_steps, key0, use_metropolis=use_metropolis)

    assert jnp.allclose(while_loop_result, scan_result, atol=1e-6, rtol=1e-6)


TOKEN_S = 16
TOKEN_Z = 8


def _make_small_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=2,
    transformer_heads=2,
    key=key,
  )


class TestJitVmapCompatibility:
  """(5) jit/vmap compatibility for langevin_step and run_langevin_equilibration, real small model."""

  def test_langevin_step_is_jit_compatible(self) -> None:
    model = _make_small_model(jax.random.PRNGKey(10))
    coords = jax.random.normal(jax.random.PRNGKey(11), (N, 3)) * 0.1
    t = jnp.array(0.2)
    key = jax.random.PRNGKey(12)

    eager = langevin_step(model, coords, AATYPE, t, 1e-3, MASK, key)

    jitted_fn = eqx.filter_jit(
      lambda m, c, a, tt, mm, kk: langevin_step(m, c, a, tt, 1e-3, mm, kk)
    )
    jitted = jitted_fn(model, coords, AATYPE, t, MASK, key)
    assert jnp.allclose(eager, jitted, atol=1e-5)

  def test_langevin_step_is_vmap_compatible_over_keys(self) -> None:
    model = _make_small_model(jax.random.PRNGKey(13))
    coords = jax.random.normal(jax.random.PRNGKey(14), (N, 3)) * 0.1
    t = jnp.array(0.2)
    keys = jax.random.split(jax.random.PRNGKey(15), 4)

    vmapped = eqx.filter_vmap(
      lambda kk: langevin_step(model, coords, AATYPE, t, 1e-3, MASK, kk),
    )(keys)
    assert vmapped.shape == (4, N, 3)
    assert jnp.all(jnp.isfinite(vmapped))

    # Cross-check against sequential eager calls per key.
    sequential = jnp.stack([langevin_step(model, coords, AATYPE, t, 1e-3, MASK, kk) for kk in keys])
    assert jnp.allclose(vmapped, sequential, atol=1e-5)

  def test_run_langevin_equilibration_is_jit_compatible(self) -> None:
    model = _make_small_model(jax.random.PRNGKey(16))
    coords = jax.random.normal(jax.random.PRNGKey(17), (N, 3)) * 0.1
    t = jnp.array(0.2)
    key = jax.random.PRNGKey(18)
    n_steps = 3
    dt = 1e-3

    eager = run_langevin_equilibration(model, coords, AATYPE, t, MASK, n_steps, dt, key)

    jitted_fn = eqx.filter_jit(
      lambda m, c, a, tt, mm, kk: run_langevin_equilibration(m, c, a, tt, mm, n_steps, dt, kk)
    )
    jitted = jitted_fn(model, coords, AATYPE, t, MASK, key)
    assert jnp.allclose(eager, jitted, atol=1e-5)

  def test_run_langevin_equilibration_is_vmap_compatible_over_keys(self) -> None:
    model = _make_small_model(jax.random.PRNGKey(19))
    coords = jax.random.normal(jax.random.PRNGKey(20), (N, 3)) * 0.1
    t = jnp.array(0.2)
    keys = jax.random.split(jax.random.PRNGKey(21), 3)
    n_steps = 3
    dt = 1e-3

    vmapped = eqx.filter_vmap(
      lambda kk: run_langevin_equilibration(model, coords, AATYPE, t, MASK, n_steps, dt, kk),
    )(keys)
    assert vmapped.shape == (3, N, 3)
    assert jnp.all(jnp.isfinite(vmapped))

  def test_run_langevin_equilibration_with_metropolis_is_jit_compatible(self) -> None:
    model = _make_small_model(jax.random.PRNGKey(22))
    coords = jax.random.normal(jax.random.PRNGKey(23), (N, 3)) * 0.1
    t = jnp.array(0.2)
    key = jax.random.PRNGKey(24)
    n_steps = 3
    dt = 1e-3

    eager = run_langevin_equilibration(model, coords, AATYPE, t, MASK, n_steps, dt, key, use_metropolis=True)

    jitted_fn = eqx.filter_jit(
      lambda m, c, a, tt, mm, kk: run_langevin_equilibration(
        m, c, a, tt, mm, n_steps, dt, kk, use_metropolis=True
      )
    )
    jitted = jitted_fn(model, coords, AATYPE, t, MASK, key)
    assert jnp.allclose(eager, jitted, atol=1e-5)
