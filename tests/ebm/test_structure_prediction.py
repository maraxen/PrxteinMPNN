"""Tests for ``aminx.ebm.structure_prediction`` (backlog node **E10**).

Matches the rigor established by ``tests/ebm/test_langevin.py`` /
``tests/ebm/test_langevin_schedule.py``: non-vacuous statistical checks on
:func:`resample_ensemble` in isolation (per ``~/.claude/rules/BATHOS.md``
"verify your measurement pipeline before trusting any research conclusion" --
these are ground-truth checks on a synthetic identity-tagged batch, not just
shape/smoke tests), an end-to-end toy-model run confirming
:func:`run_structure_prediction` really executes ``num_rounds`` rounds and
that later rounds are no worse (in expectation) than round 0, and a jit/vmap
spot check against a real small ``ProteinEBMModel``. Scope matches the
dispatch: same model set across all rounds, plain quantile-threshold
resampling only, no AF2Rank rescoring -- see ``structure_prediction.py``'s
own module docstring for the full scope-limit disclosure.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.ebm import structure_prediction as sp
from aminx.ebm.langevin_schedule import run_annealing_schedule
from aminx.ebm.model import ProteinEBMModel
from aminx.ebm.structure_prediction import resample_ensemble, run_structure_prediction

N = 5
AATYPE = jnp.zeros((N,), dtype=jnp.int32)
MASK = jnp.ones((N,), dtype=bool)


class _ToyCenteredModel:
  """Toy quadratic energy, translation-invariant by construction.

  Identical construction to ``test_langevin.py``'s ``_ToyCenteredModel`` (see
  that file's docstring for why translation-invariance matters here --
  ``langevin_step``'s per-step ``center_random_augmentation`` re-centers
  *and* translates every step). Duplicated locally rather than imported
  across test files, matching this test suite's existing convention (see
  ``test_langevin_schedule.py``'s own locally-redefined
  ``_make_small_model``).
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


class TestResampleEnsemble:
  """(1) ``resample_ensemble`` in isolation: synthetic identity-tagged batch, no models involved.

  Every candidate's coordinates are set to a constant "identity" value
  (broadcast across all ``(N, 3)`` elements), and energies are assigned so
  identity == energy (lower identity == lower energy == "better"). Because
  :func:`resample_ensemble` renoises its output (a real, documented step --
  see that function's docstring), a *tiny* ``resample_noise_time`` is used
  throughout so the injected Gaussian noise stays far smaller than the
  spacing between adjacent identity values (``1.0``), keeping the recovered
  ``mean(resampled, axis=(1, 2))`` a faithful proxy for "which identity was
  resampled" -- this is verified, not assumed (see the shape/finiteness
  asserts below).
  """

  @staticmethod
  def _identity_coords_batch(n_candidates: int) -> jax.Array:
    idx = jnp.arange(n_candidates, dtype=jnp.float32)
    return jnp.broadcast_to(idx[:, None, None], (n_candidates, N, 3))

  def test_quantile_filter_excludes_high_energy_majority(self) -> None:
    """25th-percentile quantile filter must keep only the best ~10/40 candidates."""
    n_candidates = 40
    coords_batch = self._identity_coords_batch(n_candidates)
    energies_batch = jnp.arange(n_candidates, dtype=jnp.float32)
    thresh = float(jnp.quantile(energies_batch, 0.25))

    resampled = resample_ensemble(
      coords_batch,
      energies_batch,
      jax.random.PRNGKey(100),
      batch_size=3000,
      quantile_thresh=0.25,
      energy_scaling=-1.0,
      resample_noise_time=1e-3,
    )
    assert resampled.shape == (3000, N, 3)
    assert jnp.all(jnp.isfinite(resampled))

    identity_hat = jnp.mean(resampled, axis=(1, 2))
    max_identity = float(jnp.max(identity_hat))
    min_identity = float(jnp.min(identity_hat))
    print(f"quantile_thresh_value={thresh:.3f} min_identity_hat={min_identity:.3f} max_identity_hat={max_identity:.3f}")

    # Non-vacuous: the high-energy majority (identities 10..39, i.e. the
    # 30/40 candidates >= the 25th-percentile threshold) never appears in
    # the resampled batch -- not even once across 3000 draws. A generous
    # +2.0 margin absorbs the tiny renoise perturbation without weakening
    # the claim (the excluded majority starts at identity ~10, far above
    # threshold+2).
    assert max_identity < thresh + 2.0
    # The best candidate (identity 0) is reachable.
    assert min_identity < 1.0

    # Cross-check: with uniform (energy_scaling=-1) weighting over the kept
    # ~10 candidates {0..9}, the theoretical mean is (10-1)/2 = 4.5. Batch
    # size 3000 over a discrete-uniform-ish categorical with this small a
    # support gives a tight statistical bound (see analogous std-derived
    # tolerances in test_langevin.py's MH detailed-balance test).
    expected_uniform_mean = 4.5
    empirical_mean = float(jnp.mean(identity_hat))
    print(f"empirical_mean={empirical_mean:.3f} expected_uniform_mean={expected_uniform_mean:.3f}")
    assert abs(empirical_mean - expected_uniform_mean) < 0.75

  def test_boltzmann_weights_biased_toward_lower_energy(self) -> None:
    """Energy-based (Boltzmann) weighting must pull the resampled mean well below the uniform baseline."""
    n_candidates = 20
    coords_batch = self._identity_coords_batch(n_candidates)
    energies_batch = jnp.linspace(0.0, 100.0, n_candidates)
    batch_size = 6000
    # quantile_thresh=0.999 keeps (nearly) the whole pool -- strict `<`
    # excludes only entries at/above the near-max threshold value.
    quantile_thresh = 0.999
    energy_scaling = 10.0

    resampled_uniform = resample_ensemble(
      coords_batch,
      energies_batch,
      jax.random.PRNGKey(200),
      batch_size=batch_size,
      quantile_thresh=quantile_thresh,
      energy_scaling=-1.0,
      resample_noise_time=1e-3,
    )
    resampled_boltzmann = resample_ensemble(
      coords_batch,
      energies_batch,
      jax.random.PRNGKey(201),
      batch_size=batch_size,
      quantile_thresh=quantile_thresh,
      energy_scaling=energy_scaling,
      resample_noise_time=1e-3,
    )

    mean_uniform = float(jnp.mean(jnp.mean(resampled_uniform, axis=(1, 2))))
    mean_boltzmann = float(jnp.mean(jnp.mean(resampled_boltzmann, axis=(1, 2))))

    # Independent (host-side, plain numpy/jnp -- not calling into the
    # module's private helper) recomputation of the filtered pool + the
    # theoretical weighted/unweighted means and the weighted-mean estimator's
    # standard error, for a real statistical-consistency bound rather than a
    # magic-number tolerance.
    identities = jnp.arange(n_candidates, dtype=jnp.float32)
    keep_mask = energies_batch < jnp.quantile(energies_batch, quantile_thresh)
    filtered_identities = identities[keep_mask]
    filtered_energies = energies_batch[keep_mask]

    expected_uniform_mean = float(jnp.mean(filtered_identities))
    weights = jax.nn.softmax(-filtered_energies / energy_scaling)
    expected_boltzmann_mean = float(jnp.sum(weights * filtered_identities))
    boltzmann_var = float(jnp.sum(weights * (filtered_identities - expected_boltzmann_mean) ** 2))
    boltzmann_std_err = (boltzmann_var / batch_size) ** 0.5

    print(
      f"mean_uniform={mean_uniform:.3f} (expected {expected_uniform_mean:.3f}); "
      f"mean_boltzmann={mean_boltzmann:.3f} (expected {expected_boltzmann_mean:.3f}, "
      f"std_err={boltzmann_std_err:.3f})"
    )

    # (a) Non-vacuous: Boltzmann weighting is measurably biased toward lower
    # energy/identity relative to the uniform baseline over the *same*
    # filtered pool.
    assert mean_boltzmann < mean_uniform - 1.0
    # (b) Statistical consistency vs. the independently-recomputed
    # theoretical Boltzmann mean, within a 6-std_err bound.
    assert abs(mean_boltzmann - expected_boltzmann_mean) < max(6 * boltzmann_std_err, 0.2)

  def test_energy_scaling_negative_one_is_really_uniform_not_accidentally_boltzmann(self) -> None:
    """``energy_scaling=-1`` must give (near-)equal selection frequency to every filtered candidate.

    Uses a *widely spread* energy range specifically so that any accidental
    Boltzmann-style weighting (a temperature-convention-inversion class bug,
    per ``~/.claude/rules/BATHOS.md``) would show up as a clearly non-uniform
    empirical distribution -- this is the measurement-pipeline sanity check
    the bathos rule calls for.
    """
    n_candidates = 20
    coords_batch = self._identity_coords_batch(n_candidates)
    energies_batch = jnp.linspace(0.0, 500.0, n_candidates)  # wide spread
    batch_size = 6000
    quantile_thresh = 0.999  # keep (nearly) all n_candidates

    resampled = resample_ensemble(
      coords_batch,
      energies_batch,
      jax.random.PRNGKey(300),
      batch_size=batch_size,
      quantile_thresh=quantile_thresh,
      energy_scaling=-1.0,
      resample_noise_time=1e-3,
    )
    identity_hat = jnp.mean(resampled, axis=(1, 2))
    recovered_idx = jnp.round(identity_hat).astype(jnp.int32)

    keep_mask = energies_batch < jnp.quantile(energies_batch, quantile_thresh)
    n_kept = int(jnp.sum(keep_mask))
    counts = jnp.bincount(recovered_idx, length=n_candidates)

    expected_count = batch_size / n_kept
    # Multinomial per-bin std: sqrt(n * p * (1-p)), p = 1/n_kept.
    p = 1.0 / n_kept
    binomial_std = (batch_size * p * (1 - p)) ** 0.5
    tolerance = 6 * binomial_std

    print(f"n_kept={n_kept} expected_count={expected_count:.1f} binomial_std={binomial_std:.2f} tolerance={tolerance:.2f}")
    for idx in range(n_candidates):
      if bool(keep_mask[idx]):
        count = float(counts[idx])
        assert abs(count - expected_count) < tolerance, (
          f"identity {idx}: count={count} expected={expected_count:.1f} tolerance={tolerance:.2f}"
        )
      else:
        # The single excluded (highest-energy) candidate must never appear.
        assert int(counts[idx]) == 0


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


class TestRunStructurePredictionEndToEnd:
  """(2) End-to-end toy-model run: real rounds executed, non-vacuous refinement signal."""

  def test_runs_exactly_num_rounds_rounds_and_returns_expected_shapes(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch-count ``resample_ensemble`` calls: must fire exactly ``num_rounds - 1`` times."""
    call_count = 0
    real_resample_ensemble = sp.resample_ensemble

    def _counting_resample_ensemble(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
      nonlocal call_count
      call_count += 1
      return real_resample_ensemble(*args, **kwargs)

    monkeypatch.setattr(sp, "resample_ensemble", _counting_resample_ensemble)

    target_centered = jnp.zeros((N, 3))
    model = _ToyCenteredModel(target_centered, k=5.0)
    initial_coords = jax.random.normal(jax.random.PRNGKey(400), (N, 3)) * 0.3
    noise_schedule = jnp.linspace(0.001, 0.05, 4)[::-1]
    batch_size = 6
    num_rounds = 3

    final_coords, final_energies = sp.run_structure_prediction(
      (model,),
      (),
      initial_coords,
      AATYPE,
      MASK,
      noise_schedule,
      n_steps_per_level=5,
      dt=1e-3,
      key=jax.random.PRNGKey(401),
      num_rounds=num_rounds,
      batch_size=batch_size,
      effective_temp_scaling=3.0,
    )

    assert call_count == num_rounds - 1
    assert final_coords.shape == (batch_size, N, 3)
    assert final_energies.shape == (batch_size,)
    assert jnp.all(jnp.isfinite(final_coords))
    assert jnp.all(jnp.isfinite(final_energies))

  def test_later_round_mean_energy_is_no_worse_than_round_zero(self) -> None:
    """Deterministic A/B: same base ``key`` -> round 0 is bit-identical between the two calls.

    Calling with ``num_rounds=1`` yields exactly round 0's ensemble (since
    the key-splitting sequence up to and including round 0 does not depend
    on ``num_rounds`` -- see ``run_structure_prediction``'s docstring's PRNG
    discipline). Calling again with the *same* ``key`` but ``num_rounds=3``
    reproduces that identical round-0 ensemble internally and then continues
    refining it for two more rounds. This makes "later rounds no worse than
    round 0" a real, reproducible refinement claim, not a coincidence of
    independent random seeds.
    """
    target_centered = jnp.zeros((N, 3))
    model = _ToyCenteredModel(target_centered, k=5.0)
    initial_coords = jax.random.normal(jax.random.PRNGKey(500), (N, 3)) * 0.5
    noise_schedule = jnp.linspace(0.001, 0.05, 6)[::-1]
    batch_size = 24
    base_key = jax.random.PRNGKey(501)

    _round0_coords, round0_energies = run_structure_prediction(
      (model,),
      (),
      initial_coords,
      AATYPE,
      MASK,
      noise_schedule,
      n_steps_per_level=20,
      dt=1e-3,
      key=base_key,
      num_rounds=1,
      batch_size=batch_size,
      effective_temp_scaling=3.0,
      resample_noise_time=0.03,
      quantile_thresh=0.25,
      energy_scaling=5.0,
    )
    _final_coords, final_energies = run_structure_prediction(
      (model,),
      (),
      initial_coords,
      AATYPE,
      MASK,
      noise_schedule,
      n_steps_per_level=20,
      dt=1e-3,
      key=base_key,
      num_rounds=3,
      batch_size=batch_size,
      effective_temp_scaling=3.0,
      resample_noise_time=0.03,
      quantile_thresh=0.25,
      energy_scaling=5.0,
    )

    mean_round0 = float(jnp.mean(round0_energies))
    mean_final = float(jnp.mean(final_energies))
    print(f"mean_round0_energy={mean_round0:.4f} mean_final_round_energy={mean_final:.4f}")

    assert jnp.all(jnp.isfinite(round0_energies))
    assert jnp.all(jnp.isfinite(final_energies))
    # Non-vacuous refinement signal: 2 additional rounds of (quantile-filter
    # -> Boltzmann-resample -> renoise -> re-anneal) should not leave the
    # ensemble worse off on average than round 0 alone. A 10% relative slack
    # (chosen empirically -- this is a stochastic, small-batch toy run, not a
    # noiseless optimizer) absorbs sampling noise while still ruling out a
    # "resampling made things worse" regression.
    assert mean_final < mean_round0 * 1.1 + 1e-6


class TestJitVmapCompatibility:
  """(3) jit/vmap spot check: the per-round ``vmap(run_annealing_schedule)`` call, real small model.

  **Why ``num_rounds=1`` here, not 2+.** :func:`resample_ensemble`'s host-side
  numpy quantile filter (module docstring: data-dependent-shape, "not
  something ``jax.jit`` could trace even if we wanted it to") means
  ``run_structure_prediction`` as a whole is **not** end-to-end
  ``jax.jit``-compatible whenever it actually calls
  :func:`resample_ensemble` (i.e. whenever ``num_rounds >= 2``) -- verified
  empirically: wrapping a ``num_rounds=2`` call in ``eqx.filter_jit`` raises
  ``jax.errors.TracerArrayConversionError`` inside ``resample_ensemble``'s
  ``np.asarray(jax.device_get(energies_batch))`` line, exactly where a traced
  (jit-abstract) ``energies_batch`` cannot be concretized. This is expected,
  by-design behavior (the design spec explicitly frames the resampling step
  as a host-side ``Sink``/``Tap``, outside jit), not a bug to fix. This test
  therefore spot-checks jit/vmap-compatibility of the part of
  ``run_structure_prediction`` the dispatch actually asks for -- the per-round
  ``jax.vmap(run_annealing_schedule)`` call plus the round-0 initial-noising
  and terminal-energy-scoring steps -- via ``num_rounds=1``, which never
  reaches :func:`resample_ensemble`.
  """

  def test_run_structure_prediction_single_round_is_jit_compatible_with_real_small_model(self) -> None:
    model_lo = _make_small_model(jax.random.PRNGKey(600))
    model_hi = _make_small_model(jax.random.PRNGKey(601))
    threshold = 0.1

    initial_coords = jax.random.normal(jax.random.PRNGKey(602), (N, 3)) * 0.1
    noise_schedule = jnp.array([0.5, 0.2, 0.05])
    key = jax.random.PRNGKey(603)
    batch_size = 3

    def run(coords: jax.Array, kk: jax.Array) -> tuple[jax.Array, jax.Array]:
      return run_structure_prediction(
        (model_lo, model_hi),
        (threshold,),
        coords,
        AATYPE,
        MASK,
        noise_schedule,
        n_steps_per_level=2,
        dt=1e-3,
        key=kk,
        num_rounds=1,
        batch_size=batch_size,
      )

    eager_coords, eager_energies = run(initial_coords, key)

    jitted_coords, jitted_energies = eqx.filter_jit(run)(initial_coords, key)

    assert eager_coords.shape == (batch_size, N, 3)
    assert eager_energies.shape == (batch_size,)
    assert jnp.allclose(eager_coords, jitted_coords, atol=1e-4)
    assert jnp.allclose(eager_energies, jitted_energies, atol=1e-4)
    assert jnp.all(jnp.isfinite(jitted_coords))

  def test_per_round_vmap_run_annealing_schedule_is_vmap_compatible_over_keys(self) -> None:
    """Direct vmap check on the exact per-trajectory call ``run_structure_prediction`` makes each round."""
    model_lo = _make_small_model(jax.random.PRNGKey(610))
    model_hi = _make_small_model(jax.random.PRNGKey(611))
    threshold = 0.1

    coords = jax.random.normal(jax.random.PRNGKey(612), (N, 3)) * 0.1
    noise_schedule = jnp.array([0.5, 0.2, 0.05])
    keys = jax.random.split(jax.random.PRNGKey(613), 4)

    vmapped = eqx.filter_vmap(
      lambda kk: run_annealing_schedule(
        (model_lo, model_hi),
        (threshold,),
        coords,
        AATYPE,
        MASK,
        noise_schedule,
        2,
        1e-3,
        kk,
      ),
    )(keys)

    assert vmapped.shape == (4, N, 3)
    assert jnp.all(jnp.isfinite(vmapped))
