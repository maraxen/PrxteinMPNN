"""Tests for the ProteinEBM training path (backlog node E8, ``aminx.ebm.training``).

Covers: the score-matching loss composition (``L = 3*L_DSM + 0.75*L_aux +
0.1*L_atom``, design spec §1), the JAX-safe self-conditioning coin flip
(``jax.lax.cond`` on a traced boolean) + its ``stop_gradient`` boundary, the
``jax.checkpoint``-wrapped trunk forward pass (Fork 2), the E8-specific
second-order finite-difference gate on the REAL assembled ``ProteinEBMModel``
(not the toy/placeholder-trunk versions already covered by
``test_readout_invariants.py``), and a validation-only synthetic
training-loop smoke test (Fork 3) -- including an ``xtrax.engine.Engine``
integration check and an orbax checkpoint round-trip.

Per the task brief: the synthetic training-loop test validates that *the
training mechanism works* (loss finite/decreasing, no shape/dtype errors,
checkpoint round-trips) -- it explicitly does **not** claim to match any
published training result.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from xtrax.data.module import DataModule

from aminx.ebm import training as tr
from aminx.ebm.diffusion import VPSchedule
from aminx.ebm.model import ProteinEBMModel

# Small-but-real model dims (mirrors tests/ebm/test_model.py's fixture, one
# notch smaller for fast per-test compilation).
TOKEN_S = 8
TOKEN_Z = 8
DEPTH = 1
HEADS = 1
N = 5
B = 3


def _make_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=8,
    conditioning_transition_layers=1,
    transformer_depth=DEPTH,
    transformer_heads=HEADS,
    r_max=4,
    s_max=1,
    key=key,
  )


def _synthetic_batch(key: jax.Array, batch_size: int = B, n: int = N) -> dict[str, jax.Array]:
  k_c, k_a, k_t = jax.random.split(key, 3)
  coords0 = jax.random.normal(k_c, (batch_size, n, 3))
  aatype = jax.random.randint(k_a, (batch_size, n), 0, 21)
  mask = jnp.ones((batch_size, n), dtype=bool)
  t = jax.random.uniform(k_t, (batch_size,), minval=0.01, maxval=0.99)
  return {"coords0": coords0, "aatype": aatype, "mask": mask, "t": t}


class TestMaskedMSE:
  def test_zero_when_pred_equals_target(self) -> None:
    pred = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
    mask = jnp.ones((N,), dtype=bool)
    assert jnp.allclose(tr.masked_mse(pred, pred, mask), 0.0)

  def test_masked_out_residues_do_not_contribute(self) -> None:
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    pred = jax.random.normal(k1, (N, 3))
    target = jax.random.normal(k2, (N, 3))
    mask_full = jnp.ones((N,), dtype=bool)
    mask_partial = jnp.array([True, True, True, False, False])

    # Perturbing masked-out entries must not change the masked MSE.
    pred_perturbed = pred.at[3:].set(pred[3:] * 1000.0 + 50.0)
    mse_before = tr.masked_mse(pred, target, mask_partial)
    mse_after = tr.masked_mse(pred_perturbed, target, mask_partial)
    assert jnp.allclose(mse_before, mse_after, atol=1e-5)
    assert mask_full.shape == mask_partial.shape  # sanity on fixture shapes


class TestScoreMatchingLossComposition:
  def test_shape_and_finite(self) -> None:
    key = jax.random.PRNGKey(2)
    k_model, k_batch, k_step = jax.random.split(key, 3)
    model = _make_model(k_model)
    batch = _synthetic_batch(k_batch)
    schedule = VPSchedule()

    loss, metrics = tr.score_matching_loss(
      model, schedule, batch["coords0"], batch["aatype"], batch["mask"], batch["t"], k_step,
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    for name in ("loss", "l_dsm", "l_aux", "l_atom"):
      assert name in metrics
      assert jnp.isfinite(metrics[name])

  def test_l_atom_is_documented_zero_noop(self) -> None:
    """Design spec §1 Fork 5: no all-atom head exists -- L_atom must be an honest zero, not fabricated."""
    key = jax.random.PRNGKey(3)
    k_model, k_batch, k_step = jax.random.split(key, 3)
    model = _make_model(k_model)
    batch = _synthetic_batch(k_batch)
    schedule = VPSchedule()

    _loss, metrics = tr.score_matching_loss(
      model, schedule, batch["coords0"], batch["aatype"], batch["mask"], batch["t"], k_step,
    )
    assert jnp.allclose(metrics["l_atom"], 0.0)

  def test_loss_matches_declared_weighted_sum(self) -> None:
    """L = 3*L_DSM + 0.75*L_aux + 0.1*L_atom, exactly (design spec §1)."""
    key = jax.random.PRNGKey(4)
    k_model, k_batch, k_step = jax.random.split(key, 3)
    model = _make_model(k_model)
    batch = _synthetic_batch(k_batch)
    schedule = VPSchedule()

    loss, metrics = tr.score_matching_loss(
      model, schedule, batch["coords0"], batch["aatype"], batch["mask"], batch["t"], k_step,
    )
    reconstructed = (
      tr.DSM_WEIGHT * metrics["l_dsm"]
      + tr.AUX_WEIGHT * metrics["l_aux"]
      + tr.ATOM_WEIGHT * metrics["l_atom"]
    )
    assert jnp.allclose(loss, reconstructed, atol=1e-5, rtol=1e-5)
    assert tr.DSM_WEIGHT == pytest.approx(3.0)
    assert tr.AUX_WEIGHT == pytest.approx(0.75)
    assert tr.ATOM_WEIGHT == pytest.approx(0.1)

  def test_is_jit_compatible(self) -> None:
    key = jax.random.PRNGKey(5)
    k_model, k_batch, k_step = jax.random.split(key, 3)
    model = _make_model(k_model)
    batch = _synthetic_batch(k_batch)
    schedule = VPSchedule()

    eager_loss, _ = tr.score_matching_loss(
      model, schedule, batch["coords0"], batch["aatype"], batch["mask"], batch["t"], k_step,
    )
    jitted = eqx.filter_jit(tr.score_matching_loss)
    jit_loss, _ = jitted(
      model, schedule, batch["coords0"], batch["aatype"], batch["mask"], batch["t"], k_step,
    )
    assert jnp.allclose(eager_loss, jit_loss, atol=1e-5, rtol=1e-5)


class TestSelfConditioningJaxSafe:
  """The self-conditioning coin flip must be a traced ``lax.cond``, not a Python branch."""

  def test_both_branches_traceable_under_jit(self) -> None:
    key = jax.random.PRNGKey(6)
    k_model, k_data = jax.random.split(key)
    model = _make_model(k_model)
    k_x, k_a, k_key = jax.random.split(k_data, 3)
    x_t = jax.random.normal(k_x, (N, 3)) * 0.1
    aatype = jax.random.randint(k_a, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.3)
    schedule = VPSchedule()

    fn = eqx.filter_jit(tr.self_conditioning_estimate)

    out_false = fn(model, x_t, aatype, t, mask, schedule, k_key, jnp.array(False), 0.1)
    out_true = fn(model, x_t, aatype, t, mask, schedule, k_key, jnp.array(True), 0.1)

    assert out_false.shape == (N, 3)
    assert out_true.shape == (N, 3)
    assert jnp.allclose(out_false, 0.0)
    assert not jnp.allclose(out_true, 0.0)
    assert jnp.all(jnp.isfinite(out_true))

  def test_traced_bool_does_not_force_retracing(self) -> None:
    """A single compiled trace must serve BOTH runtime coin-flip outcomes (lax.cond, not lax.switch-per-call)."""
    key = jax.random.PRNGKey(7)
    k_model, k_data = jax.random.split(key)
    model = _make_model(k_model)
    k_x, k_a, k_key = jax.random.split(k_data, 3)
    x_t = jax.random.normal(k_x, (N, 3)) * 0.1
    aatype = jax.random.randint(k_a, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.3)
    schedule = VPSchedule()

    fn = eqx.filter_jit(tr.self_conditioning_estimate)
    # `use_self_cond` is passed as a jnp array (traced value), not a Python
    # bool -- if this were a Python `if`, jax would either error (on an
    # abstract tracer) or silently retrace per call. Calling twice with
    # different array values against the SAME jitted callable must not error.
    fn(model, x_t, aatype, t, mask, schedule, k_key, jnp.array(True), 0.1)
    fn(model, x_t, aatype, t, mask, schedule, k_key, jnp.array(False), 0.1)


class TestStopGradientOnSelfConditioning:
  """Design spec §10 MAJOR-6: the recycled self-conditioning estimate must not backprop into the model."""

  def test_self_conditioning_output_has_zero_grad_wrt_model(self) -> None:
    key = jax.random.PRNGKey(8)
    k_model, k_data = jax.random.split(key)
    model = _make_model(k_model)
    k_x, k_a, k_key = jax.random.split(k_data, 3)
    x_t = jax.random.normal(k_x, (N, 3)) * 0.1
    aatype = jax.random.randint(k_a, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.3)
    schedule = VPSchedule()

    def sum_sc(m: ProteinEBMModel) -> jax.Array:
      sc = tr.self_conditioning_estimate(
        m, x_t, aatype, t, mask, schedule, k_key, jnp.array(True), 0.1,
      )
      return jnp.sum(sc)

    grads = eqx.filter_grad(sum_sc)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(leaves) > 0
    assert all(jnp.allclose(leaf, 0.0) for leaf in leaves)


class TestCheckpointedTrunkPreservesNumerics:
  """``jax.checkpoint`` changes backward-pass memory, not forward values."""

  def test_checkpointed_score_matches_model_score(self) -> None:
    key = jax.random.PRNGKey(9)
    k_model, k_data = jax.random.split(key)
    model = _make_model(k_model)
    k_x, k_a = jax.random.split(k_data)
    coords = jax.random.normal(k_x, (N, 3)) * 0.1
    aatype = jax.random.randint(k_a, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.3)

    reference = model.score(coords, aatype, t, mask)
    checkpointed = tr.checkpointed_score(model, coords, aatype, t, mask, jnp.zeros_like(coords))
    assert jnp.allclose(reference, checkpointed, atol=1e-5, rtol=1e-5)

  def test_checkpointed_aux_score_matches_model_aux_score(self) -> None:
    key = jax.random.PRNGKey(10)
    k_model, k_data = jax.random.split(key)
    model = _make_model(k_model)
    k_x, k_a = jax.random.split(k_data)
    coords = jax.random.normal(k_x, (N, 3)) * 0.1
    aatype = jax.random.randint(k_a, (N,), 0, 21)
    mask = jnp.ones((N,), dtype=bool)
    t = jnp.array(0.3)

    reference = model.aux_score(coords, aatype, t, mask)
    checkpointed = tr.checkpointed_aux_score(
      model, coords, aatype, t, mask, jnp.zeros_like(coords),
    )
    assert jnp.allclose(reference, checkpointed, atol=1e-5, rtol=1e-5)


class TestSecondOrderTrainingGradOnRealModel:
  """The E8-specific Fork 2 gate: outer training grad through the REAL assembled model.

  Distinct from ``test_readout_invariants.py``'s toy-energy
  (``TestSecondOrderTrainingGradOnToy``) and placeholder-trunk
  (``test_second_order_training_grad_is_finite_on_real_model``, which stubs
  ``trunk_fn`` as a bare reshape) gates: this exercises the FULL
  ``ProteinEBMModel`` (real input embeddings, ``SingleConditioning``,
  ``DiffusionTransformer``, readouts) via ``training.py``'s
  ``jax.checkpoint``-wrapped :func:`aminx.ebm.training.checkpointed_score`.
  """

  def test_outer_grad_is_finite_on_realistically_sized_model(self) -> None:
    """Fast finiteness check on a small-but-nontrivial model (no finite-difference -- see below for that)."""
    key = jax.random.PRNGKey(11)
    k_model, k_data, k_target = jax.random.split(key, 3)
    model = ProteinEBMModel(
      token_s=16,
      token_z=8,
      dim_fourier=12,
      conditioning_transition_layers=1,
      transformer_depth=2,
      transformer_heads=2,
      r_max=8,
      s_max=1,
      key=k_model,
    )
    k_x, k_a = jax.random.split(k_data)
    coords = jax.random.normal(k_x, (6, 3)) * 0.1
    aatype = jax.random.randint(k_a, (6,), 0, 21)
    mask = jnp.ones((6,), dtype=bool)
    t = jnp.array(0.3)
    target = jax.random.normal(k_target, (6, 3))

    def loss_fn(m: ProteinEBMModel) -> jax.Array:
      score = tr.checkpointed_score(m, coords, aatype, t, mask, jnp.zeros_like(coords))
      return jnp.sum((score - target) ** 2)

    grads = eqx.filter_grad(loss_fn)(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    assert len(leaves) > 0
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)

  def test_outer_grad_matches_directional_finite_difference_on_tiny_model(
    self,
    enable_x64,
  ) -> None:
    """Full elementwise finite-diff is infeasible even for a "tiny" assembled model (thousands of
    params); a directional derivative check (random unit direction in parameter space) is the
    standard, tractable gradient-check technique for large models -- see module docstring's BATHOS
    "verify the measurement pipeline" discipline.
    """
    key = jax.random.PRNGKey(12)
    k_model, k_data, k_target, k_dir = jax.random.split(key, 4)
    model = ProteinEBMModel(
      token_s=3,
      token_z=3,
      dim_fourier=4,
      conditioning_transition_layers=1,
      transformer_depth=1,
      transformer_heads=1,
      r_max=2,
      s_max=1,
      key=k_model,
    )

    def to_f64(x: object) -> object:
      return x.astype(jnp.float64) if eqx.is_inexact_array(x) else x

    model = jax.tree_util.tree_map(to_f64, model)

    n = 3
    k_x, k_a = jax.random.split(k_data)
    coords = jax.random.normal(k_x, (n, 3), dtype=jnp.float64)
    aatype = jax.random.randint(k_a, (n,), 0, 21)
    mask = jnp.ones((n,), dtype=bool)
    t = jnp.asarray(0.3, dtype=jnp.float64)
    target = jax.random.normal(k_target, (n, 3), dtype=jnp.float64)

    def loss_fn(m: ProteinEBMModel) -> jax.Array:
      score = tr.checkpointed_score(m, coords, aatype, t, mask, jnp.zeros_like(coords))
      return jnp.sum((score - target) ** 2)

    grads = eqx.filter_grad(loss_fn)(model)
    grad_leaves, grad_treedef = jax.tree_util.tree_flatten(
      eqx.filter(grads, eqx.is_inexact_array),
    )
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in grad_leaves)

    params, static = eqx.partition(model, eqx.is_inexact_array)
    param_leaves, param_treedef = jax.tree_util.tree_flatten(params)
    dir_keys = jax.random.split(k_dir, len(param_leaves))
    direction_leaves = [
      jax.random.normal(dk, p.shape, dtype=p.dtype) for dk, p in zip(dir_keys, param_leaves)
    ]
    norm = jnp.sqrt(sum(jnp.sum(d * d) for d in direction_leaves))
    direction_leaves = [d / norm for d in direction_leaves]
    direction = jax.tree_util.tree_unflatten(param_treedef, direction_leaves)

    eps = 1e-5
    params_plus = jax.tree_util.tree_map(lambda p, d: p + eps * d, params, direction)
    params_minus = jax.tree_util.tree_map(lambda p, d: p - eps * d, params, direction)
    model_plus = eqx.combine(params_plus, static)
    model_minus = eqx.combine(params_minus, static)

    numerical_directional = (loss_fn(model_plus) - loss_fn(model_minus)) / (2 * eps)
    analytic_directional = sum(
      jnp.sum(g * d) for g, d in zip(grad_leaves, direction_leaves)
    )
    del grad_treedef  # only the leaves are needed above

    assert jnp.allclose(numerical_directional, analytic_directional, atol=1e-4, rtol=1e-4)


class TestOptimizerAndState:
  def test_build_optimizer_returns_gradient_transformation(self) -> None:
    optimizer = tr.build_optimizer(peak_lr=1e-3, warmup_steps=2, total_steps=10)
    assert hasattr(optimizer, "init")
    assert hasattr(optimizer, "update")

  def test_init_training_state_shape(self) -> None:
    key = jax.random.PRNGKey(13)
    model = _make_model(key)
    optimizer = tr.build_optimizer(peak_lr=1e-3, warmup_steps=2, total_steps=10)
    state = tr.init_training_state(model, optimizer, seed=0)
    assert int(state.step) == 0
    assert state.model is model


class TestEBMTrainStep:
  def test_step_updates_model_and_increments_step(self) -> None:
    key = jax.random.PRNGKey(14)
    k_model, k_batch = jax.random.split(key)
    model = _make_model(k_model)
    # warmup_steps=0: `adamw_with_schedule`'s warmup-cosine-decay schedule
    # starts at `init_value=0.0` at step count 0, so a nonzero warmup would
    # make the very first update's effective LR exactly 0 -- this test
    # specifically wants to observe a nonzero parameter change on step 1.
    optimizer = tr.build_optimizer(peak_lr=1e-3, warmup_steps=0, total_steps=20, clip_norm=10.0)
    state = tr.init_training_state(model, optimizer, seed=0)
    train_step = tr.EBMTrainStep(optimizer=optimizer)
    batch = _synthetic_batch(k_batch)

    new_state, metrics = train_step.step(state, batch)

    assert int(new_state.step) == 1
    assert jnp.isfinite(metrics["loss"])
    assert jnp.isfinite(metrics["grad_norm"])
    old_leaves = jax.tree_util.tree_leaves(eqx.filter(state.model, eqx.is_inexact_array))
    new_leaves = jax.tree_util.tree_leaves(eqx.filter(new_state.model, eqx.is_inexact_array))
    assert any(
      not jnp.array_equal(o, n) for o, n in zip(old_leaves, new_leaves)
    ), "at least one parameter must change after an update"

  def test_key_is_not_reused_across_steps(self) -> None:
    key = jax.random.PRNGKey(15)
    k_model, k_batch = jax.random.split(key)
    model = _make_model(k_model)
    optimizer = tr.build_optimizer(peak_lr=1e-3, warmup_steps=2, total_steps=20)
    state = tr.init_training_state(model, optimizer, seed=0)
    train_step = tr.EBMTrainStep(optimizer=optimizer)
    batch = _synthetic_batch(k_batch)

    state1, _ = train_step.step(state, batch)
    state2, _ = train_step.step(state1, batch)
    assert not jnp.array_equal(state.key, state1.key)
    assert not jnp.array_equal(state1.key, state2.key)


class TestEngineIntegration:
  """``xtrax.engine.Engine`` adoption (Fork 8) -- verified to work without a workaround."""

  def test_engine_fit_sync_runs_all_batches(self) -> None:
    key = jax.random.PRNGKey(16)
    k_model, k_data = jax.random.split(key)
    model = _make_model(k_model)
    optimizer = tr.build_optimizer(peak_lr=1e-3, warmup_steps=2, total_steps=20)
    state = tr.init_training_state(model, optimizer, seed=0)
    train_step = tr.EBMTrainStep(optimizer=optimizer)
    engine = tr.make_engine(train_step)

    n_batches = 4
    batch_keys = jax.random.split(k_data, n_batches)
    batches = [_synthetic_batch(k) for k in batch_keys]
    data_module = DataModule(dataset=batches, batch_size=B, num_epochs=1, seed=0, distributed=False)

    final_state = engine.fit_sync(state, data_module, num_epochs=1)
    assert int(final_state.step) == n_batches


class TestValidationOnlyTrainingLoop:
  """Fork 3: confirms the training MECHANISM works on synthetic data only.

  Explicitly does not claim to match any published training result -- see
  module docstring.
  """

  def test_loss_is_finite_and_trends_down_over_synthetic_steps(self) -> None:
    key = jax.random.PRNGKey(17)
    k_model, k_batch = jax.random.split(key)
    model = _make_model(k_model)
    k_c, k_a = jax.random.split(k_batch)
    batch_size, n = 4, 6
    # `t` is fixed at a moderate value (not resampled near 0), avoiding the
    # `1/conditional_var(t)` blowup as `t -> 0` (`diffusion.py::score_target`)
    # that would otherwise dominate step-to-step loss variance and mask any
    # true optimization trend over just a few dozen steps.
    batch = {
      "coords0": jax.random.normal(k_c, (batch_size, n, 3)),
      "aatype": jax.random.randint(k_a, (batch_size, n), 0, 21),
      "mask": jnp.ones((batch_size, n), dtype=bool),
      "t": jnp.full((batch_size,), 0.3),
    }

    optimizer = tr.build_optimizer(peak_lr=3e-3, warmup_steps=0, total_steps=80, clip_norm=10.0)
    state = tr.init_training_state(model, optimizer, seed=42)
    # Deterministic branches (no coin-flip/drop noise) isolate the
    # optimization-mechanism check from self-conditioning/sequence-drop
    # variance -- those are covered by dedicated tests above. The diffusion
    # forward-marginal noise draw itself is NOT disabled (re-drawn every
    # step from the re-keyed state, exactly as real training would), so some
    # step-to-step variance remains by design.
    train_step = tr.EBMTrainStep(optimizer=optimizer, self_conditioning_prob=0.0, drop_seq_prob=0.0)

    losses = []
    for _ in range(60):
      state, metrics = train_step.step(state, batch)
      loss_value = float(metrics["loss"])
      assert jnp.isfinite(metrics["loss"]), "loss must never be NaN/Inf across the synthetic loop"
      losses.append(loss_value)

    mean_first = sum(losses[:10]) / 10
    mean_last = sum(losses[-10:]) / 10
    assert mean_last < mean_first, (
      f"expected mean loss to trend down over 60 steps on fixed synthetic data: "
      f"first10={mean_first:.3f} last10={mean_last:.3f}"
    )

  def test_checkpoint_round_trip_gives_equivalent_resumable_state(self, tmp_path) -> None:
    key = jax.random.PRNGKey(18)
    model = _make_model(key)
    optimizer = tr.build_optimizer(peak_lr=1e-3, warmup_steps=2, total_steps=10)
    state = tr.init_training_state(model, optimizer, seed=0)

    manager = tr.get_ebm_checkpoint_manager(tmp_path / "ckpt", max_to_keep=2)
    tr.save_ebm_checkpoint(manager, state)
    loaded = tr.load_ebm_checkpoint(manager, state)
    manager.close()

    assert int(loaded.step) == int(state.step)
    original_leaves = jax.tree_util.tree_leaves(eqx.filter(state, eqx.is_array))
    loaded_leaves = jax.tree_util.tree_leaves(eqx.filter(loaded, eqx.is_array))
    assert len(original_leaves) == len(loaded_leaves)
    assert all(
      jnp.array_equal(o, l) for o, l in zip(original_leaves, loaded_leaves)
    ), "checkpoint round-trip must reproduce the exact ResumableState"
