"""Training path for the ProteinEBM energy/score model (backlog node **E8**).

Ported from ``~/repos/ProteinEBM/protein_ebm/scripts/train.py``'s
``ProteinScoreMatchingTrainer.generic_step`` (the loss composition -- the
Lightning scaffolding itself is not runnable as shipped, per design spec §4,
and is not ported). See
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §2 (E8 row) and
``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §1 (loss
weights), §3.4 (training path), §10 (MAJOR-6: 2nd-order AD).

**Scope boundary.** This module imports only from ``aminx.ebm.contracts``,
``aminx.ebm.diffusion``, ``aminx.ebm.model``, and ``aminx.ebm.readout`` (all
untouched by this backlog node) plus ``xtrax``/``optax``/``orbax``. It does
**not** import from ``aminx.training`` (the existing, tested sequence-
diffusion trainer) -- per Fork 8 (EPIC §1), E8 adopts ``xtrax.engine.Engine``
for a brand-new trainer without touching or backfilling that path.

**Loss.** ``L = 3*L_DSM + 0.75*L_aux + 0.1*L_atom`` (design spec §1), all
masked MSE against the closed-form DSM score target
(:func:`aminx.ebm.diffusion.forward_marginal`):

* ``L_DSM``: conservative score ``model.score(...)`` (nested ``jax.grad``,
  i.e. the reverse-over-reverse AD pattern this node gates -- Fork 2).
* ``L_aux``: non-conservative ``model.aux_score(...)`` (no inner grad).
* ``L_atom``: **scoped gap, not fabricated (design spec §1 Fork 5).**
  ``ProteinEBMModel``/``EnergyReadout`` (E3/E3.6) implement **no** all-atom /
  sidechain prediction head -- the reference's ``r_atom`` output has no
  counterpart here. Inventing one now would be a large, under-specified new
  architectural addition (a fresh linear head + all-atom loss masking logic)
  well beyond this node's brief. So ``L_atom`` is a **documented zero
  no-op** (``l_atom = jnp.zeros_like(l_dsm)`` in ``_per_example_loss``): the
  literal loss formula keeps its three-term shape (so a future all-atom head
  only needs to replace this one term), but no term is silently invented.
  This is an honest scope reduction, not a hidden approximation.

**Self-conditioning + sequence-drop (design spec §3.4/§10, the "single
subtle correctness item").** The reference's per-batch Python coin-flip
(``random.random() > 0.5`` gating the *whole* self-conditioning branch,
``train.py``'s ``generic_step``) becomes :func:`jax.lax.cond` on a traced
boolean (:func:`self_conditioning_estimate`) -- only one branch executes at
runtime, unlike an always-both-branches ``jnp.where`` select, which would
pay for the (already-expensive, checkpointed) recycled forward pass every
step regardless of the coin flip. Sequence-drop is already a per-example
``torch.where`` in the reference (``drop_mask`` has shape ``(B,)``) and
needs no control-flow conversion -- ported here as a per-example
``jnp.where`` on an independent Bernoulli draw per example, exactly
mirroring the reference (see :func:`_per_example_loss`).

**Stop-gradient on the recycled self-conditioning estimate (design spec §10
MAJOR-6).** Verified against the reference: ``train.py``'s self-conditioning
branch wraps its recycled forward pass in ``with torch.no_grad():`` before
setting ``input_feats['selfcond_coords']`` -- confirming no gradient flows
back through the recycled estimate. :func:`self_conditioning_estimate`
applies :func:`jax.lax.stop_gradient` to the returned coordinates,
replicating that no-grad boundary exactly.

**Second-order gradient handling (Fork 2 gate).** The trunk's forward pass
is wrapped in :func:`jax.checkpoint` at every call site in this module
(:func:`checkpointed_trunk_fn`) to bound activation memory for the
reverse-over-reverse AD: the *outer* training gradient (w.r.t. model
parameters, via ``eqx.filter_value_and_grad`` in :class:`EBMTrainStep`)
differentiates through a loss that itself contains an *inner*
``jax.grad`` (the conservative score). This is done entirely from this
module -- ``aminx.ebm.model``/``readout`` are not modified; the composition
seam already exists (``ScoreReadout.__call__`` and
``ProteinEBMModel.trunk_features`` both take/are an externally-supplied
``trunk_fn``, per their own docstrings), so wrapping it in ``jax.checkpoint``
here is additive composition, not a change to those files.

**Efficiency note (honest, not hidden).** Because this module cannot modify
``readout.py``'s ``ScoreReadout`` to expose the trunk activations it computes
internally, :func:`checkpointed_score` and :func:`checkpointed_aux_score`
each run their own, independent trunk forward pass (two total per main
prediction, plus a third for the self-conditioning recycled estimate when
the coin flip fires) -- unlike the PyTorch reference, which shares one
forward graph between ``energy``/``aux_score``/``autograd.grad`` in a single
``compute_score`` call. This is a direct, documented consequence of the
"peer composition, don't touch existing files" design constraint (this
EPIC's central rule), not an oversight.

**Training harness (Fork 8).** ``xtrax.engine.Engine`` **does** work as
expected for this new trainer: :class:`EBMTrainStep` is a plain
``eqx.Module`` implementing ``xtrax.engine.engine.TrainStepLike``'s
duck-typed ``step(state, batch) -> (state, metrics)`` protocol (verified by
reading ``xtrax/engine/engine.py`` directly -- ``Engine.trainer`` accepts
``Trainer | SafetyTrainStep | TrainStepLike``, and ``TrainStepLike`` is a
``@runtime_checkable`` ``Protocol`` requiring only that one method). No
workaround was needed; the existing ``xtrax.training.trainer.Trainer`` class
was **not** reused because its ``step`` hardcodes
``predictions = model(batch["inputs"]); loss_fn(predictions, batch["targets"])``,
too rigid for a loss that needs its own PRNG key (self-conditioning coin
flip, sequence-drop, the diffusion forward-marginal draw) and multiple model
outputs (``score`` and ``aux_score``) per step -- so :class:`EBMTrainStep`
duck-types the same protocol with its own ``step`` body instead.

**Optimizer.** ``xtrax.training.optim.adamw_with_schedule`` (warmup +
cosine decay + AdamW + gradient clipping) -- the closest available xtrax
primitive to the reference's ``torch.optim.lr_scheduler.OneCycleLR``.
**Not** a literal one-cycle schedule (xtrax has no such helper): this is a
documented approximation of the reference's learning-rate *intent*
(warm up, then decay to ~0), not a byte-faithful port of OneCycle's
symmetric ramp-then-decay shape. Gradient clipping defaults to ``10.0``,
matching the reference's hardcoded ``gradient_clip_val=10.0``.

**Checkpointing.** ``xtrax.checkpoint`` (``get_checkpoint_manager``,
``save_checkpoint``, ``load_checkpoint``) is used directly -- it is xtrax's
own orbax wrapper (verified importable, near-identical to
``aminx.training.checkpoint``, which this node must not touch). Using
xtrax's copy avoids duplicating or depending on the existing tested
sequence-diffusion trainer's module.

**Validation-only retrain (Fork 3).** The training loop here is validated by
:mod:`tests.ebm.test_training` on synthetic random data only: it confirms
the training *mechanism* works (loss finite/non-NaN, no shape/dtype errors,
checkpoint round-trip), explicitly **not** that it reproduces any published
training result. There is no data pipeline in this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from xtrax.checkpoint import get_checkpoint_manager, load_checkpoint, save_checkpoint
from xtrax.engine import Engine
from xtrax.training.optim import adamw_with_schedule
from xtrax.training.types import ResumableState

from aminx.ebm.diffusion import VPSchedule
from aminx.ebm.readout import ScoreReadout

if TYPE_CHECKING:
  from collections.abc import Callable
  from pathlib import Path

  import orbax.checkpoint as ocp
  from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

  from aminx.ebm.contracts import AAType, Coords, DiffusionTime, ResidueMask, Score
  from aminx.ebm.model import ProteinEBMModel

  BatchCoords = Float[Array, "B N 3"]
  BatchAAType = Int[Array, "B N"]
  BatchResidueMask = Bool[Array, "B N"]
  BatchTime = Float[Array, "B"]

# Loss weights, verbatim from design spec §1 / EPIC backlog DAG §2 (E8 row):
# L = 3*L_DSM + 0.75*L_aux + 0.1*L_atom.
DSM_WEIGHT = 3.0
AUX_WEIGHT = 0.75
ATOM_WEIGHT = 0.1

# 21-way aatype vocabulary's mask token (contracts.py: "[0, 20], 21-way
# embedding incl. mask token 20").
MASK_AA_TOKEN = 20

# Reference defaults (train.py's generic_step / configure_optimizers).
DEFAULT_DROP_SEQ_PROB = 0.1
DEFAULT_SELF_CONDITIONING_PROB = 0.5
DEFAULT_GRADIENT_CLIP_NORM = 10.0

# Small numerical floor for the masked-mean reduction's denominator, matching
# the reference's `+ 1e-10` guard (train.py: `loss_mask.sum(dim=-1) + 1e-10`).
_MASK_SUM_EPS = 1e-10


def masked_mse(
  pred: Score,
  target: Score,
  mask: ResidueMask,
) -> Float[Array, ""]:
  """Masked mean squared error between two ``(N, 3)`` vector fields.

  ``sum_i mask_i * ||pred_i - target_i||**2 / (sum_i mask_i + eps)`` --
  matches the reference's ``trans_score_mse`` reduction (sum over the last
  two axes, normalized by ``loss_mask.sum(dim=-1) + 1e-10``).
  """
  sq_err = jnp.sum((pred - target) ** 2, axis=-1)
  masked = mask.astype(sq_err.dtype) * sq_err
  denom = jnp.sum(mask.astype(sq_err.dtype)) + _MASK_SUM_EPS
  return jnp.sum(masked) / denom


def checkpointed_trunk_fn(
  model: ProteinEBMModel,
  aatype: AAType,
  t: DiffusionTime,
  sc_coords: Coords,
) -> Callable[[Coords, ResidueMask], Float[Array, "N two_token_s"]]:
  """Build a ``jax.checkpoint``-wrapped closure over ``model.trunk_features``.

  This is the Fork-2 gate's memory-bounding mechanism: the outer training
  gradient (w.r.t. model params) treats the wrapped trunk forward pass as an
  opaque checkpoint boundary -- intermediate activations are not stored for
  the backward pass; they are recomputed from the (cheap) saved inputs
  instead. Composes with ``aminx.ebm.model.ProteinEBMModel.trunk_features``
  and ``aminx.ebm.readout.ScoreReadout``'s externally-supplied ``trunk_fn``
  seam without modifying either file (both already document this as the
  intended extension point).
  """

  def trunk_fn(coords: Coords, mask: ResidueMask) -> Float[Array, "N two_token_s"]:
    return model.trunk_features(coords, aatype, t, mask, sc_coords=sc_coords)

  return jax.checkpoint(trunk_fn)


def checkpointed_score(
  model: ProteinEBMModel,
  coords: Coords,
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  sc_coords: Coords,
) -> Score:
  """Conservative score ``-jax.grad(energy)(coords)`` through a checkpointed trunk.

  The reverse-over-reverse AD pattern this node gates: the OUTER training
  gradient (``eqx.filter_value_and_grad`` in :class:`EBMTrainStep`)
  differentiates through this function, which itself contains an INNER
  ``jax.grad`` (via :class:`aminx.ebm.readout.ScoreReadout`).
  """
  trunk_fn = checkpointed_trunk_fn(model, aatype, t, sc_coords)
  score_readout = ScoreReadout(model.energy_readout)
  return score_readout(coords, mask, trunk_fn)


def checkpointed_aux_score(
  model: ProteinEBMModel,
  coords: Coords,
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  sc_coords: Coords,
) -> Score:
  """Non-conservative aux score (no inner ``jax.grad``) through a checkpointed trunk."""
  trunk_fn = checkpointed_trunk_fn(model, aatype, t, sc_coords)
  trunk_out = trunk_fn(coords, mask)
  return model.aux_score_readout(trunk_out)


def self_conditioning_estimate(
  model: ProteinEBMModel,
  x_t: Coords,
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  schedule: VPSchedule,
  key: PRNGKeyArray,
  use_self_cond: Bool[Array, ""],
  drop_seq_prob: float,
) -> Coords:
  """JAX-safe self-conditioning: ``lax.cond`` on a traced coin flip, stop-gradient'd.

  Mirrors ``train.py``'s ``generic_step``:
  ``if self.config.training.self_conditioning and random.random() > 0.5:
  with torch.no_grad(): sc_output = self.model.compute_energy(...)`` (the
  reference uses ``direct_score=False`` in the shipped configs, i.e. the aux
  head, not the conservative score, generates the recycled estimate -- this
  port follows that convention: the recycled pass always uses the cheap
  :func:`checkpointed_aux_score`, never the conservative score, so
  self-conditioning does not itself pay the inner-``jax.grad`` cost twice).

  The reference also draws an *independent* sequence-drop mask for this
  recycled pass ("independent dropping", ``train.py``), which this function
  replicates via its own ``key`` split.

  ``use_self_cond`` gates via :func:`jax.lax.cond` (only one branch executes
  at runtime, unlike an always-both-branches ``jnp.where`` select) -- both
  branches are still traced at compile time and must agree on output
  shape/dtype, which they do (both return ``Coords``).

  The returned estimate is always :func:`jax.lax.stop_gradient`'d --
  verified against the reference's ``torch.no_grad()`` context, which
  confirms no gradient is meant to flow back through the recycled pass
  (design spec §10 MAJOR-6).
  """

  def _generate(_: None) -> Coords:
    drop_sc = jax.random.bernoulli(key, drop_seq_prob)
    aatype_sc = jnp.where(drop_sc, jnp.full_like(aatype, MASK_AA_TOKEN), aatype)
    aux = checkpointed_aux_score(model, x_t, aatype_sc, t, mask, sc_coords=jnp.zeros_like(x_t))
    recycled = schedule.calc_trans_0(aux, x_t, t)
    return jax.lax.stop_gradient(recycled)

  def _zero(_: None) -> Coords:
    return jnp.zeros_like(x_t)

  # `key` is consumed exactly once (a single Bernoulli draw inside
  # `_generate`; `_zero` does not touch it) -- only one branch ever executes
  # at runtime (`jax.lax.cond`), so there is no double-consumption risk.
  return jax.lax.cond(use_self_cond, _generate, _zero, None)


def _per_example_loss(
  model: ProteinEBMModel,
  schedule: VPSchedule,
  coords0: Coords,
  aatype: AAType,
  mask: ResidueMask,
  t: DiffusionTime,
  key: PRNGKeyArray,
  use_self_cond: Bool[Array, ""],
  drop_seq_prob: float,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
  """Single-structure score-matching loss (vmapped over the batch axis by callers)."""
  k_diffuse, k_drop_main, k_sc = jax.random.split(key, 3)

  x_t, target = schedule.forward_marginal(coords0, t, k_diffuse)

  drop_main = jax.random.bernoulli(k_drop_main, drop_seq_prob)
  aatype_main = jnp.where(drop_main, jnp.full_like(aatype, MASK_AA_TOKEN), aatype)

  sc_coords = self_conditioning_estimate(
    model,
    x_t,
    aatype_main,
    t,
    mask,
    schedule,
    k_sc,
    use_self_cond,
    drop_seq_prob,
  )

  pred_score = checkpointed_score(model, x_t, aatype_main, t, mask, sc_coords)
  pred_aux = checkpointed_aux_score(model, x_t, aatype_main, t, mask, sc_coords)

  l_dsm = masked_mse(pred_score, target, mask)
  l_aux = masked_mse(pred_aux, target, mask)
  # L_atom: documented no-op -- see module docstring's "Loss" section
  # (design spec §1 Fork 5: no all-atom head exists on ProteinEBMModel/
  # EnergyReadout; not fabricated here).
  l_atom = jnp.zeros_like(l_dsm)

  loss = DSM_WEIGHT * l_dsm + AUX_WEIGHT * l_aux + ATOM_WEIGHT * l_atom
  metrics = {"loss": loss, "l_dsm": l_dsm, "l_aux": l_aux, "l_atom": l_atom}
  return loss, metrics


def score_matching_loss(
  model: ProteinEBMModel,
  schedule: VPSchedule,
  coords0: BatchCoords,
  aatype: BatchAAType,
  mask: BatchResidueMask,
  t: BatchTime,
  key: PRNGKeyArray,
  *,
  drop_seq_prob: float = DEFAULT_DROP_SEQ_PROB,
  self_conditioning_prob: float = DEFAULT_SELF_CONDITIONING_PROB,
) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
  """Batched score-matching loss: ``L = 3*L_DSM + 0.75*L_aux + 0.1*L_atom``.

  The self-conditioning coin flip is drawn **once per step** (shared across
  the whole batch), matching the reference's single
  ``random.random() > 0.5`` draw per ``generic_step`` call -- not per
  example. Sequence-drop, by contrast, is independent per example (matching
  the reference's ``(B,)``-shaped ``drop_mask``).

  Args:
      model: The ``ProteinEBMModel`` being trained.
      schedule: VP-SDE hyperparameters (frozen, non-trainable).
      coords0: Raw/physical per-structure CA coordinates, ``(B, N, 3)``
        (``coordinate_scaling`` is applied once inside
        ``schedule.forward_marginal``, not here).
      aatype: Per-residue amino-acid indices, ``(B, N)``.
      mask: Per-residue validity mask, ``(B, N)``.
      t: Per-example diffusion time, ``(B,)``.
      key: PRNG key for this step (coin flip + per-example draws). Not
        reused by the caller after this call.
      drop_seq_prob: Sequence-drop probability (reference default 0.1).
      self_conditioning_prob: Self-conditioning coin-flip probability
        (reference default 0.5).

  Returns:
      ``(loss, metrics)`` where ``metrics`` has keys ``loss``/``l_dsm``/
      ``l_aux``/``l_atom``, each the batch mean.

  """
  batch_size = coords0.shape[0]
  k_coin, k_batch = jax.random.split(key)
  use_self_cond = jax.random.bernoulli(k_coin, self_conditioning_prob)
  per_example_keys = jax.random.split(k_batch, batch_size)

  def _one(
    c0: Coords,
    aa: AAType,
    m: ResidueMask,
    tt: DiffusionTime,
    k: PRNGKeyArray,
  ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
    return _per_example_loss(model, schedule, c0, aa, m, tt, k, use_self_cond, drop_seq_prob)

  losses, metrics = jax.vmap(_one)(coords0, aatype, mask, t, per_example_keys)
  total_loss = jnp.mean(losses)
  agg_metrics = jax.tree_util.tree_map(jnp.mean, metrics)
  return total_loss, agg_metrics


class EBMTrainStep(eqx.Module):
  """Duck-typed ``xtrax.engine`` ``TrainStepLike`` step for the ProteinEBM training path.

  A brand-new trainer (Fork 8): does **not** reuse or modify
  ``xtrax.training.trainer.Trainer`` -- that class hardcodes a single
  ``predictions = model(batch["inputs"]); loss_fn(predictions,
  batch["targets"])`` shape, too rigid for a loss needing its own PRNG key
  (self-conditioning coin flip, sequence-drop, the forward-marginal noise
  draw) and multiple model call sites (``score``/``aux_score``) per step.
  Instead this class implements ``TrainStepLike``'s bare
  ``step(state, batch) -> (state, metrics)`` protocol directly, matching
  ``xtrax.engine.engine.Engine.trainer``'s accepted type
  (``Trainer | SafetyTrainStep | TrainStepLike``) -- verified by reading
  ``xtrax/engine/engine.py`` directly; no workaround needed.
  """

  optimizer: optax.GradientTransformation
  schedule: VPSchedule = eqx.field(default_factory=VPSchedule)
  drop_seq_prob: float = eqx.field(static=True, default=DEFAULT_DROP_SEQ_PROB)
  self_conditioning_prob: float = eqx.field(static=True, default=DEFAULT_SELF_CONDITIONING_PROB)

  @eqx.filter_jit
  def step(
    self,
    state: ResumableState,
    batch: dict[str, Array],
  ) -> tuple[ResumableState, dict[str, Array]]:
    """One training step: loss, grad, optax update, incremented + re-keyed state.

    ``batch`` must contain ``"coords0"``/``"aatype"``/``"mask"``/``"t"``
    (see :func:`score_matching_loss`'s Args for shapes). PRNG discipline:
    ``state.key`` is split into ``(key_step, key_next)`` -- ``key_step``
    is consumed exactly once by the loss, ``key_next`` becomes the new
    state's key. Never reused.
    """
    key_step, key_next = jax.random.split(state.key)

    def loss_fn(
      model: ProteinEBMModel,
    ) -> tuple[Float[Array, ""], dict[str, Float[Array, ""]]]:
      return score_matching_loss(
        model,
        self.schedule,
        batch["coords0"],
        batch["aatype"],
        batch["mask"],
        batch["t"],
        key_step,
        drop_seq_prob=self.drop_seq_prob,
        self_conditioning_prob=self.self_conditioning_prob,
      )

    (_loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)

    filtered_params = eqx.filter(state.model, eqx.is_array)
    updates, new_opt_state = self.optimizer.update(grads, state.opt_state, filtered_params)
    new_model = eqx.apply_updates(state.model, updates)

    grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
    grad_sq_sum = sum((jnp.sum(g * g) for g in grad_leaves), start=jnp.zeros(()))
    metrics = {**metrics, "grad_norm": jnp.sqrt(grad_sq_sum)}

    new_state = eqx.tree_at(
      lambda s: (s.model, s.opt_state, s.step, s.key),
      state,
      (new_model, new_opt_state, state.step + 1, key_next),
    )
    return new_state, metrics


def build_optimizer(
  peak_lr: float,
  warmup_steps: int,
  total_steps: int,
  *,
  weight_decay: float = 1e-2,
  clip_norm: float = DEFAULT_GRADIENT_CLIP_NORM,
) -> optax.GradientTransformation:
  """AdamW + warmup/cosine-decay schedule + grad clipping (see module docstring's "Optimizer" note).

  Thin wrapper over ``xtrax.training.optim.adamw_with_schedule``. Not a
  literal one-cycle schedule (xtrax ships no such helper) -- an intentional,
  documented approximation of the reference's ``OneCycleLR`` intent.
  ``clip_norm`` defaults to ``10.0``, matching the reference's hardcoded
  ``gradient_clip_val=10.0``.
  """
  return adamw_with_schedule(
    peak_lr=peak_lr,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    weight_decay=weight_decay,
    clip_norm=clip_norm,
  )


def init_training_state(
  model: ProteinEBMModel,
  optimizer: optax.GradientTransformation,
  seed: int,
) -> ResumableState:
  """Build the initial ``ResumableState`` (step 0, fresh PRNG key, optimizer-initialized)."""
  return ResumableState(
    step=jnp.asarray(0, dtype=jnp.int32),
    key=jax.random.PRNGKey(seed),
    model=model,
    opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
    extras={},
  )


def make_engine(train_step: EBMTrainStep, callbacks: tuple = ()) -> Engine:
  """Wrap an :class:`EBMTrainStep` in an ``xtrax.engine.Engine`` for ``fit``/``fit_sync``."""
  return Engine(trainer=train_step, callbacks=callbacks)


def get_ebm_checkpoint_manager(
  directory: str | Path,
  max_to_keep: int | None = 5,
) -> ocp.CheckpointManager:
  """Thin re-export of ``xtrax.checkpoint.get_checkpoint_manager`` (see module docstring)."""
  return get_checkpoint_manager(directory, max_to_keep=max_to_keep)


def save_ebm_checkpoint(manager: ocp.CheckpointManager, state: ResumableState) -> None:
  """Thin re-export of ``xtrax.checkpoint.save_checkpoint``."""
  save_checkpoint(manager, state)


def load_ebm_checkpoint(
  manager: ocp.CheckpointManager,
  state_template: ResumableState,
  step: int | None = None,
) -> ResumableState:
  """Thin re-export of ``xtrax.checkpoint.load_checkpoint``."""
  return load_checkpoint(manager, state_template, step=step)


__all__ = [
  "ATOM_WEIGHT",
  "AUX_WEIGHT",
  "DEFAULT_DROP_SEQ_PROB",
  "DEFAULT_GRADIENT_CLIP_NORM",
  "DEFAULT_SELF_CONDITIONING_PROB",
  "DSM_WEIGHT",
  "MASK_AA_TOKEN",
  "EBMTrainStep",
  "build_optimizer",
  "checkpointed_aux_score",
  "checkpointed_score",
  "checkpointed_trunk_fn",
  "get_ebm_checkpoint_manager",
  "init_training_state",
  "load_ebm_checkpoint",
  "make_engine",
  "masked_mse",
  "save_ebm_checkpoint",
  "score_matching_loss",
  "self_conditioning_estimate",
]
