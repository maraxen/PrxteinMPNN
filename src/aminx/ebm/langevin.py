"""Fixed-noise-level Langevin/reverse-SDE equilibration (backlog node **E9**, CORE only).

Ported from ``~/repos/ProteinEBM/protein_ebm/scripts/run_dynamics.py`` (the
inner per-``t`` update loop, lines ~750-870) and
``~/repos/ProteinEBM/protein_ebm/model/r3_diffuser.py`` (``diffusion_coef``/
``drift_coef``, now in :mod:`aminx.ebm.diffusion`). See
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §1 Fork 7 and
§3 risk register (BLOCKER-1/MAJOR-4), and design spec §10/§3.3.

**Deliberate scope limit for this dispatch.** This module implements *only*
the inner Langevin loop at a single, fixed noise level ``t``, against a
single, already-loaded :class:`~aminx.ebm.model.ProteinEBMModel` instance.
The **outer** noise-schedule loop -- an ``xtrax`` ``CarrySpec``+``Scan`` over
a ``t`` schedule with a net-new ``lax.cond`` dispatcher that swaps between
several *pre-loaded* checkpoints as ``t`` crosses per-checkpoint thresholds
(the reference's ``get_dynamics_model(t)``) -- is explicitly **out of
scope** here and is left as a documented extension point (see
:func:`run_langevin_equilibration`'s docstring). A future backlog node will
wrap this module's :func:`run_langevin_equilibration` as the body of that
outer scan.

**Inference-only, not differentiable.** Per the ``using-jax`` skill's
"``lax.scan`` vs ``lax.while_loop`` for autoregressive loops" guidance (and
the design's Fork 7 resolution): :func:`run_langevin_equilibration` uses
``jax.lax.while_loop`` for its fixed (but not compile-time-constant)
``n_steps`` trip count. This lowers to a single XLA ``While`` op --
~300-400x faster to compile than the ``lax.scan`` equivalent on SM120 -- at
the cost of **not being reverse-mode differentiable**. That tradeoff is
correct here because this whole module is an *inference-time* sampler, never
called under ``jax.grad``/``eqx.filter_grad`` (contrast E8's training loop,
which legitimately needs ``lax.scan``).

**Score/energy source: the cheap aux head, not the conservative one.** Every
step in this module reads ``model.aux_score``/``model.energy`` -- the cheap,
non-conservative surrogate head (no ``jax.grad`` inside the sampling loop).
This matches the reference's ``use_aux_for_this_round`` branch. The
*conservative* score (``model.score``, ``-jax.grad(energy)``) is reserved for
a separate, not-yet-built "rescoring" step (the reference's
``args.scoring_time`` / final-frame re-evaluation pass) -- **not** part of
this dispatch. Extension point: a future ``rescore_trajectory`` function
would call ``model.score``/``model.energy`` once per retained sample, after
:func:`run_langevin_equilibration` has produced it, entirely outside the hot
loop.

PRNG discipline: every function here takes a single ``key`` and documents
exactly how many times/where it is split -- see each function's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import jax
import jax.numpy as jnp

from aminx.ebm.diffusion import DEFAULT_COORDINATE_SCALING, drift_coef
from aminx.ebm.diffusion import diffusion_coef as _diffusion_coef
from aminx.ebm.trunk import center_random_augmentation

if TYPE_CHECKING:
  from jaxtyping import Bool, PRNGKeyArray

  from aminx.ebm.contracts import AAType, Coords, DiffusionTime, Energy, ResidueMask, Score


class _AuxScoringModel(Protocol):
  """Structural protocol for this module's ``model`` argument.

  Anything exposing these two call signatures works -- production callers
  pass a :class:`aminx.ebm.model.ProteinEBMModel`; tests may substitute a
  lightweight stand-in exposing the same two methods (see
  ``tests/ebm/test_langevin.py``). Declared as ``typing.Protocol`` (not a
  plain class) specifically so this is *structural* typing -- callers don't
  need to subclass it, they just need matching method signatures; ``ty``/
  ``pyright`` check this duck-typing contract statically.
  """

  if TYPE_CHECKING:

    def aux_score(
      self,
      coords: Coords,
      aatype: AAType,
      t: DiffusionTime,
      mask: ResidueMask,
    ) -> Score: ...

    def energy(
      self,
      coords: Coords,
      aatype: AAType,
      t: DiffusionTime,
      mask: ResidueMask,
    ) -> Energy: ...


DEFAULT_EFFECTIVE_TEMP_SCALING = 1.0


def langevin_step(
  model: _AuxScoringModel,
  coords: Coords,
  aatype: AAType,
  t: DiffusionTime,
  dt: float,
  mask: ResidueMask,
  key: PRNGKeyArray,
  *,
  effective_temp_scaling: float = DEFAULT_EFFECTIVE_TEMP_SCALING,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> Coords:
  """One plain (non-Metropolis) Euler-Maruyama reverse-SDE step at fixed ``t``.

  ``pos_t_new = pos_t - (drift_coef(pos_t, t) - diffusion_coef(t)**2 * score
  * effective_temp_scaling / coordinate_scaling) * dt + diffusion_coef(t) *
  sqrt(dt) * noise / coordinate_scaling``, followed by a re-centering (**not**
  re-rotating) call to :func:`aminx.ebm.trunk.center_random_augmentation`
  (``rotate=False``) -- mirrors ``run_dynamics.py``'s non-Metropolis branch
  (~lines 830-843) exactly, including that ``augmentation`` stays at its
  default ``True`` (the reference's call site does not pass
  ``augmentation=False`` either), so a random rigid translation is applied
  every step in addition to centering. ``score``/``energy`` come from
  ``model.aux_score`` -- the cheap non-conservative surrogate (module
  docstring).

  PRNG discipline: ``key`` is split exactly once into ``key_noise`` (the
  Euler-Maruyama Gaussian draw, consumed by ``jax.random.normal``) and
  ``key_center`` (passed to ``center_random_augmentation``, which internally
  splits it again for its own rotation/translation draws). Callers must
  split before reuse.

  Args:
      model: Anything exposing ``aux_score(coords, aatype, t, mask)``.
      coords: Current coordinates, ``(N, 3)``, scaled space.
      aatype: Per-residue amino-acid type, ``(N,)``.
      t: Fixed diffusion time for this step (``t in [0, 1]``).
      dt: Euler-Maruyama step size.
      mask: Residue validity mask, ``(N,)``.
      key: PRNG key, split exactly once (see above).
      effective_temp_scaling: Temperature-correction hyperparameter (design
        spec's ``T_train/T_sim``; default 1.0 == no correction).
      coordinate_scaling: VP-SDE coordinate-scaling constant (must match the
        value used to produce ``coords`` in the first place).

  Returns:
      Updated, re-centered coordinates, ``(N, 3)``.

  """
  key_noise, key_center = jax.random.split(key)

  score = model.aux_score(coords, aatype, t, mask)
  g = _diffusion_coef(t)
  drift = drift_coef(coords, t)
  noise = jax.random.normal(key_noise, shape=coords.shape, dtype=coords.dtype)

  coords_new = (
    coords
    - (drift - g**2 * score * effective_temp_scaling / coordinate_scaling) * dt
    + g * jnp.sqrt(dt) * noise / coordinate_scaling
  )

  return center_random_augmentation(coords_new, mask, key_center, rotate=False)


def metropolis_hastings_step(
  model: _AuxScoringModel,
  coords: Coords,
  aatype: AAType,
  t: DiffusionTime,
  dt: float,
  mask: ResidueMask,
  key: PRNGKeyArray,
  *,
  effective_temp_scaling: float = DEFAULT_EFFECTIVE_TEMP_SCALING,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> tuple[Coords, Energy, Score, Bool[jax.Array, ""]]:
  """One Metropolis-Hastings-corrected Langevin proposal/accept step at fixed ``t``.

  Mirrors ``run_dynamics.py``'s ``metropolis`` branch (~lines 762-799)
  exactly. Both the current-state and proposed-state score/energy are
  computed **unconditionally** (both are needed for the accept ratio
  regardless of the outcome), then the accepted result is selected via
  ``jnp.where`` on a scalar boolean mask -- **no Python ``if``**, so this is
  fully ``jax.jit``/``jax.vmap``-traceable.

  ::

      next_mean = pos_t + diffusion_coef(t)**2 * score * temp_scale / coordinate_scaling * dt
      next_noise = sqrt(2) * diffusion_coef(t) * sqrt(dt) * noise / coordinate_scaling
      pos_t_proposed = next_mean + next_noise
      proposed_next_mean = pos_t_proposed + diffusion_coef(t)**2 * score_proposed * temp_scale / coordinate_scaling * dt
      energy_ratio = exp(-(energy_proposed - energy_current) * temp_scale)
      kernel_ratio = exp(-((pos_t - proposed_next_mean)**2.sum() - next_noise**2.sum())
                          / (2 * diffusion_coef(t)**2 * dt / coordinate_scaling**2) / 2)
      accept_ratio = energy_ratio * kernel_ratio
      accept = uniform_random() < accept_ratio

  Note (matches the reference exactly, not a mask-aware approximation):
  ``kernel_ratio``'s squared-difference sums run over *all* elements of
  ``(N, 3)`` -- there is no residue-mask restriction in this term in
  ``run_dynamics.py`` either (verified by reading the source). Masked
  residues still contribute zero ``score``/``aux_score`` (per
  ``ProteinEBMModel``'s own masking, see ``tests/ebm/test_model.py``'s
  ``test_masked_residues_do_not_contribute_to_score``), so their positions
  only random-walk under the noise term -- a documented, reference-faithful
  approximation, not a bug in this port.

  PRNG discipline: ``key`` is split exactly once into ``key_noise`` (the
  proposal's Gaussian draw) and ``key_accept`` (the uniform accept/reject
  draw). Callers must split before reuse.

  Args:
      model: Anything exposing ``aux_score``/``energy``.
      coords: Current coordinates, ``(N, 3)``, scaled space.
      aatype: Per-residue amino-acid type, ``(N,)``.
      t: Fixed diffusion time for this step.
      dt: Euler-Maruyama step size.
      mask: Residue validity mask, ``(N,)``.
      key: PRNG key, split exactly once (see above).
      effective_temp_scaling: Temperature-correction hyperparameter.
      coordinate_scaling: VP-SDE coordinate-scaling constant.

  Returns:
      ``(coords_new, energy_new, score_new, accepted)`` -- ``accepted`` is a
      scalar boolean (single-structure convention, matching
      ``ProteinEBMModel.energy``'s scalar return; batch over structures via
      an external ``jax.vmap``, not built into this function).

  """
  key_noise, key_accept = jax.random.split(key)

  score_current = model.aux_score(coords, aatype, t, mask)
  energy_current = model.energy(coords, aatype, t, mask)
  g = _diffusion_coef(t)

  next_mean = coords + g**2 * score_current * effective_temp_scaling / coordinate_scaling * dt
  noise = jax.random.normal(key_noise, shape=coords.shape, dtype=coords.dtype)
  next_noise = jnp.sqrt(2.0) * g * jnp.sqrt(dt) * noise / coordinate_scaling
  coords_proposed = next_mean + next_noise

  score_proposed = model.aux_score(coords_proposed, aatype, t, mask)
  energy_proposed = model.energy(coords_proposed, aatype, t, mask)

  proposed_next_mean = (
    coords_proposed + g**2 * score_proposed * effective_temp_scaling / coordinate_scaling * dt
  )

  energy_ratio = jnp.exp(-(energy_proposed - energy_current) * effective_temp_scaling)
  sq_diff_to_proposed_mean = jnp.sum((coords - proposed_next_mean) ** 2)
  sq_noise = jnp.sum(next_noise**2)
  kernel_ratio = jnp.exp(
    -(sq_diff_to_proposed_mean - sq_noise) / (2 * g**2 * dt / coordinate_scaling**2) / 2,
  )
  accept_ratio = energy_ratio * kernel_ratio

  u = jax.random.uniform(key_accept, shape=())
  accept = u < accept_ratio

  coords_new = jnp.where(accept, coords_proposed, coords)
  energy_new = jnp.where(accept, energy_proposed, energy_current)
  score_new = jnp.where(accept, score_proposed, score_current)

  return coords_new, energy_new, score_new, accept


def run_langevin_equilibration(
  model: _AuxScoringModel,
  coords: Coords,
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  n_steps: int,
  dt: float,
  key: PRNGKeyArray,
  *,
  use_metropolis: bool = False,
  effective_temp_scaling: float = DEFAULT_EFFECTIVE_TEMP_SCALING,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> Coords:
  """Run exactly ``n_steps`` of :func:`langevin_step`/:func:`metropolis_hastings_step` at fixed ``t``.

  This is the **inner** loop only (module docstring: single, fixed noise
  level; single, pre-loaded model). It uses ``jax.lax.while_loop`` with a
  carry of ``(step_idx, coords, key)`` -- **not** a richer carry threading
  ``energy``/``score`` across steps. That is a deliberate simplification,
  not an oversight: :func:`metropolis_hastings_step`'s signature always
  recomputes ``model.aux_score``/``model.energy`` at the *current* position
  from scratch on every call (its contract computes both current- and
  proposed-state quantities unconditionally, every call, so they can be
  selected via ``jnp.where``) -- so carrying the previous step's
  energy/score forward would not actually save any model evaluations under
  the current function signatures. A future optimization could thread
  ``(energy, score)`` through the carry *and* give
  ``metropolis_hastings_step`` an optional ``current_energy``/
  ``current_score`` override to skip that recomputation; this dispatch keeps
  the carry minimal and the two step functions' contracts self-contained
  instead, since correctness (not throughput) is this node's acceptance bar.

  **Extension point (future node, out of scope here).** The *outer* loop --
  varying ``t`` across a noise schedule, with a ``lax.cond`` dispatcher
  swapping between multiple pre-loaded checkpoints as ``t`` crosses
  thresholds (the reference's ``get_dynamics_model(t)``) -- is not
  implemented in this module. A future node wraps *this* function as the
  per-``t`` body of an ``xtrax`` ``CarrySpec``+``Scan`` over the noise
  schedule, with ``coords``/``key`` (and, if the checkpoint-swap is
  incorporated, the model-selection state) as the scan carry.

  PRNG discipline: ``key`` is split once per step inside the loop body
  (``key, step_key = jax.random.split(key)``); each of the ``n_steps`` calls
  to :func:`langevin_step`/:func:`metropolis_hastings_step` gets its own
  fresh, never-reused subkey.

  Args:
      model: Anything exposing ``aux_score``/``energy``.
      coords: Initial coordinates, ``(N, 3)``, scaled space.
      aatype: Per-residue amino-acid type, ``(N,)``.
      t: Fixed diffusion time for the entire run.
      mask: Residue validity mask, ``(N,)``.
      n_steps: Exact number of steps to run (traced or static; ``while_loop``
        does not require it to be a compile-time constant, unlike
        ``lax.scan``'s ``length``).
      dt: Euler-Maruyama step size, held fixed across all ``n_steps``.
      key: PRNG key for the whole run (see PRNG discipline above).
      use_metropolis: If ``True``, use :func:`metropolis_hastings_step`
        (accept/reject-corrected); if ``False`` (default), use
        :func:`langevin_step` (plain Euler-Maruyama). Must be a Python
        ``bool`` (static), not a traced value -- it selects which pure
        function the loop body calls, decided once at trace time.
      effective_temp_scaling: Temperature-correction hyperparameter, forwarded
        to the per-step function.
      coordinate_scaling: VP-SDE coordinate-scaling constant, forwarded to
        the per-step function.

  Returns:
      Final coordinates after ``n_steps``, ``(N, 3)``.

  """
  n_steps_arr = jnp.asarray(n_steps)

  def cond_fun(carry: tuple[jax.Array, Coords, PRNGKeyArray]) -> jax.Array:
    step, _coords, _key = carry
    return step < n_steps_arr

  def body_fun(
    carry: tuple[jax.Array, Coords, PRNGKeyArray],
  ) -> tuple[jax.Array, Coords, PRNGKeyArray]:
    step, coords_i, key_i = carry
    key_i, step_key = jax.random.split(key_i)
    if use_metropolis:
      coords_next, _energy, _score, _accepted = metropolis_hastings_step(
        model,
        coords_i,
        aatype,
        t,
        dt,
        mask,
        step_key,
        effective_temp_scaling=effective_temp_scaling,
        coordinate_scaling=coordinate_scaling,
      )
    else:
      coords_next = langevin_step(
        model,
        coords_i,
        aatype,
        t,
        dt,
        mask,
        step_key,
        effective_temp_scaling=effective_temp_scaling,
        coordinate_scaling=coordinate_scaling,
      )
    return step + 1, coords_next, key_i

  init_carry = (jnp.asarray(0), coords, key)
  _final_step, final_coords, _final_key = jax.lax.while_loop(cond_fun, body_fun, init_carry)
  return final_coords


__all__ = [
  "DEFAULT_EFFECTIVE_TEMP_SCALING",
  "langevin_step",
  "metropolis_hastings_step",
  "run_langevin_equilibration",
]
