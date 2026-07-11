"""Multi-round structure-prediction pipeline (backlog node **E10**).

Ported from ``~/repos/ProteinEBM/protein_ebm/scripts/run_dynamics.py``'s
**multi-round resampling** driver (the ``for round_idx in
range(args.num_resample_rounds):`` outer loop, lines ~592-962) -- the part of
that file explicitly carved *out* of E9's scope (see
``aminx.ebm.langevin_schedule``'s module docstring: "The reference's
multi-round resampling machinery ... is a separate backlog node, E10").
Design authority: ``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md``
§2 (E10 row) + §3 risk register, and
``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §3.3's
"Structure prediction (MSA-free)" row: *"initial-sample -> resample
(Boltzmann) -> 3x refine -> optional AF2Rank"* / *"``pipeline()`` of ``Scan``
stages, with an importance-resampling ``Fuse`` between rounds"* / *"Clustering
+ AF2Rank are host-side post-processing (numpy/scipy, outside jit) -- a
``Sink``/``Tap``, not a traced stage."*

**No literal ``xtrax.pipeline()`` API exists in this repo's installed xtrax
version** (grepped, no match) -- the design spec's "``pipeline()`` of
``Scan`` stages" describes the *shape* of the computation, not a function
name. This module implements that shape as a plain Python ``for round_idx in
range(num_rounds)`` loop (a small, static round count) that calls the
already-committed :func:`aminx.ebm.langevin_schedule.run_annealing_schedule`
once per round (via ``jax.vmap`` for a batch of independent trajectories),
with a real, host-side (numpy) resampling step
(:func:`resample_ensemble`) between rounds -- **not** an
``jax.lax.scan`` over rounds, because the host-side quantile filter produces
a data-dependent-shape intermediate (the filtered candidate pool) that is not
JAX-traceable, exactly as the design authority's own "outside jit" framing
says.

**Deliberate scope limits (matching the dispatch brief, not oversights):**

1. **Same model set across all rounds.** The reference's
   ``resample_dynamics_model`` (``get_dynamics_model``, lines 262-268) is
   populated only from ``round_idx >= 1``, via a *separate*, optional
   checkpoint (``--resample_dynamics_checkpoint``, default empty/off). That
   would require loading a **third** checkpoint just for rounds 2+ -- a
   real, separate feature the reference itself gates behind its own opt-in
   CLI flag, genuinely out of scope here. This module reuses the *same*
   ``models``/``thresholds`` pair (E9's ``select_model_for_t`` dispatch) for
   every round.
2. **Plain quantile-threshold resampling only.** The reference's default
   path (verified, ``run_dynamics.py`` lines 954-958, the ``else`` branch of
   its clustering-vs-plain-quantile fork):

   .. code-block:: python

      round_thresh = torch.quantile(round_min_traj_nrg, args.quantile_thresh_resample)
      round_filtered_nrgs = round_min_traj_nrg[round_min_traj_nrg < round_thresh]
      round_filtered_structs = round_final_structures[round_min_traj_nrg < round_thresh]
      round_filtered_idxs = round_min_traj_idx[round_min_traj_nrg < round_thresh]

   The scipy-hierarchical-clustering alternative
   (``cluster_structures_for_resampling``, lines 282-324, gated behind
   ``args.cluster_resampling_allrounds``, itself optional in the reference)
   is a **deliberately deferred extension point** -- not implemented here,
   not silently dropped either.
3. **AF2Rank rescoring is explicitly out of scope** (design spec: "optional")
   -- not implemented; a future extension point would rescore each round's
   (or the final round's) retained structures with AF2Rank as a further
   host-side ``Sink``, after :func:`run_structure_prediction` returns.

**Renoising and the coordinate-scaling boundary (a documented departure from
verbatim reference replication, forced by E9's existing return contract).**
The reference's between-round renoise step (lines 686-702) calls
``diffuser.forward_marginal`` on ``pos_allatom`` -- the model's own
*predicted clean-structure* (``out['pred_coords']``/``pred_coords_aux``),
extracted CA-only, in **raw/physical** coordinate space, then applies
``coordinate_scaling`` exactly once inside ``forward_marginal``. This
module's caller, :func:`aminx.ebm.langevin_schedule.run_annealing_schedule`,
does not expose any such clean-structure estimate in its return contract --
it returns only the final scan-carried ``coords``, already in the **scaled**
``Coords`` space (E9 is explicitly not modified by this dispatch). Rather
than requiring a third-file change to thread out an unavailable quantity,
:func:`resample_ensemble` renoises the round's final (already-low-``t``,
already-scaled) ``coords`` **directly**, treating them as a serviceable proxy
for the true clean-structure estimate (they are close to it by construction,
having descended to the bottom of the schedule) and calling
``forward_marginal`` with ``coordinate_scaling=1.0`` (a documented no-op) so
the already-applied scaling is never re-applied -- preserving the "applied
exactly once" invariant ``aminx.ebm.diffusion``'s own module docstring
establishes, rather than reintroducing the reference's three-call-site
footgun (design spec §10 MINOR finding). This is a real, flagged numerical
approximation of the reference's exact renoising target, not a bug.

**Round 0 starting-batch construction (resolved from the reference, not
guessed).** ``run_dynamics.py`` lines 613-627 (``if args.start_unfolded ...
else:``, the default/non-unfolded path) show every trajectory in the batch
starts from the **same** input structure -- ``atom_positions[...,1,:]
.unsqueeze(0).expand([bsize,-1,-1])`` -- centered via
``center_random_augmentation`` (default, rotating too). Then lines 704-713
(the ``round_idx == 0`` branch, ``args.t_max != 1.0``) call
``diffuser.forward_marginal`` **once** on that (batch-tiled) structure; since
``forward_marginal``'s Gaussian noise draw has one independent sample per
tensor element, tiling the *same* mean across the batch before a single
noise draw is exactly equivalent to giving each of the ``bsize`` trajectories
an *independent* Gaussian draw around an *identical* mean -- i.e. **same
starting structure, independent per-trajectory noise**, not independent
starting structures. :func:`run_structure_prediction` reproduces this via
``jnp.broadcast_to`` + ``jax.vmap(forward_marginal)`` over ``batch_size``
independent keys. The reference's extra pre-noise pose-randomization
(``center_random_augmentation``'s rigid rotation before the noise draw) is
**not** reproduced here -- it is a batch-diversity nicety on top of the core
noising step, not part of the noise-schedule/resampling algorithm itself,
and every :func:`aminx.ebm.langevin.langevin_step` call already re-centers
(``rotate=False``) every step regardless, so trajectory diversity still
comes from independent per-step/per-trajectory Gaussian noise, just as in the
reference's actual divergence mechanism.

PRNG discipline: every function here takes a single ``key`` and documents
exactly how many times/where it is split -- see each function's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aminx.ebm.diffusion import DEFAULT_COORDINATE_SCALING, forward_marginal
from aminx.ebm.langevin import DEFAULT_EFFECTIVE_TEMP_SCALING
from aminx.ebm.langevin_schedule import run_annealing_schedule, select_model_for_t

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from aminx.ebm.contracts import AAType, Coords, Energy, ResidueMask
  from aminx.ebm.model import ProteinEBMModel

  NoiseLevelSchedule = Float[Array, "L"]
  StepCountSchedule = Int[Array, "L"]
  Coords_batch = Float[Array, "B N 3"]
  Energy_batch = Float[Array, "B"]

DEFAULT_RESAMPLE_NOISE_TIME = 0.2
DEFAULT_QUANTILE_THRESH = 0.25
DEFAULT_ENERGY_SCALING = 200.0


def _calculate_importance_weights(
  filtered_energies: jax.Array,
  energy_scaling: float,
) -> jax.Array:
  """Port of ``run_dynamics.py``'s ``calculate_importance_weights`` (lines 270-279).

  ``energy_scaling == -1`` -> uniform weights (``1/len``); otherwise
  ``softmax(-filtered_energies / energy_scaling)`` (Boltzmann weighting at
  temperature ``energy_scaling``). No PRNG use -- purely deterministic given
  ``filtered_energies``.
  """
  if energy_scaling == -1:
    return jnp.ones_like(filtered_energies) / filtered_energies.shape[0]
  return jax.nn.softmax(-filtered_energies / energy_scaling)


def resample_ensemble(
  coords_batch: Coords_batch,
  energies_batch: Energy_batch,
  key: PRNGKeyArray,
  *,
  batch_size: int,
  quantile_thresh: float = DEFAULT_QUANTILE_THRESH,
  energy_scaling: float = DEFAULT_ENERGY_SCALING,
  resample_noise_time: float = DEFAULT_RESAMPLE_NOISE_TIME,
) -> Coords_batch:
  """Host-side (numpy) quantile filter -> importance weights -> multinomial resample -> renoise.

  Ports, in order: ``run_dynamics.py``'s plain quantile-threshold filtering
  path (lines 954-958, the ``else`` branch of the clustering-vs-plain fork --
  the scipy hierarchical-clustering alternative,
  ``cluster_structures_for_resampling`` lines 282-324, is a documented,
  deliberately deferred extension point, not implemented here);
  :func:`_calculate_importance_weights` (lines 270-279); the reference's
  ``torch.multinomial(importance_weights, bsize, replacement=True)``
  (line 693/959, here as ``jax.random.choice(..., replace=True, p=...)``);
  and the between-round forward-renoise (lines 686-702), via
  :func:`aminx.ebm.diffusion.forward_marginal` -- see this module's own
  docstring's "Renoising and the coordinate-scaling boundary" section for
  why ``coordinate_scaling=1.0`` is passed to that call (a documented
  no-op, not a bug).

  **Host-side vs. JAX-math split.** The quantile filter itself is done in
  plain numpy (the reference's ``torch.quantile``/boolean-mask filtering is
  data-dependent-shape -- exactly the ``Sink``/host-side framing the design
  spec calls for, and not something ``jax.jit`` could trace even if we
  wanted it to). Everything downstream of the filter -- softmax weighting,
  the multinomial draw, and the renoise math -- uses JAX arrays/PRNG, since
  their *output* shape (``batch_size``) is static even though the *input*
  candidate-pool size is not.

  **Degenerate-pool guard (not in the reference, added defensively).** If
  the quantile filter's strict ``<`` comparison happens to exclude every
  candidate (e.g. all energies exactly tied at the threshold value), this
  function falls back to keeping the *entire* unfiltered batch rather than
  raising or dividing by zero -- an edge case the reference's PyTorch code
  does not guard against either, but one this port chooses not to crash on.

  PRNG discipline: ``key`` is split exactly twice: once into
  ``key_resample`` (consumed by the single ``jax.random.choice`` multinomial
  draw) and ``key_renoise`` (further split, once per batch element, into
  ``batch_size`` independent subkeys, each consumed by exactly one
  :func:`aminx.ebm.diffusion.forward_marginal` call). Callers must split
  before reuse.

  Args:
      coords_batch: This round's final coordinates, ``(B, N, 3)``, scaled
        ``Coords`` space (see module docstring for why these -- not a
        separate clean-structure estimate -- are what gets renoised).
      energies_batch: This round's final per-trajectory energies, ``(B,)``,
        used both to rank (quantile filter) and to weight (importance
        weights) candidates.
      key: PRNG key for this call (see PRNG discipline above).
      batch_size: Number of independently-resampled trajectories to produce
        for the next round -- need not equal ``coords_batch.shape[0]``,
        matching the reference's ``bsize`` (next round's batch size) vs.
        ``len(filtered_structs)`` (this round's candidate pool) distinction.
      quantile_thresh: Keep only the best (lowest-energy) fraction of
        trajectories, by final energy, strictly below this quantile.
      energy_scaling: Forwarded to :func:`_calculate_importance_weights`
        (``-1`` -> uniform; else Boltzmann temperature).
      resample_noise_time: The ``t`` to renoise resampled candidates up to
        before the next round (reference's ``args.resample_noise_time``,
        default ``0.2``).

  Returns:
      The next round's starting batch, ``(batch_size, N, 3)``, already
      renoised to ``resample_noise_time`` in the scaled ``Coords`` space.

  """
  key_resample, key_renoise = jax.random.split(key)

  energies_np = np.asarray(jax.device_get(energies_batch))
  round_thresh = np.quantile(energies_np, quantile_thresh)
  keep_mask = energies_np < round_thresh
  if not np.any(keep_mask):
    # Degenerate-pool guard (see docstring) -- not in the reference.
    keep_mask = np.ones_like(keep_mask, dtype=bool)

  filtered_coords = coords_batch[keep_mask]
  filtered_energies = jnp.asarray(energies_np[keep_mask])

  importance_weights = _calculate_importance_weights(filtered_energies, energy_scaling)

  candidate_idx = jax.random.choice(
    key_resample,
    filtered_coords.shape[0],
    shape=(batch_size,),
    replace=True,
    p=importance_weights,
  )
  resampled_coords = filtered_coords[candidate_idx]

  renoise_keys = jax.random.split(key_renoise, batch_size)

  def _renoise_one(coords: Coords, renoise_key: PRNGKeyArray) -> Coords:
    x_t, _score = forward_marginal(
      coords,
      jnp.asarray(resample_noise_time),
      renoise_key,
      coordinate_scaling=1.0,
    )
    return x_t

  return jax.vmap(_renoise_one)(resampled_coords, renoise_keys)


def _schedule_for_round(
  noise_schedule_arr: jax.Array,
  n_steps_per_level: int | StepCountSchedule,
  resample_noise_time: float,
) -> tuple[jax.Array, int | jax.Array]:
  """Truncate the full schedule to the suffix at/below ``resample_noise_time`` (rounds 1+ only).

  **Why this exists (a real correctness fix, not a style choice).** Rounds
  1+ start from structures renoised only up to ``resample_noise_time``
  (reference default ``0.2``), *not* up to the schedule's top ``t_max``
  (reference ``run_dynamics.py`` lines 628-632: ``args.t_max`` is
  reassigned to ``args.resample_noise_time`` for ``round_idx > 0`` before
  that round's ``reverse_steps = np.linspace(args.t_min, args.t_max, ...)
  [::-1]`` is (re)built). Naively reusing the *full* ``noise_schedule``
  (starting back at the original ``t_max``) for every round would silently
  feed :func:`aminx.ebm.langevin_schedule.run_annealing_schedule` structures
  that are physically at ``t=resample_noise_time`` while telling it they are
  at ``t=noise_schedule[0]`` (the original, much higher, ``t_max``) -- a real
  ``t``-mislabeling bug, not merely an approximation. This helper instead
  takes the **suffix** of the caller's descending ``noise_schedule`` at or
  below ``resample_noise_time`` (valid because the schedule is assumed
  monotonically descending, matching
  :func:`aminx.ebm.langevin_schedule.run_annealing_schedule`'s own documented
  ``linspace(...)[::-1]`` convention) -- same per-level spacing/step-counts as
  originally supplied, just starting partway through, rather than
  reconstructing a wholly new ``linspace`` with an independent
  ``resample_reverse_steps`` count (the reference's ``args.resample_steps``/
  ``args.resample_reverse_steps`` knobs are not part of this function's
  simplified signature -- a documented scope limit, not an oversight).

  **Degenerate-schedule guard (not in the reference).** If
  ``resample_noise_time`` falls below every scheduled ``t`` (e.g. a coarse
  schedule whose lowest level already exceeds it), this falls back to
  keeping just the schedule's last (lowest-``t``) level, so the round still
  runs at least one level instead of operating on an empty schedule.

  Args:
      noise_schedule_arr: The full, original (round-0) descending schedule,
        ``(L,)``.
      n_steps_per_level: Either a single ``int`` (returned unchanged -- it
        already broadcasts to any schedule length) or a per-level ``(L,)``
        array (sliced to match the truncated schedule).
      resample_noise_time: Keep schedule entries ``<= resample_noise_time``.

  Returns:
      ``(round_schedule, round_n_steps)`` for this (non-zero) round.

  """
  keep = noise_schedule_arr <= resample_noise_time
  if not bool(jnp.any(keep)):
    keep = jnp.zeros_like(noise_schedule_arr, dtype=bool).at[-1].set(True)

  round_schedule = noise_schedule_arr[keep]
  if isinstance(n_steps_per_level, int):
    round_n_steps: int | jax.Array = n_steps_per_level
  else:
    round_n_steps = jnp.asarray(n_steps_per_level)[keep]
  return round_schedule, round_n_steps


def run_structure_prediction(
  models: tuple[ProteinEBMModel, ...],
  thresholds: tuple[float, ...],
  initial_coords: Coords,
  aatype: AAType,
  mask: ResidueMask,
  noise_schedule: NoiseLevelSchedule,
  n_steps_per_level: int | StepCountSchedule,
  dt: float,
  key: PRNGKeyArray,
  *,
  num_rounds: int = 3,
  batch_size: int,
  resample_noise_time: float = DEFAULT_RESAMPLE_NOISE_TIME,
  quantile_thresh: float = DEFAULT_QUANTILE_THRESH,
  energy_scaling: float = DEFAULT_ENERGY_SCALING,
  use_metropolis: bool = False,
  effective_temp_scaling: float = DEFAULT_EFFECTIVE_TEMP_SCALING,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> tuple[Coords_batch, Energy_batch]:
  """Multi-round structure-prediction pipeline (backlog node **E10**).

  Python-level loop over ``num_rounds`` (module docstring: not a
  ``lax.scan``, since the between-round resampling step is inherently
  data-dependent-shape / non-jittable). Each round: ``jax.vmap`` the
  already-committed :func:`aminx.ebm.langevin_schedule.run_annealing_schedule`
  over ``batch_size`` independent trajectories/keys, score the resulting
  batch with the round's terminal-``t`` model's ``.energy()`` (via
  :func:`aminx.ebm.langevin_schedule.select_model_for_t` on
  ``noise_schedule``'s *last* value -- the schedule's lowest ``t``, where the
  round's descent ends), then (if not the final round) call
  :func:`resample_ensemble` to build the next round's starting batch.

  **Round 0's starting batch** is the *same* ``initial_coords`` structure,
  broadcast across ``batch_size`` and forward-noised to ``noise_schedule[0]``
  (the schedule's *first*/highest ``t``) via ``jax.vmap(forward_marginal)``
  over independent per-trajectory keys -- see module docstring's "Round 0
  starting-batch construction" section for exactly what this does and does
  not reproduce from the reference.

  **AF2Rank rescoring and scipy hierarchical-clustering resampling are both
  explicitly out of scope** -- see module docstring.

  PRNG discipline: ``key`` is split once for the round-0 initial-noising
  step (into ``batch_size`` independent subkeys, one per trajectory), then
  once per round thereafter into ``round_key`` (further split into
  ``batch_size`` independent subkeys, one per ``run_annealing_schedule``
  call) and, for every round but the last, a ``resample_key`` handed to
  :func:`resample_ensemble` (which does its own further splitting -- see
  that function's docstring). Every subkey is used exactly once.

  Args:
      models: Pre-loaded model instances, ascending t-range order (see
        :func:`aminx.ebm.langevin_schedule.select_model_for_t`); reused
        unchanged across every round (module docstring scope limit 1).
      thresholds: ``len(models) - 1`` ascending t-boundaries.
      initial_coords: The single starting structure, ``(N, 3)``,
        raw/physical (pre-``coordinate_scaling``) -- this function performs
        the one-time jump-to-``t_max`` noising itself (E9's own module
        docstring finding 3: that jump is the caller's responsibility, not
        ``run_annealing_schedule``'s).
      aatype: Per-residue amino-acid type, ``(N,)``, held fixed across every
        round and trajectory.
      mask: Residue validity mask, ``(N,)``, held fixed across every round
        and trajectory.
      noise_schedule: Per-level ``t`` values, ``(L,)``, typically descending
        from ``t_max`` (index 0) to ``t_min`` (index ``L-1``) -- see
        :func:`aminx.ebm.langevin_schedule.run_annealing_schedule`'s own
        docstring. This function reads ``noise_schedule[0]`` (round-0 initial
        noise level) and ``noise_schedule[-1]`` (terminal scoring ``t``)
        directly; it does not itself construct or validate the schedule.
      n_steps_per_level: Forwarded unchanged to every round's
        ``run_annealing_schedule`` call (see that function's own contract).
      dt: Euler-Maruyama step size, forwarded unchanged to every round.
      key: PRNG key for the whole pipeline (see PRNG discipline above).
      num_rounds: Number of resampling rounds (reference's
        ``args.num_resample_rounds``; design spec's "3x refine"). Must be
        ``>= 1``.
      batch_size: Number of independent trajectories per round (reference's
        ``bsize``) -- held constant across rounds in this port (the
        reference allows a distinct ``resample_batch_size`` for rounds 1+;
        not reproduced here, a further documented simplification).
      resample_noise_time: Forwarded to :func:`resample_ensemble`.
      quantile_thresh: Forwarded to :func:`resample_ensemble`.
      energy_scaling: Forwarded to :func:`resample_ensemble`.
      use_metropolis: Forwarded to every round's ``run_annealing_schedule``
        call.
      effective_temp_scaling: Forwarded to every round's
        ``run_annealing_schedule`` call and to the terminal energy scoring.
      coordinate_scaling: Forwarded to every round's ``run_annealing_schedule``
        call and to the round-0 initial noising step.

  Returns:
      ``(final_coords, final_energies)`` -- the *last* round's full ensemble,
      ``(batch_size, N, 3)`` and ``(batch_size,)`` respectively. Selecting a
      single "best" structure (e.g. ``jnp.argmin`` over ``final_energies``)
      is a caller-side concern, not this function's.

  Raises:
      ValueError: If ``num_rounds < 1``.

  """
  if num_rounds < 1:
    msg = f"run_structure_prediction: num_rounds must be >= 1, got {num_rounds}."
    raise ValueError(msg)

  noise_schedule_arr = jnp.asarray(noise_schedule)
  t_max = noise_schedule_arr[0]
  terminal_t = noise_schedule_arr[-1]

  key, key_init = jax.random.split(key)
  init_keys = jax.random.split(key_init, batch_size)
  coords_tiled = jnp.broadcast_to(initial_coords, (batch_size, *initial_coords.shape))

  def _initial_noise(coords: Coords, noise_key: PRNGKeyArray) -> Coords:
    x_t, _score = forward_marginal(
      coords,
      t_max,
      noise_key,
      coordinate_scaling=coordinate_scaling,
    )
    return x_t

  current_coords = jax.vmap(_initial_noise)(coords_tiled, init_keys)

  final_coords: Coords_batch = current_coords
  final_energies: Energy_batch

  for round_idx in range(num_rounds):
    key, round_key = jax.random.split(key)
    round_keys = jax.random.split(round_key, batch_size)

    if round_idx == 0:
      round_schedule, round_n_steps = noise_schedule_arr, n_steps_per_level
    else:
      # Rounds 1+ start from structures renoised only up to
      # `resample_noise_time`, not the original `t_max` -- see
      # `_schedule_for_round`'s docstring for why the schedule must be
      # truncated accordingly (a real t-mislabeling bug otherwise, not a
      # style choice).
      round_schedule, round_n_steps = _schedule_for_round(
        noise_schedule_arr,
        n_steps_per_level,
        resample_noise_time,
      )

    def _run_one_trajectory(coords: Coords, traj_key: PRNGKeyArray) -> Coords:
      return run_annealing_schedule(
        models,
        thresholds,
        coords,
        aatype,
        mask,
        round_schedule,  # noqa: B023
        round_n_steps,  # noqa: B023
        dt,
        traj_key,
        use_metropolis=use_metropolis,
        effective_temp_scaling=effective_temp_scaling,
        coordinate_scaling=coordinate_scaling,
      )

    round_coords = jax.vmap(_run_one_trajectory)(current_coords, round_keys)

    terminal_model = select_model_for_t(models, thresholds, terminal_t)

    def _score_one(coords: Coords) -> Energy:
      return terminal_model.energy(coords, aatype, terminal_t, mask)  # noqa: B023

    round_energies = jax.vmap(_score_one)(round_coords)

    final_coords, final_energies = round_coords, round_energies

    if round_idx < num_rounds - 1:
      key, resample_key = jax.random.split(key)
      current_coords = resample_ensemble(
        round_coords,
        round_energies,
        resample_key,
        batch_size=batch_size,
        quantile_thresh=quantile_thresh,
        energy_scaling=energy_scaling,
        resample_noise_time=resample_noise_time,
      )

  return final_coords, final_energies


__all__ = [
  "DEFAULT_ENERGY_SCALING",
  "DEFAULT_QUANTILE_THRESH",
  "DEFAULT_RESAMPLE_NOISE_TIME",
  "resample_ensemble",
  "run_structure_prediction",
]
