"""Three disjoint ProteinEBM dispatch entry points wired onto xtrax tiling (backlog node **E4**).

Per the EPIC DAG's explicit rationale (`.praxia/docs/plans/260709_proteinebm-
epic-backlog-dag.md` §1 decision 9, §2 E4 row): E5 (decoy ranking), E6 (ΔΔG
stability), and E7 (conformational biasing) are separate, later backlog nodes
that will each build out their own accuracy-gated application logic. If E4
pre-stubbed a single shared ``_score_batch``-style function the way
``host/kernel_dispatch.py::_sample_batch`` centralizes the inverse-folding
sampling path, three concurrent E5/E6/E7 subagents would merge-conflict
editing the same function. Instead, E4 lands three genuinely separate
functions in this one file (thin/stub-level bodies — the *wiring contract*,
not the accuracy-gated application logic) so E5/E6/E7 each own one function
outright.

Each function:

* Takes a :class:`~aminx.ebm.model.ProteinEBMModel` and per-element inputs.
* Builds a single-axis :class:`~xtrax.tiling.AxisDecision` via
  ``aminx.ebm.plan.plan_axis`` (BatchPlanner's cardinality-vs-batch-size rule
  picks ``Vmap`` for small N, ``SafeMap`` for large N).
* Dispatches ``ProteinEBMModel.energy`` over that axis via
  ``aminx.ebm.plan.dispatch_axis``.
* (mutant/state variants) reduces the per-element results with one of the
  ``EnergyFusionFn`` primitives from ``aminx.ebm.plan``.

**No PRNG key is threaded anywhere in this file.**
``ProteinEBMModel.energy``/``.score``/``.aux_score`` take no ``key`` argument
— energy/score are deterministic given ``coords``/``aatype``/``t``/``mask``
(design spec §1: only the *training* path and the deferred Langevin sampler
need randomness). The DAG's Fork 10 ``jax.random.fold_in(base_key, idx)``
per-element-key guidance is reserved for the day a dispatch entry point here
actually needs one (e.g. a future stochastic augmentation); inventing one now
would be dead code, not "consistency with the pattern".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aminx.ebm.plan import (
  EBMAxisNames,
  EnergyFusionFn,
  EnergyVector,
  dispatch_axis,
  plan_axis,
)
from aminx.ebm.plan import difference_fuse as _default_difference_fuse
from aminx.ebm.plan import mean_fuse as _default_mean_fuse

if TYPE_CHECKING:
  from jaxtyping import Array, Float, Int

  from aminx.ebm.contracts import AAType, Coords, DiffusionTime, Energy, ResidueMask
  from aminx.ebm.model import ProteinEBMModel


def score_decoy_batch(
  model: ProteinEBMModel,
  coords: Float[Array, "D N 3"],
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  *,
  contacts: Array | None = None,
  default_batch_size: int | None = None,
) -> EnergyVector:
  """Score ``D`` decoy backbones of one fixed sequence at one fixed noise time.

  Wiring primitive for **E5** (decoy ranking / structure QA, design spec §3.3
  row 1): each decoy has distinct coordinates but shares ``aatype``/``t``/
  ``mask``, so this dispatches ``model.energy`` over the leading decoy axis of
  ``coords`` via Vmap (small ``D``) or SafeMap (large ``D``), per
  ``BatchPlanner``'s own cardinality-vs-batch-size rule — no bespoke loop.
  E5 will extend this with the noise-time sweep axis and the real Rosetta-
  decoy scoring harness; this function is the axis-dispatch wiring only.

  Args:
      model: The assembled ProteinEBM-equivalent model.
      coords: Stacked decoy coordinates, leading axis = decoy count ``D``.
      aatype: Fixed sequence, shared across all decoys.
      t: Fixed diffusion time, shared across all decoys.
      mask: Fixed residue mask, shared across all decoys.
      default_batch_size: Override the ``n_decoys`` axis's Vmap/SafeMap
          cutoff (default from ``aminx.ebm.plan``'s axis template).

  Returns:
      Per-decoy energy, shape ``(D,)``.

  """
  n_decoys = coords.shape[0]
  decision = plan_axis(EBMAxisNames.N_DECOYS, n_decoys, default_batch_size=default_batch_size)

  def _score_one(c: Coords) -> Energy:
    return model.energy(c, aatype, t, mask, contacts=contacts)

  return dispatch_axis(decision.strategy, _score_one, coords)


def score_mutant_ensemble(
  model: ProteinEBMModel,
  coords: Coords,
  mutant_aatype: Int[Array, "M N"],
  t: DiffusionTime,
  mask: ResidueMask,
  *,
  wildtype_aatype: AAType | None = None,
  unfolded_ensemble_aatype: Int[Array, "U N"] | None = None,
  mean_fuse: EnergyFusionFn = _default_mean_fuse,
  default_batch_size: int | None = None,
) -> EnergyVector:
  """Score ``M`` mutant sequences of one fixed structure, optionally as ΔΔG.

  Wiring primitive for **E6** (ΔΔG stability, design spec §3.3 row 2): one
  fixed backbone (``coords``/``t``/``mask``), ``M`` mutant sequences dispatched
  over the mutant axis via Vmap/SafeMap. If ``wildtype_aatype`` is given, the
  result is ``E(mutant) - E(wildtype)``; if ``unfolded_ensemble_aatype`` is
  *also* given (``U`` unfolded-state samples of the same backbone), the
  Monte-Carlo unfolded-state correction — the textbook mean-``Fuse`` over the
  ensemble axis (design spec §3.3 row 2) — is subtracted as well:
  ``ddG = E(mutant) - E(wildtype) - mean_fuse(E(unfolded_ensemble))``.
  E6 owns the real accuracy-gated ΔΔG protocol (state-exact split, dedup over
  shared WT references, etc.); this function is the axis-dispatch + mean-Fuse
  wiring only.

  Args:
      model: The assembled ProteinEBM-equivalent model.
      coords: Fixed backbone coordinates, shared across all mutants/WT/
          ensemble members.
      mutant_aatype: Stacked mutant sequences, leading axis = mutant count
          ``M``.
      t: Fixed diffusion time.
      mask: Fixed residue mask.
      wildtype_aatype: If given, subtract ``E(wildtype)`` from every mutant
          energy (turning the return value into a raw energy difference
          rather than an absolute per-mutant energy).
      unfolded_ensemble_aatype: If given (requires ``wildtype_aatype`` to be
          meaningful as a ΔΔG), stacked unfolded-ensemble sequences, leading
          axis = ensemble size ``U``; their mean energy is subtracted too.
      mean_fuse: The mean-``Fuse`` primitive applied to the unfolded-ensemble
          energies. Defaults to ``aminx.ebm.plan.mean_fuse``.
      default_batch_size: Override both the ``n_mutants`` and ``n_ensemble``
          axes' Vmap/SafeMap cutoff.

  Returns:
      Per-mutant energy (shape ``(M,)``) if neither correction is given;
      per-mutant ``E(mutant) - E(wildtype)`` if only ``wildtype_aatype`` is
      given; full ΔΔG (mutant minus wildtype minus unfolded-ensemble mean) if
      both are given.

  """
  n_mutants = mutant_aatype.shape[0]
  mutant_decision = plan_axis(
    EBMAxisNames.N_MUTANTS, n_mutants, default_batch_size=default_batch_size,
  )

  def _score_mutant(aatype_m: AAType) -> Energy:
    return model.energy(coords, aatype_m, t, mask)

  mutant_energies = dispatch_axis(mutant_decision.strategy, _score_mutant, mutant_aatype)

  if wildtype_aatype is None:
    return mutant_energies

  wildtype_energy = model.energy(coords, wildtype_aatype, t, mask)
  ddg = mutant_energies - wildtype_energy

  if unfolded_ensemble_aatype is None:
    return ddg

  n_ensemble = unfolded_ensemble_aatype.shape[0]
  ensemble_decision = plan_axis(
    EBMAxisNames.N_ENSEMBLE, n_ensemble, default_batch_size=default_batch_size,
  )

  def _score_ensemble_member(aatype_u: AAType) -> Energy:
    return model.energy(coords, aatype_u, t, mask)

  ensemble_energies = dispatch_axis(
    ensemble_decision.strategy, _score_ensemble_member, unfolded_ensemble_aatype,
  )
  unfolded_correction = mean_fuse(ensemble_energies)
  return ddg - unfolded_correction


def score_state_difference(
  model: ProteinEBMModel,
  coords_states: Float[Array, "S N 3"],
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  *,
  difference_fuse: EnergyFusionFn = _default_difference_fuse,
  default_batch_size: int | None = None,
) -> Energy:
  """Score the per-state energy gap for a fixed sequence across conformational states.

  Wiring primitive for **E7** (conformational biasing / multistate, design
  spec §3.3 row 3, §5): one fixed sequence, ``S`` conformational states
  (typically ``S=2``, e.g. LplA open/closed), dispatched over the state axis
  via Vmap (states are few, always Vmap-eligible at the default batch size),
  then reduced with the difference-``Fuse`` primitive into a single scalar
  energy gap. E7 owns the real biasing/multistate objective (units of
  ``k_BT``, sign conventions vs. activity, information-ablation probes); this
  function is the axis-dispatch + difference-Fuse wiring only.

  Args:
      model: The assembled ProteinEBM-equivalent model.
      coords_states: Stacked per-state coordinates, leading axis = state count
          ``S``.
      aatype: Fixed sequence, shared across all states.
      t: Fixed diffusion time.
      mask: Fixed residue mask.
      difference_fuse: The difference-``Fuse`` primitive reducing the ``S``
          per-state energies to one scalar. Defaults to
          ``aminx.ebm.plan.difference_fuse`` (``state[0] - state[1]``).
      default_batch_size: Override the ``n_states`` axis's Vmap/SafeMap
          cutoff.

  Returns:
      The fused (by default, ``E(state_0) - E(state_1)``) scalar energy gap.

  """
  n_states = coords_states.shape[0]
  decision = plan_axis(EBMAxisNames.N_STATES, n_states, default_batch_size=default_batch_size)

  def _score_one(c: Coords) -> Energy:
    return model.energy(c, aatype, t, mask)

  per_state_energies = dispatch_axis(decision.strategy, _score_one, coords_states)
  return difference_fuse(per_state_energies)


__all__ = ["score_decoy_batch", "score_mutant_ensemble", "score_state_difference"]
