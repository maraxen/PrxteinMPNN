"""Decoy-ranking application logic for ProteinEBM (backlog node **E5**).

Builds the noise-time (``t``) sweep + Spearman-correlation evaluation on top
of ``aminx.ebm.dispatch.score_decoy_batch`` (backlog node **E4**'s axis-
dispatch wiring primitive -- see that module's docstring). Design spec
``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §1/§4/§8.2
and the EPIC DAG (``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md``
§2, E5 row): decoy-ranking performance is reported *t*-dependent, peaking
near ``t=0.05`` rather than at ``t=0`` -- a real decoy-ranking harness must
sweep ``t``, not hardcode it, and must compute Spearman(E, quality) against a
ground-truth structural-quality label.

**This module does NOT touch ``aminx.ebm.dispatch``** (E4's file, shared with
E6/E7 running in parallel) -- it only imports ``score_decoy_batch`` from it.

**Scope boundary (read before citing any number this module produces).**
This module implements the *real* application logic (sweep + correlation).
It does **not** ship the Rosetta decoy benchmark itself: the actual decoy
set (133 native structures + thousands of decoys, TM-scores, Rosetta
energies) referenced by the design spec's Spearman **0.838** parity target
lives behind ``wget https://files.ipd.uw.edu/pub/decoyset/decoys.zip`` (size
unknown, likely large) plus ``huggingface.co/jproney/ProteinEBM/resolve/
main/{rmsd,rosettascore}.txt`` -- **not downloaded in this environment, and
downloading it was explicitly out of scope / not authorized for this
dispatch** (unlike E3.5's checkpoint port, which the user did authorize).
Do not treat any Spearman number produced by this module's own tests as a
measurement of the paper's 0.838 target -- see
``tests/ebm/test_decoy_ranking.py`` for the honest small-scale proxy
validation this module is actually exercised against (synthetic coordinate
noise on real local PDB fixtures, RMSD-to-native as the quality label, real
ported checkpoint weights), and its module docstring for the full honesty
disclosure.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from scipy.stats import spearmanr

from aminx.ebm.dispatch import score_decoy_batch

if TYPE_CHECKING:
  from aminx.ebm.contracts import AAType, Coords, ResidueMask
  from aminx.ebm.model import ProteinEBMModel
  from aminx.ebm.plan import EnergyVector

QualityLabels = Float[Array, "D"]
"""Per-decoy ground-truth quality/degradation label vector (``D`` = decoy count).

Defined as a plain module-level assignment (not an inline annotation) so its
single-token jaxtyping shape string survives ``ruff``'s ``UP037`` "redundant
quoted annotation" fixer: under ``from __future__ import annotations``,
``ruff`` treats a bare single-identifier string *inside a function
annotation's subscript* (e.g. ``Float[Array, "D"]`` written directly as a
parameter type) as a redundant forward-reference quote and silently strips
it -- which is correct for an actual forward reference, but wrong here,
since jaxtyping needs ``"D"`` to remain a real string at runtime to parse
the shape spec, not a bare (nonexistent) name ``D``. A plain assignment
(mirrors ``aminx.ebm.contracts``'s ``Coords``/``Energy``/etc. and
``aminx.ebm.plan``'s ``EnergyVector``) is evaluated eagerly, outside any
annotation context, so ``UP037`` does not touch it.
"""

DEFAULT_NOISE_TIME_GRID: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2, 0.4)
"""Default ``t``-sweep grid for decoy ranking.

Design spec §4/§9 decision 2: the paper's MVP parity scope is
"ProteinEBM-x @ t=0.05", but decoy-ranking Spearman is itself reported as a
function of ``t`` that peaks near ``0.05``, not at the diffusion-time
boundary ``t=0``. This grid brackets that reported peak with two lower and
two higher values rather than assuming the peak location holds for every
structure/checkpoint.
"""


def sweep_noise_time_energies(
  model: ProteinEBMModel,
  coords: Float[Array, "D N 3"],
  aatype: AAType,
  mask: ResidueMask,
  t_values: tuple[float, ...] = DEFAULT_NOISE_TIME_GRID,
  *,
  default_batch_size: int | None = None,
) -> Float[Array, "T D"]:
  """Score every decoy in ``coords`` at every noise time in ``t_values``.

  A plain Python loop over ``t_values``, each iteration a single call to
  ``aminx.ebm.dispatch.score_decoy_batch`` (which itself picks Vmap/SafeMap
  over the decoy axis via ``aminx.ebm.plan.plan_axis``/``dispatch_axis``).
  This is deliberately **not** a second vmapped axis over ``t``: E4's
  dispatch registry (``aminx.ebm.plan.EBMAxisNames``) is single-axis only by
  design (see that module's docstring), and a handful of noise times (the
  paper's own ablations sweep a similarly small grid) is cheap enough as an
  outer Python loop -- no new tiling primitive is needed for this node.

  Args:
      model: The assembled ProteinEBM-equivalent model. For any
          parity-relevant use this must be the checkpoint-ported model
          (``aminx.ebm.checkpoint.load_pytorch_checkpoint``), not an
          untrained/randomly-initialized instance.
      coords: Stacked decoy coordinates (already ``coordinate_scaling``-
          scaled, per ``aminx.ebm.contracts.Coords``), leading axis = decoy
          count ``D``.
      aatype: Fixed sequence, shared across all decoys.
      mask: Fixed residue mask, shared across all decoys.
      t_values: Diffusion times to sweep. Defaults to
          :data:`DEFAULT_NOISE_TIME_GRID`.
      default_batch_size: Forwarded to ``score_decoy_batch``'s Vmap/SafeMap
          cutoff override.

  Returns:
      Energy matrix, shape ``(T, D)`` where ``T = len(t_values)`` -- row
      ``i`` is ``score_decoy_batch(model, coords, aatype, t_values[i],
      mask)``.

  """
  rows = [
    score_decoy_batch(
      model,
      coords,
      aatype,
      jnp.asarray(t),
      mask,
      default_batch_size=default_batch_size,
    )
    for t in t_values
  ]
  return jnp.stack(rows, axis=0)


def rmsd_to_reference(
  decoy_coords: Float[Array, "D N 3"],
  reference_coords: Coords,
  mask: ResidueMask | None = None,
) -> QualityLabels:
  """Per-decoy RMSD to a single reference structure, no re-superposition.

  A generic structural-quality proxy for a decoy set already sharing one
  coordinate frame (e.g. additively-perturbed copies of one native
  structure, or a real decoy set already superposed onto its native, as
  Rosetta decoy sets are). **This is not TM-score** -- the design spec's
  actual pinned decoy-ranking metric (§4.2/§8.2) is ``Spearman(E,
  TMScore)``; RMSD-to-native is this node's honest, dependency-light
  substitute for a small-scale pipeline-validation run (see this module's
  docstring's scope boundary and ``tests/ebm/test_decoy_ranking.py``).
  Adding a real structural aligner (e.g. US-align/TM-align) to compute an
  actual TM-score was out of scope for this dispatch.

  Args:
      decoy_coords: Stacked decoy coordinates, leading axis = decoy count.
      reference_coords: The single reference/native structure, ``(N, 3)``,
          same coordinate frame/units as ``decoy_coords``.
      mask: Optional residue mask (``True`` = include in the RMSD sum). If
          omitted, every residue contributes.

  Returns:
      Per-decoy RMSD, shape ``(D,)``, in whatever length units the inputs
      are already in (no internal rescaling).

  """
  diff = decoy_coords - reference_coords[None, :, :]
  squared_dev = jnp.sum(diff**2, axis=-1)  # (D, N)
  if mask is not None:
    weights = mask.astype(squared_dev.dtype)
    denom = jnp.maximum(jnp.sum(weights), 1.0)
    mean_squared_dev = jnp.sum(squared_dev * weights[None, :], axis=-1) / denom
  else:
    mean_squared_dev = jnp.mean(squared_dev, axis=-1)
  return jnp.sqrt(mean_squared_dev)


def spearman_energy_quality_correlation(
  energies: EnergyVector | np.ndarray,
  quality_labels: QualityLabels | np.ndarray,
) -> float:
  """Spearman rank correlation between per-decoy energy and a quality label.

  Thin wrapper over ``scipy.stats.spearmanr`` (``scipy`` is already an aminx
  dev dependency -- see ``pyproject.toml``'s "scipy.stats.pearsonr used in
  soluble/membrane parity tests" comment) -- the design spec's headline
  decoy-ranking metric is literally ``Spearman(E, TMScore)`` (§4.2/§8.2), so
  this is not reimplemented from scratch.

  **Sign convention -- read before interpreting the return value.** This
  function does **no** sign-flipping or absolute-valuing: it returns exactly
  what ``scipy.stats.spearmanr`` computes for ``(energies,
  quality_labels)`` in the order given. The paper's parity target (0.838) is
  reported as a positive ``Spearman(E, TMScore)`` where higher TM-score =
  more native-like; if lower energy genuinely means "more native-like" (the
  physically expected direction for a learned potential), the correlation
  between raw energy and a "higher = better" quality axis should be
  *negative*, and the paper's positive-sounding figure is consistent with an
  implicit absolute-value convention (common in this literature) or a
  quality axis oriented the other way. Callers own the orientation of
  ``quality_labels`` and the interpretation of the sign they get back --
  this module does not decide it for you. See
  ``rank_decoys_over_noise_time``'s "best" selection (max **absolute**
  correlation) for how this ambiguity is handled without silently forcing a
  sign.

  Args:
      energies: Per-decoy energies, shape ``(D,)``.
      quality_labels: Per-decoy ground-truth quality/degradation labels,
          shape ``(D,)``, same decoy ordering as ``energies``.

  Returns:
      The Spearman rank-correlation coefficient, in ``[-1, 1]``.

  """
  energies_np = np.asarray(energies)
  quality_np = np.asarray(quality_labels)
  result = spearmanr(energies_np, quality_np)
  return float(result.statistic)


@dataclasses.dataclass(frozen=True)
class NoiseTimeSweepResult:
  """Itemized result of a noise-time-swept decoy-ranking Spearman evaluation.

  Plain Python metadata (not a JAX pytree) -- mirrors
  ``aminx.ebm.checkpoint.CheckpointPortReport``'s reporting-only role, so a
  caller (a test, a notebook, a future bathos sidecar) gets the full sweep,
  not just the single "best" number.
  """

  t_values: tuple[float, ...]
  """The ``t`` grid this result was computed over."""

  spearman_by_t: tuple[float, ...]
  """``Spearman(energy_at_t, quality_labels)`` for each ``t`` in ``t_values``, same order."""

  best_t: float
  """The ``t`` in ``t_values`` with the largest-magnitude correlation."""

  best_spearman: float
  """``spearman_by_t`` at ``best_t`` (signed -- see
  :func:`spearman_energy_quality_correlation`'s sign-convention note)."""

  energies: np.ndarray
  """Full ``(T, D)`` energy matrix, for downstream inspection/plotting."""

  quality_labels: np.ndarray
  """The ``(D,)`` quality labels this result was evaluated against."""


def rank_decoys_over_noise_time(
  model: ProteinEBMModel,
  coords: Float[Array, "D N 3"],
  aatype: AAType,
  mask: ResidueMask,
  quality_labels: QualityLabels | np.ndarray,
  t_values: tuple[float, ...] = DEFAULT_NOISE_TIME_GRID,
  *,
  default_batch_size: int | None = None,
) -> NoiseTimeSweepResult:
  """The E5 decoy-ranking application: noise-time sweep + Spearman correlation.

  Composes :func:`sweep_noise_time_energies` (the axis-dispatch-backed
  energy sweep) with :func:`spearman_energy_quality_correlation` (one
  correlation per ``t``), and reports the ``t`` with the strongest ranking
  signal. "Strongest" is measured by **maximum absolute correlation**, not
  signed maximum -- this mirrors the paper's framing of searching for the
  ``t`` that gives the best-discriminating energy landscape, independent of
  which sign convention ``quality_labels`` happens to use (see
  :func:`spearman_energy_quality_correlation`'s docstring).

  This is the real, complete E5 application logic (design spec §3.3 row 1 +
  §4/§9's ``t``-sweep requirement). It does **not** ship the actual Rosetta
  decoy benchmark or a TM-score computation -- see this module's docstring
  and ``tests/ebm/test_decoy_ranking.py`` for the honest proxy validation
  this function is exercised against, and the explicit scope-limitation
  disclosure (Rosetta ``decoys.zip``/rmsd/rosettascore download not
  authorized for this dispatch).

  Args:
      model: The assembled ProteinEBM-equivalent model (real ported
          checkpoint weights for any parity-relevant use).
      coords: Stacked decoy coordinates, leading axis = decoy count ``D``.
      aatype: Fixed sequence, shared across all decoys.
      mask: Fixed residue mask, shared across all decoys.
      quality_labels: Per-decoy ground-truth structural-quality labels
          (e.g. TM-score, or -- this node's honest proxy -- RMSD-to-native),
          shape ``(D,)``, same decoy ordering as ``coords``'s leading axis.
      t_values: Diffusion times to sweep. Defaults to
          :data:`DEFAULT_NOISE_TIME_GRID`.
      default_batch_size: Forwarded to ``score_decoy_batch``'s Vmap/SafeMap
          cutoff override.

  Returns:
      A :class:`NoiseTimeSweepResult` with the per-``t`` correlations and
      the best-``t`` summary.

  """
  energies = sweep_noise_time_energies(
    model,
    coords,
    aatype,
    mask,
    t_values,
    default_batch_size=default_batch_size,
  )
  spearman_by_t = tuple(
    spearman_energy_quality_correlation(energies[i], quality_labels) for i in range(len(t_values))
  )
  best_idx = int(np.argmax(np.abs(np.asarray(spearman_by_t))))
  return NoiseTimeSweepResult(
    t_values=tuple(t_values),
    spearman_by_t=spearman_by_t,
    best_t=t_values[best_idx],
    best_spearman=spearman_by_t[best_idx],
    energies=np.asarray(energies),
    quality_labels=np.asarray(quality_labels),
  )


__all__ = [
  "DEFAULT_NOISE_TIME_GRID",
  "NoiseTimeSweepResult",
  "rank_decoys_over_noise_time",
  "rmsd_to_reference",
  "spearman_energy_quality_correlation",
  "sweep_noise_time_energies",
]
