"""Pure VP-SDE math for the ProteinEBM CA-coordinate diffusion process.

This module is the JAX port of ``protein_ebm.model.r3_diffuser.R3Diffuser``
(``~/repos/ProteinEBM/protein_ebm/model/r3_diffuser.py``) -- the **math only**,
not the learned model (backlog node **E0**; see
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §2 and design
spec §1). ``EnergyReadout``/``ScoreReadout``/the transformer trunk land in
backlog nodes E1-E3.

**This is a variance-preserving (VP) SDE, not variance-exploding (VE).** The
noise schedule is ``b(t) = min_b + t*(max_b - min_b)``; the integrated
schedule ``marginal_b_t(t) = t*min_b + 0.5*t**2*(max_b - min_b)`` gives the
closed-form marginal ``p(x_t | x_0) = N(exp(-0.5*marginal_b_t(t)) * x_0,
conditional_var(t) * I)`` with ``conditional_var(t) = 1 - exp(-marginal_b_t(t))``.

**``coordinate_scaling`` is applied exactly once**, inside
:func:`forward_marginal`, converting the raw/physical ``x0`` input into the
scaled coordinate space (nanometres; see ``aminx.ebm.contracts.Coords``) used
by every other function in this module and, downstream, by the energy
readout. :func:`score_target` and :func:`calc_trans_0` never re-apply it --
they are unit-parametric closed-form math over whatever consistent
coordinate space their ``x_t``/``x0`` arguments already live in. This
resolves the reference implementation's footgun of applying
``coordinate_scaling`` at three separate call sites (the diffuser, ``ebm.py``
``forward``'s ``rescale_input_coords``, and ``precondition``) -- design spec
§10, MINOR finding.

PRNG discipline: every function that draws randomness (:func:`forward_marginal`)
takes a single ``key`` and consumes it exactly once (one Gaussian draw over
the full coordinate array). Callers must split before reuse -- this module
never reuses or internally re-splits a key.

All functions are pure, jittable, and vmappable over a batch of ``t`` (and,
for :func:`forward_marginal`, a batch of ``key``) -- see
``tests/ebm/test_diffusion_invariants.py`` invariant (d).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from jaxtyping import PRNGKeyArray

  from aminx.ebm.contracts import Coords, DiffusionTime, Score

# NaN-safety epsilon for the `conditional_var` reciprocal near t=0, where
# `conditional_var(0) == 0` exactly and `score_target` would otherwise divide
# by zero. Chosen small enough to leave `score_target`/`forward_marginal`
# numerically unchanged away from t=0 (see the invariant tests' tolerance).
_VAR_EPS = 1e-6

DEFAULT_MIN_B = 0.1
DEFAULT_MAX_B = 20.0
DEFAULT_COORDINATE_SCALING = 0.1


def beta_t(
  t: DiffusionTime,
  min_b: float = DEFAULT_MIN_B,
  max_b: float = DEFAULT_MAX_B,
) -> DiffusionTime:
  """Instantaneous VP-SDE noise-schedule coefficient, ``b(t) = min_b + t*(max_b - min_b)``."""
  return min_b + t * (max_b - min_b)


def marginal_b_t(
  t: DiffusionTime,
  min_b: float = DEFAULT_MIN_B,
  max_b: float = DEFAULT_MAX_B,
) -> DiffusionTime:
  """Integrated noise schedule, ``t*min_b + 0.5*t**2*(max_b - min_b)``.

  Equivalent to ``-log(alpha_bar_t)`` in DDPM notation; matches
  ``R3Diffuser.marginal_b_t`` in the reference.
  """
  return t * min_b + 0.5 * (t**2) * (max_b - min_b)


def conditional_var(
  t: DiffusionTime,
  min_b: float = DEFAULT_MIN_B,
  max_b: float = DEFAULT_MAX_B,
) -> DiffusionTime:
  """Marginal variance ``Var[x_t | x_0] = 1 - exp(-marginal_b_t(t))`` (VP-SDE, not VE)."""
  return 1.0 - jnp.exp(-marginal_b_t(t, min_b=min_b, max_b=max_b))


def score_target(
  x_t: Coords,
  x0: Coords,
  t: DiffusionTime,
  min_b: float = DEFAULT_MIN_B,
  max_b: float = DEFAULT_MAX_B,
) -> Score:
  """Closed-form denoising-score-matching target.

  ``-(x_t - mean) / var`` where ``mean = exp(-0.5*marginal_b_t(t)) * x0`` and
  ``var = conditional_var(t)`` (epsilon-guarded via ``_VAR_EPS`` so the
  reciprocal is finite at ``t=0``, where ``var == 0`` exactly).

  ``x_t`` and ``x0`` must already be in the same coordinate space -- this
  function performs no ``coordinate_scaling`` (see module docstring; that
  boundary is :func:`forward_marginal`, applied exactly once).
  """
  mbt = marginal_b_t(t, min_b=min_b, max_b=max_b)
  mean = jnp.exp(-0.5 * mbt) * x0
  var = conditional_var(t, min_b=min_b, max_b=max_b)
  safe_var = jnp.maximum(var, _VAR_EPS)
  return -(x_t - mean) / safe_var


def calc_trans_0(
  score: Score,
  x_t: Coords,
  t: DiffusionTime,
  min_b: float = DEFAULT_MIN_B,
  max_b: float = DEFAULT_MAX_B,
) -> Coords:
  """Invert :func:`score_target` to recover ``x0`` from ``(score, x_t, t)``.

  ``x0 = (score * var + x_t) / exp(-0.5*marginal_b_t(t))`` -- the exact
  algebraic inverse (matches ``R3Diffuser.calc_trans_0``). Operates in the
  same coordinate space as its inputs; no ``coordinate_scaling``.

  Note: this uses the *unguarded* ``conditional_var(t)`` (not the
  ``_VAR_EPS``-clamped value :func:`score_target` used internally), so exact
  recovery of a ``score_target``-produced score holds away from ``t=0``; the
  ``t=0`` singularity is inherent to the closed form (``var == 0`` there) and
  is a NaN-safety concern for :func:`score_target`'s reciprocal, not this
  function's.
  """
  mbt = marginal_b_t(t, min_b=min_b, max_b=max_b)
  var = conditional_var(t, min_b=min_b, max_b=max_b)
  return (score * var + x_t) / jnp.exp(-0.5 * mbt)


def forward_marginal(
  x0: Coords,
  t: DiffusionTime,
  key: PRNGKeyArray,
  min_b: float = DEFAULT_MIN_B,
  max_b: float = DEFAULT_MAX_B,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> tuple[Coords, Score]:
  """Sample ``x_t ~ p(x_t | x0)`` and its closed-form DSM score target.

  ``coordinate_scaling`` is applied **exactly once, here** -- the sole
  transition from raw/physical input coordinates into the scaled
  representation (``aminx.ebm.contracts.Coords``, nanometres) used throughout
  the rest of the energy path. See the module docstring for why this single
  boundary matters.

  PRNG discipline: ``key`` is consumed exactly once (one Gaussian draw over
  the full ``x0`` shape); callers must split before reuse.

  Args:
    x0: Raw/physical input coordinates (pre-``coordinate_scaling``).
    t: Continuous diffusion time in ``[0, 1]``.
    key: PRNG key for the single Gaussian draw. Not reused internally.
    min_b: VP-SDE schedule lower bound.
    max_b: VP-SDE schedule upper bound.
    coordinate_scaling: Applied once to ``x0`` before diffusing.

  Returns:
    ``(x_t, score)`` -- both in the scaled coordinate space.
  """
  x0_scaled = x0 * coordinate_scaling
  mbt = marginal_b_t(t, min_b=min_b, max_b=max_b)
  mean = jnp.exp(-0.5 * mbt) * x0_scaled
  var = conditional_var(t, min_b=min_b, max_b=max_b)
  noise = jax.random.normal(key, shape=x0.shape, dtype=x0.dtype)
  x_t = mean + jnp.sqrt(var) * noise
  score = score_target(x_t, x0_scaled, t, min_b=min_b, max_b=max_b)
  return x_t, score


__all__ = [
  "DEFAULT_COORDINATE_SCALING",
  "DEFAULT_MAX_B",
  "DEFAULT_MIN_B",
  "beta_t",
  "calc_trans_0",
  "conditional_var",
  "forward_marginal",
  "marginal_b_t",
  "score_target",
]
