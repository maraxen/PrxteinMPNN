"""Jaxtyping contracts for the ProteinEBM energy/score path (``aminx.ebm``).

These type aliases pin the array shapes/dtypes that flow through the VP-SDE
diffusion + energy-readout path (EPIC ``260709_aminxtension``, backlog node
**E0** -- see ``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §2
and the design spec ``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md``
§1). They are intentionally separate from ``aminx.types.arrays`` (the
inverse-folding logit path): ProteinEBM is a *peer* forward/generative energy
axis, not a variant of the logit ``StageSet``, so the two contract families
stay decoupled until backlog node E4 composes them under xtrax.
"""

from __future__ import annotations

from jaxtyping import Array, Bool, Float, Int

Coords = Float[Array, "N 3"]
"""Per-residue CA coordinates, in nanometres.

This is the *scaled* representation used throughout ``aminx.ebm``: the single
``coordinate_scaling`` boundary lives in
:func:`aminx.ebm.diffusion.forward_marginal` (applied exactly once -- see its
docstring). Every other function in this subpackage consumes/produces
coordinates already in this scaled space; re-applying ``coordinate_scaling``
downstream is the reference-code footgun this contract exists to prevent
(spec §10, MINOR finding on ``coordinate_scaling`` double-application).
"""

Atom37 = Float[Array, "N 37 3"]
"""All-atom auxiliary target: CA-centered heavy-atom coordinates, atom index 1 = CA.

Not diffused (``diffuse_sidechain = False`` in both shipped configs); used
only as auxiliary supervision (``L_atom``) in E8 training. Out of scope for
E0-E7 (spec §1 Fork 5: all-atom aux head dropped for E3-E7, included for E8).
"""

Energy = Float[Array, ""]
"""Scalar total energy, ``E = sum_i mask_i * ||r_i||**2`` (masked sum over residues).

Non-negative by construction (sum of squared norms). See
``aminx.ebm.contracts.PerResidueEnergy`` for the pre-reduction per-residue
term.
"""

PerResidueEnergy = Float[Array, "N"]
"""Per-residue energy contribution, ``mask_i * ||r_i||**2``, prior to the sum reduction."""

Score = Float[Array, "N 3"]
"""Coordinate score/force, ``-grad_x E`` (conservative) or the aux-head surrogate.

Same shape/units as ``Coords``: a score is a vector field over the diffused
coordinate space, not a separate physical quantity.
"""

DiffusionTime = Float[Array, ""]
"""Continuous VP-SDE diffusion time, ``t in [0, 1]``. A tunable inference knob, not fixed at 0."""

AAType = Int[Array, "N"]
"""Amino-acid type index per residue, in ``[0, 20]`` (21-way embedding incl. mask token 20)."""

ResidueMask = Bool[Array, "N"]
"""``True`` where a residue is present/valid; ``False`` for padding (bucketed residue axis)."""

__all__ = [
  "AAType",
  "Atom37",
  "Coords",
  "DiffusionTime",
  "Energy",
  "PerResidueEnergy",
  "ResidueMask",
  "Score",
]
