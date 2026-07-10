"""ProteinEBM composable energy/score path (EPIC ``260709_aminxtension``).

Foundational subpackage for aminx's forward, generative energy/score axis --
the JAX/Equinox port of ProteinEBM's VP-SDE-over-CA-coordinates energy path
(Roney, Ou, Ovchinnikov, *Protein Diffusion Models as Statistical
Potentials*, bioRxiv 2025.12.09.693073v3). This is a **peer** capability to
aminx's existing inverse-folding logit ``StageSet``, not a modification of
it -- see the design spec's core finding (§2) that ProteinEBM cannot be
expressed by the logit-shaped ``StageSet`` slots and must compose alongside
them under xtrax.

This subpackage stays import-isolated (no existing aminx file references it)
until backlog node E4 wires it into ``aminx.host``/``aminx.inference`` as a
composable stage bundle.

Contents (backlog node **E0** only -- see
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §2):
  - ``contracts``: jaxtyping array-shape/dtype contracts for the energy path
    (coords, atom37, energy, score, diffusion time, aatype, residue mask).
  - ``diffusion``: pure VP-SDE math -- noise schedule, forward marginal
    sampling, the closed-form denoising-score-matching target, and its
    algebraic inverse. Not the learned model: ``EnergyReadout``/
    ``ScoreReadout``/the transformer trunk land in backlog nodes E1-E3.
"""

from __future__ import annotations

from aminx.ebm import contracts, diffusion

__all__ = ["contracts", "diffusion"]
