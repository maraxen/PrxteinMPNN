# aminx — Mission

## What aminx is

**aminx** is a functional, JAX/Equinox interface to ProteinMPNN and LigandMPNN.
It exposes structure-conditioned protein sequence design — sampling, scoring, and
conditional/unconditional logit computation — through a composable Python API and an
`aminx` CLI driven by declarative spec files (`SamplingSpecification`,
`ScoringSpecification`, `JacobianSpecification`).

- **Language / stack:** Python 3.12+ · JAX + Equinox · `uv` · `ty` (strict) · `ruff` · `pytest`
- **Status:** Alpha (v0.1.0a1). API is functional and validated against the LigandMPNN
  reference, but may change between releases.
- **Owner:** Marielle Russo
- **Consumers:** the `aminx` CLI/API, the Potts recapture layer, and (refactor target)
  cross-repo reuse via **xtrax** as the batching/run substrate. `prolix` is consumer #2.

## Why it exists

Reference ProteinMPNN/LigandMPNN implementations are PyTorch-bound and imperative.
aminx makes the same models **functional and JIT-composable**: `jax.jit` / `jax.vmap` /
`lax.scan` patterns, Equinox `eqx.Module` models with `filter_jit`, declarative
inference plans, and cross-framework numerical parity. This unlocks differentiable
pipelines (Jacobians), vectorized batch design, and ahead-of-time export (StableHLO/IREE)
that the reference stack cannot offer.

## Current strategic focus — the xtrax refactor (EPIC, P0→P5)

Spec: `.praxia/docs/specs/260611_aminx-xtrax-refactor.md` (all TBDs resolved).
Refactor aminx's host/batching/run layer onto **xtrax** as a shared substrate, via a
vertical slice spine:

- **P0** — Foundations: bump `requires-python` `>=3.12`->`>=3.13` (T0.1), add xtrax
  editable pin under `[tool.uv.sources]` (T0.2), boundary-lint atomic with first xtrax
  import (T0.4). <- current entry point (unstarted).
- **P1** — Training adoption (ResumableState + optimizer + clean-break checkpoint).
- **P2** — xtrax cross-repo integration, gated by the R1 Definition-of-Done. Off-ramp
  documented (idea 2/22) if R1 DoD fails.
- **P3** — Tiling.
- **P4** — Inference-runner.
- **P5** — Cutover.

Resolved decisions: T-R1-X = 5% hard / 0% target · T-R2 = clean-break (no migration CLI)
· T-R3b = YES (prolix is consumer #2).

## Operating principles

- **Numerical truth first.** Verify any new metric/eval against synthetic ground truth
  before trusting research conclusions (BATHOS rule). Cross-framework parity tests gate
  model changes.
- **Strict typing & format.** `uv run ty check`, `uv run ruff check .`, `uv run ruff format .`,
  `uv run pytest` are the local gates. `jaxlint` is advisory, not a CI gate.
- **Reproducible experiments.** Scientific/in-silico work runs through **bathos** with
  pre-registered sidecars; cluster jobs go through **myxcel** (engaging) with the
  L1/L2/L3 local-gate ladder before submission.
- **Human-in-the-loop.** This loop runs semi-autonomously (`run.mode = hitl_semi_autonomous`);
  the PI is available for decision points and the L2 gates (spec_confirmed, sprint_approved).
- **Never autonomously push, merge to main, or delete.** Those remain manual.

## Definition of a good loop iteration

Close one executable backlog item end-to-end (recon -> plan -> fix -> audit) with tests
green and `ty`/`ruff` clean, leaving the tree committable and the next item unblocked —
or advance one epic-pipeline stage (research / brainstorm / design / spec-review /
register) for unspecced work.
