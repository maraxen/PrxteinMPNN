# aminx Internal Docs

## Roadmaps
- [INDEX](roadmaps/INDEX.md) — all roadmap documents with status
- [260507_refactor-phases-0-6](roadmaps/260507_refactor-phases-0-6.md) — **DEPRECATED** phases 0–6 structural refactor (historical reference)
- [260508_active-roadmap](roadmaps/260508_active-roadmap.md) — **Active** — MODELINPUTS PR-4/5, EncoderPreFn/PostFn, multi_state_temperature

## Plans
- [260614_runspec-migration-map](plans/260614_runspec-migration-map.md) — RS-1 host-field inventory: 67 fields, 22 migrated, 16 to migrate, 9 RS-gaps, 21 protein-only
- [260522_comp-new-sink-unify](plans/260522_comp-new-sink-unify.md) — COMP-NEW: unify result-sink topology; streaming_tensor_sink_session for non-streaming path
- [260525_comp-unified-encoder-fusion](plans/260525_comp-unified-encoder-fusion.md) — COMP-UNIFIED: encoder fusion via InferencePlan; eliminate averaged-path branch
- [260527_sprint6-decode-axis-composability](superpowers/plans/260527_sprint6-decode-axis-composability.md) — Sprint 6 COMPLETE ✅: composable decode modes (ConditionalDecode/AR/STE), iterator injection, driver.py retired

## Specs
- [260618_rs7b-averaged-scoring-design](specs/260618_rs7b-averaged-scoring-design.md) — RS-7b averaged-topology scoring design (AC-RS-7)
- [260611_aminx-xtrax-refactor](specs/260611_aminx-xtrax-refactor.md) — **Active** — xtrax vertical-slice refactor (T0–T5), gates, prolix second-consumer API
- [260611_runspec-unification](specs/260611_runspec-unification.md) — **Active** — RunSpec + PlannerTopology addendum (RS track); blocks T4.1
- [260611_architecture-sequencing-testing-and-pack](specs/260611_architecture-sequencing-testing-and-pack.md) — brainstorm convergence record (composite winner)
- [260527_merge-readiness-hardening](specs/260527_merge-readiness-hardening.md) — ORACLE-REVIEWED PASS (15/18): pre-merge hygiene for refactor-full → main
- [260601_benchmark-spec](specs/260601_benchmark-spec.md) — ORACLE-APPROVED: GPU benchmark suite (aminx vs. LigandMPNN vs. ColabDesign)
- [260604_release-preparedness-epic](specs/260604_release-preparedness-epic-for-prxteinmpn.md) — Release Preparedness epic: two-phase backlog DAG (CI+cleanup → docs+release), Definition of Done, ADR index
- [260611_aminx-xtrax-refactor](specs/260611_aminx-xtrax-refactor.md) — **Challenger/Defender PASS**: refactor aminx onto xtrax (move tiling/sinks/inference-plan, rebuild training); vertical-slice spine + xtrax-tiling-upgrade precondition gate + branch-by-abstraction flag + dual perf-guard + off-ramp. Backlog EPIC #1541, sprint `260611_xtrax-foundations`

## Research
- [260611_aminx-xtrax-refactor-codebase-model](research/260611_aminx-xtrax-refactor-codebase-model.md) — Wave-1+2 recon synthesis + validated resolutions (A1–A8) underpinning the xtrax refactor spec
- [260616_axisspec-field-map](research/260616_axisspec-field-map.md) — R7-1 gate: canonical AxisSpec field names for RS-6 (default_batch_size, tile_granularity)
- [260616_noise-field-map](research/260616_noise-field-map.md) — R6-2 gate: 8 noise fields → FeatureNoiseBundle mapping table + dataclass design

## Archives
- [260604_example-notebook](archive/260604_example-notebook.md) — example_notebook.ipynb (API drift; pre-composable-inference)
- [260604_training-example-notebook](archive/260604_training-example-notebook.md) — training_example_notebook.ipynb (training module not ready; Sprint 3 blocker)
- [260124_proxide-integration-docs](archive/260124_proxide-integration-docs.md) — Proxide integration docs
- [260410_parity-validation-legacy](archive/260410_parity-validation-legacy.md) — Legacy parity validation docs
- [260512_docs-superpowers-status](archive/260512_docs-superpowers-status.md) — docs/superpowers/ skill outputs
- [260528_agent-scaffolding](archive/260528_agent-scaffolding.md) — .agent/ scaffolding
- [260528_agents-sprint-artifacts](archive/260528_agents-sprint-artifacts.md) — .agents/ sprint outputs

## Decisions

- [260605_potts-alphabet-alignment](decisions/260605_potts-alphabet-alignment.md) — Potts-MPNN alphabet comparison: indices 0-19 identical, index 20 (gap) semantically aligned. Identity permutation safe.
- [260605_potts-parallel-not-stageset](decisions/260605_potts-parallel-not-stageset.md) — **Accepted 2026-06-05**: PottsModel is a parallel architecture, NOT a StageSet consumer. Boundary enforced by lint #1304
- [260605_protein-features-shared-or-local](decisions/260605_protein-features-shared-or-local.md) — ProteinFeatures sourcing decision: mistypotts imports from aminx.model.features (with legacy prxteinmpnn vendored copy)
- [260612_proteinfeatures-shared-vs-local](decisions/260612_proteinfeatures-shared-vs-local.md) — **Accepted 2026-06-12**: Confirms Option A complete — aminx.potts imports from aminx.model.features; no vendor copy in potts tree. Supersedes adr/260605.

## ADRs (Legacy)

- [260604_defer-ty-ruff-ci-gates](adr/260604_defer-ty-ruff-ci-gates.md) — Defer `ty check` + `ruff check` as CI gates until code surface is stable
- [260604_parity-docs-as-ci-artifacts](adr/260604_parity-docs-as-ci-artifacts.md) — Parity validation docs are CI-autogenerated, not hand-maintained
- [260604_versioning-strategy](adr/260604_versioning-strategy.md) — Versioning strategy: 0.1.0a1 → beta → stable (alpha→beta→stable promotion criteria)

## Handoffs
- [260601_benchmark-staging-handoff](handoffs/260601_benchmark-staging-handoff.md) — Wave 0 done; Waves 1-5 implementation ready; spec oracle-approved

## Misc
- [260604_sphinx-warning-backlog](misc/260604_sphinx-warning-backlog.md) — Sphinx warnings deferred from fail_on_warning (sphinx-build unavailable at release-prep time)
- [260604_release-notes-v0.1.0a1](misc/260604_release-notes-v0.1.0a1.md) — Release notes for v0.1.0a1 alpha

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
