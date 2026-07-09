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
- [260709_proteinebm-aminx-decomposition](specs/260709_proteinebm-aminx-decomposition.md) — **DRAFT design + assessment** (task `260709_aminxtension`) — how to decompose **ProteinEBM** (energy-parameterized denoising score matching; scalar `E_θ(x,s,t)` + conservative score `−∇_xE` over CA coords; Boltz-1/AF3 non-equivariant trunk, 85M params) as a **new energy/score inference+training path in aminx composed on xtrax**. Verdict **GREEN**: it does NOT fit the existing `StageSet` logit topology (new `EnergyReadout`/`ScoreReadout` + `TOPOLOGY_ENERGY`), but the 5 paper applications each map to an xtrax primitive (decoy/ΔΔG = Vmap/SafeMap + Fuse; conformational biasing = difference-Fuse = the multistate mechanism; Langevin annealing = Scan+CarrySpec w/ `schedule_selector` handoff; structure-pred = pipeline() of scans). Training → xtrax `Trainer`/`Engine`/`ResumableState` + optax + orbax; reuses `NoiseSchedule`/`SinusoidalEmbedding`/`SwiGLU`. Port is mechanical (no custom CUDA), but reference `train.py`/`dataset.py` are non-runnable as shipped (3 undefined-name bugs) → reference-intent only. **PBCNet2.0 verdict YELLOW** (defer to a separate lower-priority epic — only the encode→fuse-by-difference→decode *pattern* overlaps; encoder/RDKit-pose/DGL/8.6M-pair data are ~90% net-new, no LigandMPNN reuse). Includes phase graph → backlog DAG, and bathos pre-registration (literature-parity `parity.bth.toml`, accuracy `claim.bth.toml` Union Gate, throughput `benchmark.bth.toml` vs PyTorch+baselines). Next: file praxia EPIC → DAG → brainstorm → adversarial spec critique → **user review gate before filing**.
- [260707_xtrax-migration-gap-audit-runspec-scaffolding](specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md) — **Cardinality bug FIXED 2026-07-07; RunSpec scaffolding findings still open** — fan-out audit for gaps of the same class as the cardinality-mismatch finding below. 27 of 72 `RunSpec` sub-config fields (across `BatchingConfig`, `TiedPositionsConfig`, `AveragingConfig`, `GridLineageConfig`, `LigandConfig`) are populated at construction and never read — the "RS track" migration is far more stalled than its own status doc claims. The RS-1 migration map (`260614_runspec-migration-map.md`) is substantially wrong in BOTH directions: all 22 "already migrated" claims are stale, while 6+ "needs migration" fields are already done. Also revises the cardinality-mismatch finding below: the *default* dispatch path (`use_unified_driver=True`) has no safety net at all (`_dispatch_axis`'s Vmap branch is an unconditional `jax.vmap`, not even the defeatable `safe_map` check). Praxia recon microflow (`vllm/qwythos`) was tried first per instruction and failed outright on this task class (repeat-loop, 0 findings, walltime timeout) after clearing two unrelated infra bugs (missing template sync, backend not globally registered) — Claude Haiku subagents used instead. **Finding E (same day, follow-up):** aminx imports from only 6 of xtrax's 15 subpackages; `utils/safe_map.py`/`safe_scan.py` are local forks of `xtrax.transforms.safe_map`/`safe_scan` (the fork's `batch_size==0` addition is what lets the cardinality bug fail silently instead of raising), and `training/trainer.py` (939 lines) hand-rolls a loop instead of using `xtrax.engine.Engine`.
- [260706_samples-axis-planner-cardinality-mismatch](specs/260706_samples-axis-planner-cardinality-mismatch.md) — **FIXED 2026-07-07** — `n_samples` axis planning (`samples_batch_size`) and the actual runtime sample count (`samples_chunk_size`/`num_samples`) were independent `SamplingSpecification` fields; when the plan picked Vmap, `safe_map`'s `batch_size==0` branch ignored the real array size, silently bypassing the memory-budget check — confirmed live (16→Vmap, then genuinely vmapped 500 with zero chunking) and the more severe default-path variant (Finding D, `_dispatch_axis`'s unconditional `jax.vmap`, no fallback at all). Introduced in `5ca2abf` (2026-05-07), which replaced a previously-safe direct `jax.lax.map` call. **Fix:** `make_sampling_planner` takes the real per-call sample count as `n_samples_override`, resolved before planning instead of after — closes both dispatch-path manifestations at the cardinality source. `tests/host/test_samples_cardinality_fix.py`. `n_structures`/`n_temperatures`/`n_noises` confirmed unaffected. Not part of EPIC #1541 — predates it.
- [260706_epic1541-p4-runner-hostsinks-scoping](specs/260706_epic1541-p4-runner-hostsinks-scoping.md) — **CONVERGED** (challenger+defender adversarial review, 2026-07-06) — EPIC #1541 P4 scoping: "move output_sinks/streaming_host to xtrax" mostly doesn't apply (boundary types already xtrax-sourced, concrete sinks are domain-specific); `StageSet` **cannot** adopt `xtrax.stages.bundle.StageBundle` at all — verified two independent blockers (Protocol/union/Any fields fail its strict validator; PEP 563 string annotations defeat it entirely) — stays local by design, permanently. Remaining P4 scope: delete dead vendored `BoundedCallbackHandler`, rename colliding `async_indexed_stream`.
- [260706_epic1541-planner-joint-budget-migration](specs/260706_epic1541-planner-joint-budget-migration.md) — **IMPLEMENTED & MERGED (2026-07-06)** — migrated `aminx.tiling.planner` onto xtrax 0.4.0a1's joint-budget `BatchPlanner`; all behavior changes (int budget, fail-loud `BudgetInfeasibleError`, `decision_for()` helper, `plan_bucketed()` rewrite, CarrySpec/DedupSpec cascade) landed and gated (T-PLANNER.GATE, 4/4 scenarios). Supersedes the two 260706 planner/carry decisions.
- [260622_1203_proxide-heterogeneous-inputs](specs/260622_1203_proxide-heterogeneous-inputs.md) — #1203 SPEC_DRAFT: URI-scheme heterogeneous inputs (pdb://, afdb://, mdcath://) resolved to local paths at CLI; offline-cluster-safe; no schema break. Task DAG T1-T9. Pending challenger/defender.
- [260618_rs7b-averaged-scoring-design](specs/260618_rs7b-averaged-scoring-design.md) — RS-7b averaged-topology scoring design (AC-RS-7)
- [260611_aminx-xtrax-refactor](specs/260611_aminx-xtrax-refactor.md) — **Active** — xtrax vertical-slice refactor (T0–T5), gates, prolix second-consumer API
- [260611_runspec-unification](specs/260611_runspec-unification.md) — **Active** — RunSpec + PlannerTopology addendum (RS track); blocks T4.1
- [260611_architecture-sequencing-testing-and-pack](specs/260611_architecture-sequencing-testing-and-pack.md) — brainstorm convergence record (composite winner)
- [260527_merge-readiness-hardening](specs/260527_merge-readiness-hardening.md) — ORACLE-REVIEWED PASS (15/18): pre-merge hygiene for refactor-full → main
- [260601_benchmark-spec](specs/260601_benchmark-spec.md) — ORACLE-APPROVED: GPU benchmark suite (aminx vs. LigandMPNN vs. ColabDesign)
- [260604_release-preparedness-epic](specs/260604_release-preparedness-epic-for-prxteinmpn.md) — Release Preparedness epic: two-phase backlog DAG (CI+cleanup → docs+release), Definition of Done, ADR index
- [260611_aminx-xtrax-refactor](specs/260611_aminx-xtrax-refactor.md) — **Challenger/Defender PASS**: refactor aminx onto xtrax (move tiling/sinks/inference-plan, rebuild training); vertical-slice spine + xtrax-tiling-upgrade precondition gate + branch-by-abstraction flag + dual perf-guard + off-ramp. Backlog EPIC #1541, sprint `260611_xtrax-foundations`. **Status (2026-07-06, appended to spec): P0-P3 + T2.GATE DONE; P4 (inference-runner/host-sinks) NOT STARTED; P5 partially satisfied (xtrax 0.4.0a1 published+repinned, clean-resolve verified; full G5 blocked on G4/P4).**

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
- [260630_runtimebundle-inputresolver-compose-not-subclass](decisions/260630_runtimebundle-inputresolver-compose-not-subclass.md) — **Accepted 2026-06-30**: aminx's planned RuntimeBundle (#1910) must compose, not subclass, xtrax.run.resolver.RuntimeBundle (frozen-vs-non-frozen dataclass conflict); InputResolver implemented via functools.singledispatch per xtrax's own prescribed pattern. Zero upstream xtrax changes needed.
- [260702_wave-color-commits-retroactive-attribution](decisions/260702_wave-color-commits-retroactive-attribution.md) — **Accepted 2026-07-02**: retroactively attributes 5 wave-color commits (54d6d84, 0be59ef, 4060e9d, 0670197, 1cec556) to mpnn_ext Epic WAVE (#2871); kept in place, no revert. Backlog #2954 tracks the enforcement lint that should have caught this.
- [260706_planner-stays-on-aminx-tiling-by-design](decisions/260706_planner-stays-on-aminx-tiling-by-design.md) — **Superseded** (2026-07-06, same day) — the algorithmic gap this recorded is closed by xtrax 0.4.0a1's joint-budget mode; see the planner-migration spec.
- [260706_carryspec-dedupspec-stay-local-carryshape-migrates](decisions/260706_carryspec-dedupspec-stay-local-carryshape-migrates.md) — **Superseded in part** (2026-07-06, same day) — `CarrySpec`/`DedupSpec` unblocked by the planner migration spec; `CarryShape` portion (migrated) still stands.
- [260706_bucketing-pad-stay-local-epic-1541-p3-scope-closed](decisions/260706_bucketing-pad-stay-local-epic-1541-p3-scope-closed.md) — **Accepted 2026-07-06**: EPIC #1541 P3 scoping closed — `bucketing.py`/`pad.py` stay local (planner companion / pure domain logic). `aminx.tiling` will NOT be fully deletable as originally envisioned; recommends updating backlog #1483 accordingly.

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
