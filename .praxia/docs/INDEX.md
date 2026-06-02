# prxteinmpnn Internal Docs

## Roadmaps
- [INDEX](roadmaps/INDEX.md) — all roadmap documents with status
- [260507_refactor-phases-0-6](roadmaps/260507_refactor-phases-0-6.md) — **DEPRECATED** phases 0–6 structural refactor (historical reference)
- [260508_active-roadmap](roadmaps/260508_active-roadmap.md) — **Active** — MODELINPUTS PR-4/5, EncoderPreFn/PostFn, multi_state_temperature

## Plans
- [260522_comp-new-sink-unify](plans/260522_comp-new-sink-unify.md) — COMP-NEW: unify result-sink topology; streaming_tensor_sink_session for non-streaming path
- [260525_comp-unified-encoder-fusion](plans/260525_comp-unified-encoder-fusion.md) — COMP-UNIFIED: encoder fusion via InferencePlan; eliminate averaged-path branch
- [260527_sprint6-decode-axis-composability](superpowers/plans/260527_sprint6-decode-axis-composability.md) — Sprint 6 COMPLETE ✅: composable decode modes (ConditionalDecode/AR/STE), iterator injection, driver.py retired

## Specs
- [260527_merge-readiness-hardening](specs/260527_merge-readiness-hardening.md) — ORACLE-REVIEWED PASS (15/18): pre-merge hygiene for refactor-full → main
- [260601_benchmark-spec](specs/260601_benchmark-spec.md) — ORACLE-APPROVED: GPU benchmark suite (prxteinmpnn vs. LigandMPNN vs. ColabDesign)

## Handoffs
- [260601_benchmark-staging-handoff](handoffs/260601_benchmark-staging-handoff.md) — Wave 0 done; Waves 1-5 implementation ready; spec oracle-approved

## Superpowers
> Skill outputs live in `.praxia/docs/superpowers/plans/` and `.praxia/docs/superpowers/specs/`.
- [plans](superpowers/plans/) — brainstorming + writing-plans outputs
- [specs](superpowers/specs/) — specification outputs
