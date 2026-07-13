---
title: No campaign-reachable code path ever constructs a genuine num_states>1 bundle for autoregressive sampling — decide the fix architecture
status: Open — deferred by explicit project-owner decision (2026-07-13); the silent-corruption symptom is fixed separately (this session), this doc scopes the remaining real gap
date: 2026-07-13
related: spec ../specs/260709_mbr-consensus-reranking-composition.md (PR #92, #95), decisions/260709_n-states-heterogeneous-flag-unenforced.md, praxia debt #572/#575, tev_design task 260709_multistate-fusion-strategy-comparison Phase 3
---

# Decision needed: how to give `aminx campaign plan`/`run` a real multi-state PoE sampling path

## Finding

**No CLI-reachable call path in aminx ever constructs a bundle with a genuinely stacked
`num_states > 1` for autoregressive sampling.** This was discovered while tev_design's Phase 3
(`260709_multistate-fusion-strategy-comparison`) tried to analyze the necklace campaign's "PoE"
production library and found each row stores **four independent, unfused single-state decodes**
(one per `--inputs` CIF), not a fused cross-state consensus — despite `multi_state_strategy="product"`
and `state_position_map` (PR #97/#99, debt #572) being set on every row.

Traced end-to-end (aminx wheel `0.1.0a14`, pinned commit `7f4c300`):

- `aminx campaign plan`'s CLI (`host/campaign.py`) has no flag distinguishing "N independent
  design targets to batch together" from "k reference states of one design, to be fused." Every
  `--inputs` path becomes one item on the generic `N_STRUCTURES` batching axis.
- `host/kernel_dispatch.py::_sample_batch` loops `structure_idx` over that axis and, for each one,
  slices **that one structure's own** `(L, 4, 3)` coords and calls `build_inference_bundle` —
  `inference/bundle_builder.py` then fixes `num_states = 1` for every one of those calls, in all
  four dispatch branches.
- The fusion machinery itself is real, correct, and independently tested: `AutoregressiveDecode.do_sample`
  + `LOGIT_STRATEGIES["product"]` + `_realign_states_to_reference` genuinely fuse a real
  `num_states=2+` stacked bundle correctly (`tests/inference/decode/test_autoregressive.py::
  test_ar_decode_state_position_map_changes_fused_logits` constructs one directly and confirms
  fused logits change with a non-identity `state_position_map`). The gap is purely that no
  host/CLI call site ever hands it one.
- tev_design's older 9-state analysis pipeline (`scripts/experiments/stage1_conditioning_logits_comparison.py`)
  achieves genuine multi-state combination a different way — `build_flat_multistate_graph`/
  `structure_mapping` concatenates all states into one *flat* graph (`num_states` stays 1;
  cross-state combination happens via attention masking on the flat sequence, not a stacked `S`
  axis) — and only for teacher-forced scoring (`score_conditional`/`score_unconditional`), never
  for `AutoregressiveDecode.do_sample`. It is not a template that already solves this for sampling.

## A second, worse bug this surfaced (fixed separately, this session)

`_sample_batch` broadcasts the caller's real `(S=4, L)` `state_position_map` unchanged into every
`structure_idx`'s single-state (`num_states=1`) bundle. `_realign_states_to_reference`
(`inference/decode/_kernel.py`) did not previously check that `state_position_map.shape[0]`
matched `logits.shape[0]`, so `jnp.take_along_axis` silently broadcast the one real state's logits
across the mismatched axis — gathering that same one state's logits through several different
(and, past row 0, genuinely wrong) permutation rows, then summing them under `product` fusion.
This fabricated a fake 4-way "fusion" out of one real state, actively distorting every per-token
sampling decision in every "PoE" row ever produced by this campaign (not merely failing to fuse).
**Fixed in this same session**: `_realign_states_to_reference` now raises `ValueError` on any
state-cardinality mismatch instead of broadcasting, with a regression test
(`tests/inference/decode/test_kernel.py::TestRealignStatesToReference::
test_state_cardinality_mismatch_raises`). This means any future campaign resubmission that reaches
this call path with a real PoE bead will now fail loudly rather than silently corrupt — which is
the correct interim behavior until the real gap below is closed.

## Practical consequence today

Any necklace-campaign PoE row generated against the current pinned aminx version will now raise
`ValueError` at sampling time (the fabrication that used to paper over the mismatch is gone, and
nothing replaces it yet). **No new PoE campaign data can be generated until one of the options
below is implemented** — this is a hard blocker on tev_design's Phase 3 Stage 0-2 work, not just a
data-quality caveat. Every row in the already-existing `full_2048_aligned/` library (job 17768233)
and every earlier necklace library version was sampled through the corrupted path described above
and should not be treated as real product-of-experts consensus data for any downstream analysis
or design decision until regenerated against a real fix.

## Decision options (not decided here — flagging for deliberate choice, per tev_design project
owner's 2026-07-13 direction to defer this and fix only the corruption symptom in this pass)

1. **New, separate sampling entry point** (small-medium): construct a genuine `(S=4, L, ...)`
   stacked bundle once per PoE bead (reusing tev_design's existing `compute_poe_chain_id`/
   `compute_state_position_map` padding-and-alignment work) and call `AutoregressiveDecode.do_sample`
   directly — mirroring how `stage1_conditioning_logits_comparison.py` calls `build_inference_bundle`
   directly for scoring, but for `mode="sample_ar"`. Needs a new `SamplingSpecification` field/
   campaign row type (or a documented way to request it) since `--inputs` today always means
   "N independent structures." Lowest regression risk — `_sample_batch`'s existing N_STRUCTURES
   path is untouched.
2. **Real `N_POE_STATES`-style axis in `_sample_batch` itself** (medium-large): give the existing
   `AxisDecision`/`Vmap`/`SafeMap` dispatch machinery (`host/plan.py`) a genuine states-to-fuse axis
   distinct from `N_STRUCTURES`, so a manifest row can request "N_STRUCTURES=1, states=4" instead of
   today's implicit "N_STRUCTURES=4, states=1 each." More architecturally consistent with how every
   other axis in this codebase is dispatched, but touches all four near-duplicate branches in
   `kernel_dispatch.py` — real engineering work on the live production sampling path, its own
   JIT/parity/test burden.
3. **Relabel the necklace campaign's intent** instead of fixing aminx: if genuine per-token
   cross-state fusion turns out not to be worth the engineering cost, formally re-scope the
   necklace "PoE" library as "4 independent single-state libraries co-manifested for convenience"
   and drop the `multi_state_strategy="product"`/`state_position_map` framing entirely — a scope
   decision, not a code fix, and would mean re-litigating the necklace campaign's core premise
   (`project_necklace_library_campaign` memory's 2026-06-29 lock).

## Non-decision for now

This document exists so the underlying gap isn't lost, decided by default, or discovered the hard
way a second time by a future session. Per explicit project-owner direction (2026-07-13), this
pass fixes only the silent-corruption symptom (see above) and defers the architecture decision
above; tev_design's Phase 3 Stage 0-2 work stays blocked until one of options 1-3 is chosen and
implemented.
