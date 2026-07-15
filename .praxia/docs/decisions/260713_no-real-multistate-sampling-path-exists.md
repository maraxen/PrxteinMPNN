---
title: No campaign-reachable code path ever constructs a genuine num_states>1 bundle for autoregressive sampling — MVP decided, tech debt filed for the general axis
status: Decided and MVP IMPLEMENTED (2026-07-13) — option 1 shipped as aminx.sampling.multistate_poe (PR TBD, branch feat-poe-stacked-bundle-sampling); genuine cross-state fusion confirmed end-to-end against real reference-state PDBs. Option 2 (general N_POE_STATES-style axis via xtrax) filed as real, should-be-done tech debt (praxia debt #589), not closed by option 1. Option 3 rejected.
date: 2026-07-13
related: spec ../specs/260709_mbr-consensus-reranking-composition.md (PR #92, #95), decisions/260709_n-states-heterogeneous-flag-unenforced.md, praxia debt #572/#575/#589, tev_design task 260709_multistate-fusion-strategy-comparison Phase 3, PR #100 (corruption-symptom fix, merged)
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

## Decision (made 2026-07-13, tev_design project owner) — option 1 IMPLEMENTED

**Option 1 is the MVP, implemented** as `aminx.sampling.multistate_poe`
(`sample_multistate_poe_bead` / `sample_states_fused`, branch `feat-poe-stacked-bundle-sampling`).
**Option 2 is real tech debt, filed below, not resolved by shipping option 1.** Option 3 is
rejected — the necklace campaign's PoE premise is being made real, not abandoned.

1. **[CHOSEN, MVP, IMPLEMENTED] New, separate sampling entry point** (small-medium): builds ONE
   genuine `(S, L, ...)` stacked bundle for a bead's k reference states (reusing
   `prep_protein_stream_and_model`'s real structure-loading/padding pipeline, unchanged, plus
   `build_inference_bundle` — already S>1-capable when given a genuinely stacked coords array,
   confirmed by the pre-existing `test_autoregressive_produces_valid_output_s4` test) and samples
   via `aminx.inference.sample_autoregressive.kernel` (the already-correct, already-tested
   `AutoregressiveMode` path). Precondition: `spec.batch_size == len(spec.inputs)`, so the protein
   dataset iterator yields exactly one combined batch. Does not touch `_sample_batch`/
   `kernel_dispatch.py`/`campaign.py` at all — every existing (single-structure) campaign row is
   unaffected; lowest regression risk, confirmed by the unchanged full test suite.

   **Two real bugs found and fixed during implementation** (both caller-side, in the new module —
   not aminx architecture bugs):
   - `_prepare_fixed_controls` returns a `(num_states, L)` array shaped for
     `kernel_dispatch.py`'s per-structure_idx *slicing* (each independent call takes its own 1D
     `(L,)` row). `build_inference_bundle`'s `fixed_mask`/`fixed_tokens` are design-level, not
     per-state (its own default is 1D: `jnp.zeros(seq_len)`). Passing the full un-sliced 2D array
     silently built a wrong-shaped `ConditioningBundle.fixed_mask`/`fixed_tokens`, which corrupted
     `AutoregressiveDecode`'s wave-scan carry shape several call-frames later (`Cannot broadcast to
     shape with fewer dimensions` inside `do_sample`'s `seq_oh_stack` construction) — a genuinely
     confusing failure mode to trace (see the module's inline comment for the full diagnosis).
     Fixed by taking row 0 (every row is already an identical broadcast, now asserted explicitly
     rather than assumed) before passing to `build_inference_bundle`.
   - **CORRECTED (2026-07-13, caught by an independent PR audit — this bullet previously claimed
     a fix that was never actually shipped):** during debugging, `jax.vmap` and `jax.lax.map`
     over the sample-key axis both appeared to fail with the same `Cannot broadcast to shape
     with fewer dimensions` error as the fixed_mask bug above. At the time this was misattributed
     to vmap/lax.map itself being incompatible with a genuine `num_states>1` bundle, and this
     document previously claimed the fix was "looping the sample-key axis explicitly instead."
     **That is not what shipped.** `sample_states_fused` uses `jax.vmap` over `n_samples`,
     unchanged, in the merged code — and the audit independently reproduced `num_states=2,
     n_samples=8, multi_state_strategy="product"` against real PDB data with no crash. The
     actual explanation: both vmap and lax.map attempts were made *before* the fixed_mask bug
     (above) was found and fixed, so both were failing for the *same* underlying reason (the
     malformed 2D fixed_mask), not because of anything specific to vmap/lax.map. Once the
     fixed_mask bug was fixed, the original vmap-based code simply worked, and no separate
     sample-dispatch fix was ever needed. There is no evidence of a real vmap/multi-state
     incompatibility in `AutoregressiveDecode` — this bullet is corrected to avoid future
     sessions treating a debugging red herring as a documented aminx limitation.
2. **[NOT CHOSEN for MVP — filed as real tech debt, praxia debt #589, see below] Real
   `N_POE_STATES`-style axis in `_sample_batch` itself.**
3. **[REJECTED] Relabel the necklace campaign's intent instead of fixing aminx** — dropping the
   `multi_state_strategy="product"`/`state_position_map` framing and re-scoping "PoE" as 4
   independent single-state libraries. Explicitly rejected by the project owner (2026-07-13): the
   goal is to make the campaign's locked PoE premise real, not retreat from it.

## Independent PR audit (2026-07-13, before merge with admin override)

An independent subagent audit of the shipped diff (not this doc's own claims) returned
**CHANGES REQUESTED**, with the following findings, all now addressed before merge:

- **F1 (High, fixed):** the original test suite proved genuine fusion only on a synthetic
  hand-built bundle, and proved the real host pipeline doesn't crash only with a trivial
  identity `state_position_map` — never both together, so "fusion works end-to-end against
  real data" was not actually demonstrated for the campaign's real use case (heterogeneous
  junction lengths). Fixed: added
  `test_real_state_position_map_through_full_pipeline_changes_output`, which computes a
  genuine non-identity map via `aminx.utils.align.build_state_position_map` on two real,
  differently-numbered PDB structures and threads it through the full real
  `sample_multistate_poe_bead` pipeline, confirming fused logits differ from the
  identity-map case under the same PRNG key.
- **F2 (resolved, no code change):** flagged the row-0-identity assumption on
  `fixed_mask`/`fixed_tokens` as a possible real-production risk. A follow-up research
  pass confirmed against the actual tev_design necklace manifest-building scripts that
  fixed positions are locked identical across all k states by design (the prereg requires
  uniform junction placement regardless of stratum) — the assumption is safe as practiced.
  Downgraded to a docstring note (added) rather than a required fix.
- **F3 (Medium, fixed):** this document's Provenance section previously claimed a
  `jax.vmap`→explicit-loop fix for a sample-dispatch/multi-state incompatibility that was
  **never actually shipped** — `sample_states_fused` still uses `jax.vmap`, unchanged, and
  the audit independently reproduced `num_states=2, n_samples=8` against real data with no
  crash. The real explanation: both `vmap` and `lax.map` were tried *before* the
  `fixed_mask` bug (above) was found, so both failed for that same reason, not because of
  anything specific to vmap/multi-state. Corrected in the Provenance section above rather
  than left as a false claim about a nonexistent aminx limitation.
- **F4 (High, fixed):** zero test coverage existed for `model_family="ligandmpnn"` +
  `sidechain_conditioning=True` — the real necklace campaign's actual production model
  configuration (e.g. `ligand_mpnn_v32_020_25_sc_only`). Fixed: added
  `test_ligandmpnn_sidechain_conditioning_end_to_end`, using `PrxteinLigandMPNN`
  (`ligand_mpnn_use_side_chain_context=True`) rather than the base `Aminx` class (a first
  attempt with the base class failed loudly with a clear `TypeError`, confirming the
  gap was real, not hypothetical).
- **F5 (Medium, fixed):** no `@eqx.filter_jit` wrapping anywhere in the new module, unlike
  every comparable production sampling call site (`host/plan.py`'s `InferencePlan.encode`/
  `.decode`/`.sample`/`.score`). Fixed: `sample_states_fused` is now `@eqx.filter_jit`-decorated,
  matching the codebase's existing convention; full test file re-verified green after adding it.

All 5 findings closed; full new test file (9 tests) passes; full aminx suite re-verified
before merge (see commit history for exact pass count).

## Tech debt (praxia debt #589): option 2 should genuinely be built, and should be xtrax-composable

**This is flagged explicitly as work that REALLY SHOULD happen, not a nice-to-have** — option 1
(the MVP) is scoped as a special-purpose entry point for exactly one case (k reference states,
one PoE bead, sampling only). It does not generalize. The moment a second axis needs the same
"fuse instead of batch-independently" treatment — e.g. an ensemble-over-checkpoints axis (debt
#536's deferred cross-model fusion), a multi-conformation axis for a different campaign, or
literally any future experiment that wants "N things fused into one call" instead of "N
independent batch items" — a hand-rolled option-1-style entry point would need to be rebuilt from
scratch for that axis too. That is the wrong shape for a codebase whose entire tiling/dispatch
layer (`host/plan.py`'s `AxisDecision`, `tiling/axes.py`'s `AxisSpec` registry,
`make_axis_dispatch_via_xtrax`) already exists specifically to make "batch/dispatch an arbitrary
axis" a solved, composable problem — `N_STRUCTURES`, `N_SAMPLES`, `N_TEMPERATURES`, `N_NOISES`,
and (for candidates) `N_CANDIDATES` all already go through this machinery. `N_STATES`/a
states-to-fuse axis for real autoregressive sampling is the one dispatch-relevant axis that
doesn't, and it should, via the same idiom, not a bespoke path per caller.

**Concretely, when this is picked up:** design a real `N_POE_STATES`-or-similar `AxisSpec`
registry entry (distinct from the existing `N_STATES` — see
`decisions/260709_n-states-heterogeneous-flag-unenforced.md`, which is about a different,
currently-unenforced heterogeneity flag on an axis that isn't wired to sampling at all) that plugs
into `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax`/`BatchPlanner` the same way
`N_CANDIDATES` already does for `make_batched_conditional_logits_split_fn`'s candidate axis
(`sampling/conditional_logits.py:287-334,412-437` — the existing, proven idiom to mirror, per this
session's earlier MBR-rerank spec correction history). The goal is that a future caller wanting
"fuse N things instead of batching them independently" — whatever N and whatever the things are —
declares an `AxisSpec` and gets `Vmap`/`SafeMap` dispatch, JIT behavior, and memory-budget
awareness for free, exactly like every other axis in this codebase, instead of every new fusion
use case getting its own one-off entry point like option 1's MVP will be. This is real engineering
work on the live production sampling path (touches all four near-duplicate dispatch branches in
`kernel_dispatch.py`) with its own JIT/parity/test burden — do not attempt it as a quick follow-up
to option 1; scope and resource it as its own real piece of work.

## Non-decision for now

Option 1's MVP unblocks tev_design's Phase 3 Stage 0-2 work for the necklace campaign
specifically. It does **not** close debt #589 (option 2) — that remains open, tracked, and should
be picked up deliberately once a second real caller needing arbitrary-axis fusion emerges (or
sooner, if resourced), not left to rot the way `N_STATES.heterogeneous` did.
