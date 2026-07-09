---
title: MBR post-hoc consensus reranking — composed from existing R×C teacher-forced scoring infra, not a new axis_boundaries Fuse
status: Draft (supersedes a factually-incorrect contemplex brainstorm artifact — see "Provenance" below)
date: 2026-07-09
related: praxia debt #536 (tev_design), tev_design prereg 260709_multistate-fusion-strategy-comparison.md
---

# Spec: MBR Post-Hoc Consensus Reranking

## Provenance — this document corrects a wrong prior conclusion

An autonomous contemplex brainstorm session (tev_design, task 260709_multistate-fusion-strategy-comparison,
session `a3680ff9`, artifact `tev_design/.praxia/docs/specs/260709_aminx-how-to-compose-new-ensemble-fusion.md`)
concluded aminx should ship MBR consensus reranking as a **new `xtrax.stages.boundaries.Fuse` populating
the dead `StageSet.axis_boundaries` field, reducing `compute_pseudo_perplexity` over `N_CANDIDATES`**.

An adversarial challenge pass (spec-challenger agent) verified this against the live source and found the
core premise **factually wrong** on two independent points, both confirmed by direct reading (not
paraphrase) during this correction:

1. `compute_pseudo_perplexity` (`src/aminx/host/logit_aggregation.py:14-53`) takes
   `[batch, samples, noise, temp, seq_len, 21]` and returns `[batch, samples, noise, temp]` — it collapses
   `seq_len`/vocab only. The candidate/sample axis is **preserved in the output, not reduced**. This is a
   per-candidate scorer, not a `Fuse` (`Stacked[S] -> single O`) over the candidate axis.
2. `N_CANDIDATES` (`src/aminx/tiling/axes.py:131-137`) is **not** an unused axis. It already has a real,
   current consumer: `make_batched_conditional_logits_split_fn`'s `batched_decode_fn`
   (`src/aminx/sampling/conditional_logits.py:412-437`), which dispatches teacher-forced decode of
   externally-provided candidate sequences via `make_axis_dispatch_via_xtrax(strategy, axis=N_CANDIDATES.name)`
   today, in production.

The brainstorm's abstraction-scope reasoning (don't build `EnsembleFuse` prematurely; defer the
cross-model debt #536 to documentation-only) is **not affected** by this correction and stands — see
§4. Only the concrete mechanism for (A) is replaced below.

## 1. What already exists (verified, reusable as-is) — and it's more than the first correction found

- **`aminx.score()`** (fully public, top-level export — `aminx/__init__.py:24`, implemented at
  `src/aminx/scoring/score.py:157-229`) already does encode→decode→score in ONE call and returns
  `(masked_average_score, logits, decoding_order)` (`score.py:203`). `masked_average_score` is the
  per-call NLL-equivalent quantity — **there is no need to separately call `compute_pseudo_perplexity`
  at all**; `aminx.score()` already produces the scoring output MBR reranking needs, per call.
  `multi_state_strategy`/`state_weights` are accepted directly (`score.py:169-171`); at a single state
  (S=1, one call per reference state — see §2) the strategy choice is moot.
- Underneath `aminx.score()` (for context, not code you need to call directly):
  `aminx.inference.score_conditional.kernel` (`score_conditional.py:164-180`) — `encode` + `score_from_encoding`
  in one call, "byte-identical to original implementation" per its own docstring; `ConditionalDecode`
  (`inference/decode/conditional.py:33-37`) fuses across states via `stage_set.logit_transform` inside
  that call — this is what makes calling with more than one state at once the WRONG shape for MBR (see §2).
- **`make_batched_conditional_logits_split_fn`** (`sampling/conditional_logits.py:337-439`) — the
  established R×C (replicate × candidate) dispatch pattern, in case batching many candidates through
  `aminx.score()`-equivalent machinery in one call (rather than looping) is worth doing for throughput;
  `batched_decode_fn` already dispatches over `N_CANDIDATES` via `BatchPlanner`. Not required for
  correctness — `aminx.score()` alone is sufficient to build a correct-but-possibly-slower version first.

## 2. The actual shape mismatch this spec resolves

MBR reranking (per tev_design idea-006 / the 260709 research prereg) needs the **average of k
independently-computed per-state scores**, not a score against an already-state-fused logit tensor.
Calling `aminx.score()` once with all k states loaded (via its `multi_state_strategy`) would produce
exactly the wrong quantity — a score against fused logits, not the average of per-state scores. These
are mathematically different, and using the fused quantity would just be re-scoring the same in-loop
mechanism the research question needs a genuine alternative to.

**Resolution:** call `aminx.score()` once per state (S=1 each call, one reference-state structure at a
time). `multi_state_strategy` is moot at a singleton state — no new "no-op fusion" mode is needed.

## 3. New code — small, correctly scoped (possibly nothing beyond a thin convenience wrapper)

Given `aminx.score()` already returns the per-call score directly, the only genuinely new logic is:

1. **`average_cross_state_scores(per_state_scores: Float[Array, "k candidates"]) -> Float[Array, "candidates"]`**
   — a small, pure-JAX reduction, genuinely `Stacked[S] -> Out[O]` (k states → one score per candidate).
2. **`select_mbr_candidates(scores, sequences, top_k=1) -> selected sequences/indices`** — argmin/top-k
   over the candidate axis. A second, distinct reduction (over candidates, not states) — keep separate
   from step 1; they reduce over different axes and have different call-site needs (e.g. wanting
   top-k > 1 for downstream inspection).

Composition (pseudocode, not final code — and note this may reduce to almost no aminx-side code beyond
a thin wrapper, since `aminx.score()` already does the heavy lifting):

```python
def mbr_rerank(model, state_structures: list[...], candidate_sequences, prng_key, top_k=1, **score_kwargs):
  per_state_scores = []
  for state in state_structures:  # k independent calls, S=1 each
    avg_score, _logits, _order = aminx.score(
      prng_key, model, state.coords, state.mask, state.residue_index, state.chain_index,
      sequence=candidate_sequences, **score_kwargs,
    )
    per_state_scores.append(avg_score)
  mean_score = average_cross_state_scores(jnp.stack(per_state_scores, axis=0))
  return select_mbr_candidates(mean_score, candidate_sequences, top_k=top_k)
```

**Open implementation question (verify before writing real code, don't assume):** does
`make_score_fn`'s `sequence` parameter accept a BATCH of candidate sequences in one call, or exactly
one? If one-at-a-time only, either loop over candidates too (simplest, possibly slow for large C) or
route through `make_batched_conditional_logits_split_fn`'s `N_CANDIDATES`-batched dispatch instead of
`aminx.score()` directly for the candidate axis. This determines whether `mbr_rerank` needs the
`conditional_logits.py` batching machinery at all, or whether `aminx.score()` alone suffices.

## 4. Where this lives — NOT `StageSet.axis_boundaries`

MBR reranking's actual input is **already-sampled sequences from a completed production run** (e.g.
tev_design's necklace P2 job, stored zarr output) being re-scored against the k reference states — this
is inherently a **post-hoc batch utility**, not a per-decode-call `StageSet` stage. `axis_boundaries`
exists to extend a *live* decode/`StageSet` pipeline; MBR reranking runs entirely after sampling is
already complete and files are already written.

**Decision:** ship `mbr_rerank` as a standalone function (new module, e.g.
`aminx.sampling.mbr_consensus` or `aminx.host.mbr_rerank` — implementer's choice, mirror the nearest
existing sibling module's layout) that composes the existing pieces in §1, reusing
`make_batched_conditional_logits_split_fn`'s candidate-axis dispatch pattern directly for the R×C-style
batching, rather than inventing a new dispatch path. `StageSet.axis_boundaries` remains unpopulated by
this work — it is a real, distinct extension point for a *future* live-decode use case, not this one.
Do not force this task to be that use case's first occupant merely because the slot exists and is
otherwise idle.

## 5. Cross-model fusion (praxia debt #536) — deferred, corrected pointer only

No code in this pass. This document is the corrected pointer debt #536 should reference: the existing
composition idiom to mirror when (B) gets a real driver is
`sampling/conditional_logits.py:337-439` (`make_batched_conditional_logits_split_fn`), **not**
`host/kernel_dispatch.py`. debt #536's original filing cited the latter; it is stale. No
`N_MODEL`/`N_CHECKPOINT` axis registry entry is added now — add one only when (B) has an actual driver,
per this project's no-premature-abstraction convention.

**Explicit trigger condition for building (B):** a real consumer needs live cross-checkpoint PoE fusion
during AR sampling (not post-hoc scoring — that's a much smaller ask and could likely reuse this same
`mbr_rerank`-style pattern with a checkpoint axis substituted for the state axis, worth checking first).

## 6. Acceptance Criteria (Given/When/Then)

- **Given** k reference-state structures and a set of already-sampled candidate sequences from a
  completed production run.
- **When** `mbr_rerank` is called with these inputs.
- **Then** it returns the same per-candidate mean score as k separate manual calls to `aminx.score()`
  (one per state) averaged by hand — i.e. a unit test must assert numerical equivalence against that
  manual reference computation, not just "runs without error."

- **Given** `average_cross_state_scores` called with `k=1` (a single state).
- **When** compared against calling `aminx.score()` directly on that one state.
- **Then** results are identical (sanity check that the reduction is a true no-op at k=1 — this specific
  claim must be tested, not just asserted by mathematical reasoning as in §2).

- **Given** the finished `mbr_rerank` module.
- **When** grepped for imports.
- **Then** it imports the public `aminx.score()` (or, if the candidate-batching question in §3 resolves
  toward it, `conditional_logits.py`'s batched dispatch) — and does **not** touch `types/stages.py`'s
  `axis_boundaries` field or `inference/decode/autoregressive.py`'s live AR-loop fusion call site.

- **Given** the candidate-batching open question in §3.
- **When** implementation begins.
- **Then** it is resolved by reading `make_score_fn`'s actual `sequence`-argument handling first — not
  assumed in either direction.

## 7. Assumptions

- The k reference-state bundles (1LVB, 1LVM, reac1-reac, reac2-reac per tev_design's necklace campaign)
  are already available as separate single-state `InferenceBundle`s — this spec does not address how
  tev_design constructs those; that's existing, working infra (canonical_bundle.npz per tev_design's
  `build_canonical_bundle.py`).
- `stage_set.logit_transform` being a true identity at S=1 for all three registered strategies
  (arithmetic_mean/geometric_mean/product) is asserted in §2 by mathematical reasoning, not yet verified
  against the actual implementation of each — Acceptance Criterion 2 (§6) closes this gap with a real
  test rather than leaving it as an unverified assumption.

## 8. TBDs

- Exact new module path/name (`aminx.sampling.mbr_consensus` vs. `aminx.host.mbr_rerank` vs. other) —
  implementer's call, follow the nearest sibling module's naming convention at time of implementation.
- **Resolved during this spec's own verification** (was originally a TBD): `aminx.score()` IS already a
  fully public, top-level export (`aminx/__init__.py:24`) that does encode→decode→score in one call and
  already returns the scoring quantity MBR needs directly — confirmed by reading `scoring/score.py`
  directly, not assumed. This means the amount of genuinely new aminx code is small (§3's two functions,
  possibly less depending on the candidate-batching answer) — most of the "composition" this spec
  describes is calling existing public API correctly (once per state), not building new machinery.
- Whether `mbr_rerank` ships in aminx (as a small reusable convenience wrapper around `aminx.score()`,
  matching the stated preference for composing in aminx rather than patching tev_design) or is simple
  enough that tev_design just calls `aminx.score()` k times directly in its own analysis script without
  needing aminx to ship anything new — given how little new logic remains (§3), this is a real judgment
  call for whoever implements it, not something this spec should force one way. Recommendation: ship the
  thin wrapper in aminx anyway, since a second consumer benefiting from the same k-state-average-and-rerank
  pattern is plausible (the tev_design necklace campaign already runs against a fixed k=4 state set that
  other analyses reuse) — but this is a recommendation, not a requirement.
