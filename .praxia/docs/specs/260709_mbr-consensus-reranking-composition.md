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

**Second round (independent cold review, no context from the above):** two fresh review passes (one on
technical correctness, one on implementability/scope discipline) found this corrected document itself
had real, fixable gaps — not rubber-stamped. Findings incorporated below, each marked at its section:
`score()` takes exactly one candidate per call, not a batch (the first version of this corrected
pseudocode got this wrong too — §1/§3); the "identity at S=1" claim was too strong, true only under
default temperature/weights (§2); no scoring-direction (lower-vs-higher) convention was ever stated,
now fixed with an explicit acceptance criterion (§6); one acceptance criterion was a process instruction
in Given/When/Then clothing, not a testable assertion (removed); two TBDs were judgment calls important
enough to decide now rather than defer (§8).

**Third round (direct correction, project maintainer):** the second round's fix was itself wrong in a
more important way than any individual bug — it accepted "loop over candidates one at a time calling
`aminx.score()`" as the reference design, with xtrax-based candidate batching relegated to an optional
throughput follow-up. That's backwards. `make_batched_conditional_logits_split_fn`'s `batched_decode_fn`
(`conditional_logits.py:412-437`) **already batches an arbitrary number of candidates via xtrax's
`AxisSpec`/`BatchPlanner`** on `N_CANDIDATES` (Vmap when it fits, SafeMap-chunked otherwise, no
hard-coded limit) — this is precisely the "batching arbitrary axes should be easy" capability xtrax was
adopted for, and it was already documented in §1 of this very file before being wrongly demoted. §3 is
rewritten below to make the batched path the actual design, not a deferred optimization.

## 1. What already exists (verified, reusable as-is)

- **`make_batched_conditional_logits_split_fn`** (`sampling/conditional_logits.py:337-439`) — **this is
  the primary primitive this design is built on**, not a fallback. `batched_encode_fn`/`batched_decode_fn`
  batch an arbitrary number of candidate sequences via xtrax's `AxisSpec`/`BatchPlanner` on `N_CANDIDATES`
  (`_plan_axis_strategy` chooses Vmap when the candidate count fits the memory budget, SafeMap-chunked
  otherwise — no hard-coded candidate-count limit). `batched_decode_fn(encodings, candidate_sequences,
  ar_mask=None)` returns `(R, C, L, 21)` logits (`conditional_logits.py:437`) — raw logits, not a score;
  a scoring step is applied on top (see §3).
- **`make_encoding_conditional_logits_split_fn`** (`conditional_logits.py:145-177`, the un-batched pair
  `batched_decode_fn` wraps) — docstring states the exact intended usage directly: *"Encode once... decode
  multiple sequences using the same encoding."* No state-fusion machinery anywhere in this pair (no
  `stage_set`/`logit_transform` argument) — this path never touches `ConditionalDecode`'s per-state
  fusion at all, side-stepping the "identity at S=1" precondition problem that applies to `aminx.score()`
  (see §2's note on why `score()` is NOT used as the primary path here).
- **`compute_pseudo_perplexity`** (`host/logit_aggregation.py:14-53`) — pure JAX, correct as-is; used on
  `batched_decode_fn`'s output to turn `(R, C, L, 21)` logits into per-candidate NLL. **Real integration
  detail, not a blocker:** its signature hardcodes a rigid 4-leading-dim shape convention
  (`[batch, samples, noise, temp, seq_len, 21]` in, `[batch, samples, noise, temp]` out, with
  `mask_sum[:, None, None, None]` baked in at line 51) inherited from a different pipeline (the stage3
  sampling grid). `batched_decode_fn`'s `(R, C, L, 21)` output only has 2 leading dims, not 4 — either
  reshape to `(R, C, 1, 1, L, 21)` before calling it (R=1 in this design, so trivial), or write the
  ~4-line NLL reduction directly (`-jnp.sum(one_hot(seq) * log_softmax(logits, axis=-1), axis=(-1,-2))`,
  which is axis-generic and doesn't need the rigid wrapper at all). Flag this explicitly at
  implementation time rather than assuming direct compatibility.
- **`aminx.score()`** (fully public, top-level export — `aminx/__init__.py:24`,
  `src/aminx/scoring/score.py:157-230`) — exists, returns `(masked_average_score, logits,
  decoding_order)` where `masked_average_score` is masked-average NLL (`_nll_from_logits`,
  `score.py:29-51`) — **lower is better**. Documented here for context and because it establishes the
  scoring-direction convention (§2), but it is **not** used as this design's primary path: it takes
  exactly one candidate sequence per call (`L = sequence.shape[0]` at `score.py:111`, no batch dimension
  anywhere), which would force a Python loop over candidates instead of the xtrax-batched dispatch
  above — precisely the wrong direction given xtrax's whole purpose is making arbitrary-axis batching
  easy, not something to bypass with a manual loop.
- Underneath both `aminx.score()` and this design's primary path:
  `ConditionalDecode` fuses per-state logits via `self._apply_logit_transform(logits_stack, stage_set,
  ...)` (`inference/decode/conditional.py:154`) when a `stage_set` is involved — this is what makes
  calling with more than one state loaded at once the WRONG shape for MBR (see §2). The primary path
  here (`make_encoding_conditional_logits_split_fn`) doesn't invoke this at all, so it isn't a concern
  for it; it only matters as an explanation of why `aminx.score()`'s `multi_state_strategy` can't be
  used to process multiple states in one call for this purpose.

## 2. The actual shape mismatch this spec resolves

MBR reranking (per tev_design idea-006 / the 260709 research prereg) needs the **average of k
independently-computed per-state scores**, not a score against an already-state-fused logit tensor.
Loading all k states into one call to anything that runs `ConditionalDecode`'s `stage_set.logit_transform`
fusion (e.g. `aminx.score()`'s `multi_state_strategy`) would produce exactly the wrong quantity — a
score against fused logits, not the average of independent per-state scores. These are mathematically
different, and using the fused quantity would just be re-scoring the same in-loop mechanism the research
question needs a genuine alternative to.

**Resolution:** encode each state separately (k small, explicit outer loop — this is a genuine semantic
requirement, not a batching gap: MBR needs independence across states, not fusion, so this axis is
correctly a loop rather than something to batch away) and, for each state, batch-decode **all** C
candidates in one call via `make_batched_conditional_logits_split_fn` (§1) — this is where xtrax's
arbitrary-cardinality batching actually applies, and applies correctly, since `make_encoding_conditional_logits_split_fn`
has no state-fusion machinery in it at all (§1) — the "identity at S=1" concern that would apply to
`aminx.score()`'s `multi_state_strategy` simply doesn't arise on this path, because this path never
touches `stage_set`/`logit_transform` in the first place.

## 3. New code — small, correctly scoped

The genuinely new logic, on top of the existing batched R×C infra (§1):

1. **A small NLL-from-logits reduction** applied to `batched_decode_fn`'s `(1, C, L, 21)` output (R=1,
   one state encoded at a time) — either `compute_pseudo_perplexity` after reshaping to its rigid
   4-leading-dim convention, or a direct ~4-line equivalent (§1's integration-detail note). Produces a
   `(C,)` per-candidate score for that state.
2. **`average_cross_state_scores(per_state_scores: Float[Array, "k candidates"]) -> Float[Array, "candidates"]`**
   — a small, pure-JAX reduction, genuinely `Stacked[S] -> Out[O]` (k states → one score per candidate).
   Simple elementwise mean over axis 0.
3. **`select_mbr_candidates(mean_scores, sequences, top_k=1) -> selected sequences/indices`** — selects
   the **lowest**-scoring candidates (lower NLL = better, per §1's scoring-direction note) via
   `jnp.argsort`/`jnp.argmin` over the candidate axis — **not** argmax; get the direction right, it is
   easy to invert by mistake. A distinct reduction from step 2 (over candidates, not states) — keep
   separate.

Composition (pseudocode — outer loop is over k states, small and explicit; the candidate axis is
batched via xtrax inside each iteration, not looped):

```python
def mbr_rerank(model, state_structures: list[StateStructure], candidate_sequences, prng_key, top_k=1):
  # StateStructure = (coords, mask, residue_index, chain_index) matching
  # make_batched_conditional_logits_split_fn's args — NOT an InferenceBundle.
  #
  # NOTE (self-caught during drafting): the UNWRAPPED decode_fn from
  # make_encoding_conditional_logits_split_fn is ALSO single-candidate-only (its own docstring example
  # calls it once per sequence, conditional_logits.py:173-174) — it is NOT batched_decode_fn's candidate
  # dispatch. Using the raw decode_fn here would silently reproduce the exact single-candidate mistake
  # this whole revision exists to fix. The batched wrapper below is what actually applies xtrax's
  # arbitrary-cardinality dispatch — must use batched_encode_fn/batched_decode_fn, not encode_fn/decode_fn.
  batched_encode_fn, batched_decode_fn = make_batched_conditional_logits_split_fn(model)
  single_replicate_key = prng_key[None]  # shape (1, ...) — R=1: one state encoded at a time (see §2)

  per_state_scores = []  # will hold k arrays of shape (C,)
  for state in state_structures:  # k independent encodes, genuinely not batched (see §2)
    encodings = batched_encode_fn(
      state.coords, state.mask, state.residue_index, state.chain_index, single_replicate_key,
    )  # R=1
    logits = batched_decode_fn(encodings, candidate_sequences)  # (1, C, L, 21) — C batched via xtrax
    scores = nll_from_logits(logits[0], candidate_sequences)  # (C,) — squeeze R=1, then §3 item 1
    per_state_scores.append(scores)
  mean_score = average_cross_state_scores(jnp.stack(per_state_scores, axis=0))  # (C,)
  return select_mbr_candidates(mean_score, candidate_sequences, top_k=top_k)
```

This is the actual design, not a reference-then-optimize split — `batched_decode_fn` already handles
arbitrary C via xtrax's `BatchPlanner` on `N_CANDIDATES`, so there is no separate "slow version to
optimize later" for the candidate axis. §6's acceptance criteria are written against this version
directly. **Implementer note:** verify `batched_encode_fn` with `R=1` doesn't add meaningful overhead
vs. a hypothetical unbatched single-state encode path — it should reduce to `Vmap` over a
length-1 axis (a no-op wrapper, per `BatchPlanner`'s own `cardinality <= batch_size → Vmap` rule,
`xtrax` skill reference), but confirm this against a real run rather than assuming.

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

- **Given** k reference-state structures and a set of already-sampled candidate sequences (arbitrary
  count C) from a completed production run.
- **When** `mbr_rerank` is called with these inputs.
- **Then** it returns the same per-candidate mean score as k×C separate manual calls to `aminx.score()`
  (one call per state per candidate, S=1 each, `multi_state_temperature=1.0`, `state_weights=None`)
  averaged by hand — i.e. a unit test must assert numerical equivalence between the xtrax-batched
  `batched_decode_fn`+NLL path and this independent single-candidate reference path, not just "runs
  without error." This cross-checks two structurally different code paths (batched dispatch vs.
  `aminx.score()`'s own encode/decode/score) against each other — a stronger check than testing either
  path in isolation.

- **Given** the same candidate set scored via `batched_decode_fn` with `C` candidates in one call vs.
  `C` separate calls each with a single candidate.
- **When** results are compared.
- **Then** per-candidate scores are identical regardless of batch size (a real test of "arbitrary
  cardinality," not just "runs for one specific C") — run this at at least two different C values
  (e.g. C below and above whatever `default_batch_size`/memory threshold triggers `BatchPlanner`'s
  Vmap→SafeMap demotion) to confirm the SafeMap-chunked path also gives identical results, not just Vmap.

- **Given** `average_cross_state_scores` called with `k=1` (a single state).
- **When** compared against that one state's own per-candidate scores directly.
- **Then** results are identical (true no-op at k=1 — trivially true for an elementwise mean over a
  length-1 axis, but assert it as a test rather than assuming).

- **Given** a candidate sequence with a known, hand-computed NLL against a single state, and a second
  candidate with a deliberately worse (higher-NLL) sequence for the same state.
- **When** `select_mbr_candidates` is called on both candidates' scores with `top_k=1`.
- **Then** it returns the FIRST (lower-NLL) candidate — i.e. explicitly test the selection direction is
  lower-is-better, not higher-is-better. This is the single easiest mistake to make silently (an
  accidental `argmax` instead of `argmin` produces a runnable, plausible-looking, wrong result with no
  error).

- **Given** the finished `mbr_rerank` module.
- **When** grepped for imports.
- **Then** it imports `make_batched_conditional_logits_split_fn` (not the raw, single-candidate
  `encode_fn`/`decode_fn` pair — see §3's self-caught note) — and does **not** touch `types/stages.py`'s
  `axis_boundaries` field or `inference/decode/autoregressive.py`'s live AR-loop fusion call site.

## 7. Assumptions

- The k reference states (1LVB, 1LVM, reac1-reac, reac2-reac per tev_design's necklace campaign) are
  already available as the raw `(coords, mask, residue_index, chain_index)` arrays
  `make_batched_conditional_logits_split_fn`'s `batched_encode_fn` takes directly (per §3's pseudocode —
  **not** an `InferenceBundle`) — this spec does not address how tev_design constructs those; that's
  existing, working infra (`canonical_bundle.npz` per tev_design's `build_canonical_bundle.py`), which
  will need a small extraction step to pull out the per-array fields this function wants.
- `BatchPlanner`'s Vmap/SafeMap strategy selection for the candidate axis is assumed to require no
  special handling beyond passing the full candidate array — verified structurally (§1) but not yet
  run against a real large-C candidate set from an actual necklace production output; §6's second
  acceptance criterion closes this gap with a real test across the Vmap/SafeMap boundary.

## 8. TBDs (only genuinely open items remain — see §1/§2/§3 for what independent review resolved)

- Exact new module path/name (`aminx.sampling.mbr_consensus` vs. `aminx.host.mbr_rerank` vs. other) —
  implementer's call, follow the nearest sibling module's naming convention at time of implementation.
  This is the only remaining item that's genuinely fine to leave to implementation time — it has no
  correctness or scope implications.

**Resolved, not left open** (two independent review passes flagged these as too important to leave as
TBDs; both are now decided in this document rather than deferred):

- **Ship location: `mbr_rerank` ships in aminx**, not as a tev_design-side script. Decided, not merely
  recommended — matches the project's stated preference for composing in aminx, and the k-state-average-
  and-rerank pattern is reusable beyond this one caller (tev_design's necklace campaign already runs
  against a fixed k=4 state set multiple analyses reuse).
- **Candidate-batching question** — resolved in §1/§3, and corrected a second time after the first
  resolution was itself wrong (see Provenance, third round): `aminx.score()` takes exactly one candidate
  per call and is therefore NOT the design's primary path; `make_batched_conditional_logits_split_fn`'s
  `batched_decode_fn` already batches an arbitrary number of candidates via xtrax's `BatchPlanner` on
  `N_CANDIDATES`, and IS the primary path. There is no separate "batched follow-up" — the batched
  version is the only version this spec describes.
