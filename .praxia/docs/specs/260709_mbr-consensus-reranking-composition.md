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

## 1. What already exists (verified, reusable as-is) — and it's more than the first correction found

- **`aminx.score()`** (fully public, top-level export — `aminx/__init__.py:24`, implemented at
  `src/aminx/scoring/score.py:157-230`) already does encode→decode→score in ONE call and returns
  `(masked_average_score, logits, decoding_order)` (`score.py:203-204`) where `masked_average_score` is
  the **masked-average negative log-likelihood** (`_nll_from_logits`, `score.py:29-51`; docstring at
  line 44: "Scalar NLL... masked and averaged") — **lower is better** (a more likely/better-fitting
  sequence has lower NLL). This is the same "lower = better" direction as `compute_pseudo_perplexity`'s
  `exp(NLL)` (monotonic in NLL), so the two are compatible in ranking terms even though `score()`'s raw
  NLL and `compute_pseudo_perplexity`'s exponentiated value are not numerically identical.
- **`aminx.score()` takes exactly ONE candidate sequence per call, not a batch** (verified,
  independent-review finding, not assumed): `make_score_fn`'s inner scoring path computes
  `L = sequence.shape[0]` (`score.py:111`) and `_nll_from_logits` has no batch dimension anywhere in its
  signature — `sequence` must be a single 1-D array of shape `(L,)`. This resolves what an earlier draft
  of this spec left as an open TBD, and **that earlier draft's own pseudocode was wrong as a result** —
  see §3.
- **`multi_state_strategy`/`state_weights`** are accepted directly by `score()` (`score.py:169-171`); at
  a single state (S=1, one call per reference state — see §2) whether this is truly a no-op depends on
  the strategy and its parameters, **not universally true** — see §2's corrected claim.
- Underneath `aminx.score()` (for context, not code you need to call directly):
  `aminx.inference.score_conditional.kernel` (`score_conditional.py:164-180`) — `encode` + `score_from_encoding`
  in one call ("byte-identical to original implementation" per an inline comment at line 177, not a
  docstring); `ConditionalDecode` fuses per-state logits via `self._apply_logit_transform(logits_stack,
  stage_set, ...)` (`inference/decode/conditional.py:154`) before the caller ever sees them — this is
  what makes calling with more than one state at once the WRONG shape for MBR (see §2).
- **`make_batched_conditional_logits_split_fn`** (`sampling/conditional_logits.py:337-439`) — the
  established R×C (replicate × candidate) dispatch pattern. Given `score()` is one-candidate-per-call
  (above), looping over C candidates in Python for a large candidate pool may be slow; this is the
  throughput escape hatch. **Note:** `batched_decode_fn` returns raw logits `(R, C, L, 21)` only
  (`conditional_logits.py:437`), not a score — if this path is used, something equivalent to
  `compute_pseudo_perplexity`/`_nll_from_logits` is still needed on its output. §1's earlier framing
  ("no need for `compute_pseudo_perplexity` at all") only holds for the simple per-candidate-loop
  version below; it does not hold if throughput later requires the batched path.

## 2. The actual shape mismatch this spec resolves

MBR reranking (per tev_design idea-006 / the 260709 research prereg) needs the **average of k
independently-computed per-state scores**, not a score against an already-state-fused logit tensor.
Calling `aminx.score()` once with all k states loaded (via its `multi_state_strategy`) would produce
exactly the wrong quantity — a score against fused logits, not the average of per-state scores. These
are mathematically different, and using the fused quantity would just be re-scoring the same in-loop
mechanism the research question needs a genuine alternative to.

**Resolution:** call `aminx.score()` once per state (S=1 each call, one reference-state structure at a
time).

**Correction (independent review finding — the original claim here was too strong):** "any strategy is
identity at S=1" is **not universally true**. Checked directly against `src/aminx/inference/logits.py`:
- `ArithmeticMeanLogits` — identity at S=1 regardless of weight (log-space cancellation holds
  algebraically for any weight value).
- `GeometricMeanLogits` (line ~175) divides by `self.temperature` — at S=1 this is `per_state /
  temperature`, **identity only if `multi_state_temperature=1.0`**.
- `ProductOfProbabilities` (line ~244) multiplies by weight — **identity only if `state_weights=None`**
  (the default per `make_stage_set`, `logits.py:418`) or all weights equal 1.
So the precondition is: **use default `multi_state_temperature=1.0` and `state_weights=None`** when
calling `score()` per-state for MBR — under those defaults (which are also aminx's own defaults, so no
special call-site handling is needed), all three strategies are true no-ops at S=1. This precondition
must be stated at the call site (e.g. an assertion or a comment), not left implicit.

## 3. New code — small, correctly scoped

Given §1/§2's corrected findings (`score()` is one-candidate-per-call; NLL direction is lower-is-better;
identity-at-S=1 needs default temperature/weights), the genuinely new logic is:

1. **`average_cross_state_scores(per_state_scores: Float[Array, "k candidates"]) -> Float[Array, "candidates"]`**
   — a small, pure-JAX reduction, genuinely `Stacked[S] -> Out[O]` (k states → one score per candidate).
   Simple elementwise mean over axis 0.
2. **`select_mbr_candidates(mean_scores, sequences, top_k=1) -> selected sequences/indices`** — this
   selects the **lowest**-scoring candidates (lower NLL = better, per §1) via `jnp.argsort`/`jnp.argmin`
   over the candidate axis — **not** argmax; get the direction right, it is easy to invert by mistake.
   A second, distinct reduction (over candidates, not states) — keep separate from step 1.

Composition (pseudocode, corrected to match `score()`'s real one-candidate-per-call signature — this
now nests two loops: states (outer, small, k~4) × candidates (inner, potentially large C)):

```python
def mbr_rerank(model, state_structures: list[StateStructure], candidate_sequences, prng_key, top_k=1):
  # StateStructure = (coords, mask, residue_index, chain_index) tuple/dataclass matching
  # aminx.score()'s positional args — NOT an InferenceBundle; score() takes raw arrays, not a bundle.
  # multi_state_temperature=1.0, state_weights=None (both aminx defaults) are REQUIRED here per §2's
  # identity-at-S=1 precondition — do not let a caller override them for this call path.
  per_state_scores = []  # will hold k arrays of shape (C,)
  for state in state_structures:  # k independent calls, S=1 each
    per_candidate_scores = []
    for seq in candidate_sequences:  # score() takes exactly ONE sequence per call (verified, §1)
      avg_score, _logits, _order = aminx.score(
        prng_key, model, state.coords, state.mask, state.residue_index, state.chain_index,
        sequence=seq, multi_state_temperature=1.0, state_weights=None,
      )
      per_candidate_scores.append(avg_score)
    per_state_scores.append(jnp.stack(per_candidate_scores))
  mean_score = average_cross_state_scores(jnp.stack(per_state_scores, axis=0))  # (C,)
  return select_mbr_candidates(mean_score, candidate_sequences, top_k=top_k)
```

This is a correct-but-likely-slow (k × C sequential `score()` calls, no batching) reference
implementation — **build and numerically validate this version first** (§6's acceptance criteria are
written against it), then optimize via `make_batched_conditional_logits_split_fn`'s candidate-axis
batching as a follow-up if C is large enough to matter, remembering §1's note that the batched path
needs its own NLL computation (`batched_decode_fn` returns raw logits, not a score).

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

- **Given** `average_cross_state_scores` called with `k=1` (a single state), `multi_state_temperature=1.0`,
  `state_weights=None`.
- **When** compared against calling `aminx.score()` directly on that one state with the same defaults.
- **Then** results are identical for `ArithmeticMeanLogits`, `GeometricMeanLogits`, and
  `ProductOfProbabilities` — this specific claim must be tested for all three strategies individually
  (not asserted by mathematical reasoning alone, and not assumed to hold at non-default temperature/weights).

- **Given** a candidate sequence with a known, hand-computed NLL against a single state, and a second
  candidate with a deliberately worse (higher-NLL) sequence for the same state.
- **When** `select_mbr_candidates` is called on both candidates' scores with `top_k=1`.
- **Then** it returns the FIRST (lower-NLL) candidate — i.e. explicitly test the selection direction is
  lower-is-better, not higher-is-better. This is the single easiest mistake to make silently (an
  accidental `argmax` instead of `argmin` produces a runnable, plausible-looking, wrong result with no
  error).

- **Given** the finished `mbr_rerank` module.
- **When** grepped for imports.
- **Then** it imports the public `aminx.score()` — and does **not** touch `types/stages.py`'s
  `axis_boundaries` field or `inference/decode/autoregressive.py`'s live AR-loop fusion call site.

## 7. Assumptions

- The k reference states (1LVB, 1LVM, reac1-reac, reac2-reac per tev_design's necklace campaign) are
  already available as the raw `(coords, mask, residue_index, chain_index)` arrays `score()` takes
  directly (per §3's corrected pseudocode — **not** an `InferenceBundle`; `score()`'s signature takes
  raw arrays, confirmed at `score.py:157-176`) — this spec does not address how tev_design constructs
  those; that's existing, working infra (`canonical_bundle.npz` per tev_design's `build_canonical_bundle.py`),
  which will need a small extraction step to pull out the per-array fields `score()` wants.

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
- **Candidate-batching question** — resolved in §1/§3: `score()` takes exactly one candidate per call;
  the reference implementation loops (correct-but-possibly-slow); batched throughput via
  `conditional_logits.py` is an explicit, separate follow-up, not part of this pass.
