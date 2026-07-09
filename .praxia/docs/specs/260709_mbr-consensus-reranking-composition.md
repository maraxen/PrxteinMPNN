---
title: MBR post-hoc consensus reranking — S×C (state × candidate) composition via xtrax AxisSpec/BatchPlanner on both axes, mirroring the existing R×C precedent, not a new axis_boundaries Fuse and not a Python loop over either axis
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
adopted for, and it was already documented in §1 of this very file before being wrongly demoted.

**Fourth round (direct correction, project maintainer):** the third round's fix fixed the candidate axis
but left a raw Python `for state in state_structures:` loop over the k reference states, justified as
"a genuine semantic requirement, not a batching gap." That justification doesn't hold up: even where an
axis genuinely needs independent (non-fused) results per element, that's an argument for *not fusing*,
not an argument for *bypassing xtrax's dispatch layer with a hand-written loop*. Checked directly:
`tiling/axes.py:44-51` already registers `N_STATES` (`cardinality=64, default_batch_size=1,
heterogeneous=True` — states genuinely have different shapes, e.g. 1LVB/1LVM carry extra
water/hetero chains reac1/reac2 don't, so a true `jax.vmap` across raw states isn't even mathematically
possible without padding) — this is an **existing, registered axis**, and `_plan_axis_strategy`
(`conditional_logits.py:287-334`, the exact function `batched_decode_fn` already uses for `N_CANDIDATES`)
is generic to any `AxisSpec`, not candidate-specific. §3 is rewritten again to dispatch the state axis
through this existing registry entry via the same `BatchPlanner`/`make_axis_dispatch_via_xtrax` idiom
used for candidates — not a bare Python loop — even though the practical result (`SafeMap(tile=1)`,
given `N_STATES`'s own heterogeneity-driven `default_batch_size=1`) executes close to sequentially. The
difference is that it goes *through* xtrax's registered, memory-budget-aware, EDA-inspectable dispatch
machinery — consistent with every other axis in this codebase — rather than around it.

## 1. What already exists (verified, reusable as-is)

- **`N_STATES`** (`tiling/axes.py:44-51`) — already registered: `cardinality=64, default_batch_size=1,
  tile_granularity=1, heterogeneous=True`. The comment above it ("Shapes vary across states") and the
  `default_batch_size=1` are the codebase's own acknowledgment that this axis cannot be naively `vmap`'d
  — a `BatchPlanner` decision for a heterogeneous axis correctly demotes toward `SafeMap` with a small
  tile (here, effectively 1) rather than a true batched Vmap. **This design dispatches the state axis
  through this existing registration**, not around it.
- **`N_CANDIDATES`** (`tiling/axes.py:131-137`) — already registered, homogeneous (all candidates for a
  given bead share sequence length `L`), consumed today by `make_batched_conditional_logits_split_fn`'s
  `batched_decode_fn` (below).
- **`_plan_axis_strategy`** (`sampling/conditional_logits.py:287-334`) — resolves any `AxisSpec` (not
  candidate-specific) to `Vmap()`/`SafeMap(tile=N)` via `BatchPlanner`, with an explicit docstring
  warning: *"never hand-roll `jax.vmap`/`lax.map` chunking here."* This is genuinely axis-generic — the
  same function this design reuses for `N_STATES` is the one `batched_decode_fn` already uses for
  `N_CANDIDATES`.
- **`make_axis_dispatch_via_xtrax`** (`tiling/dispatch.py:146-...`) — the dispatch-to-iterator layer;
  explicitly rejects `Scan` on a `heterogeneous_axes` member (not relevant here — `_plan_axis_strategy`
  only ever selects `Vmap`/`SafeMap`, both of which remain valid for a heterogeneous axis via a
  small-enough tile size) and rejects `DedupGather` outright.
- **`make_batched_conditional_logits_split_fn`** (`sampling/conditional_logits.py:337-439`) — the
  existing R×C (replicate × candidate) precedent this design's new S×C (state × candidate) function
  mirrors structurally. `batched_encode_fn`/`batched_decode_fn` batch an arbitrary number of candidates
  via `N_CANDIDATES` exactly as described above. `batched_decode_fn(encodings, candidate_sequences,
  ar_mask=None)` returns `(R, C, L, 21)` logits (`conditional_logits.py:437`) — raw logits, not a score;
  a scoring step is applied on top (see §3).
- **`make_encoding_conditional_logits_split_fn`** (`conditional_logits.py:145-177`) — the underlying
  single-call encode_fn/decode_fn pair `make_batched_conditional_logits_split_fn` wraps for R. This
  pair is generic to "one structure in, encode once, decode many sequences against it" — it has no
  R-specific or state-specific logic baked in, which is exactly why it's reusable as the inner kernel
  for a new S-axis (state) dispatcher, mirroring how it's already reused as the inner kernel for the
  R-axis (replicate) dispatcher.
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
score against fused logits, not the average of independent per-state scores; worse, `_apply_logit_transform`'s
output feeds directly into `_apply_tie_group_fuse` (`inference/decode/conditional.py:154-155`), which
almost certainly assumes already-reduced (non-stacked) logits — passing an identity/no-op transform to
get unfused per-state logits out of `ConditionalDecode` would risk breaking that downstream step in a
non-obvious way. `ConditionalDecode` is the wrong tool for this regardless of how it's parameterized.

**Resolution — dispatch BOTH axes through xtrax, none through a raw Python loop:**
- **Candidate axis (`N_CANDIDATES`, homogeneous)** — already correctly dispatched by
  `make_batched_conditional_logits_split_fn`'s `batched_decode_fn` via `BatchPlanner` (Vmap/SafeMap,
  arbitrary cardinality). Reused as-is.
- **State axis (`N_STATES`, heterogeneous)** — dispatched via the *same* `_plan_axis_strategy` +
  `make_axis_dispatch_via_xtrax` idiom, applied to the already-registered `N_STATES` `AxisSpec` instead
  of writing a new one. Because `N_STATES.heterogeneous=True` and `default_batch_size=1`,
  `BatchPlanner` correctly resolves this to `SafeMap(tile=1)` — practically sequential, but *through*
  xtrax's registered dispatch machinery (memory-budget-aware, EDA-inspectable, consistent with how every
  other axis in this codebase is handled), not a bespoke loop that bypasses it. If states ever become
  more uniform in practice (unlikely — reference states are fundamentally different structures, not
  noise replicates of one structure) `N_STATES`'s registry entry could later gain `bucket_boundaries`
  for padded Vmap batching without changing any call site — that flexibility is exactly what dispatching
  through xtrax buys over a hand-rolled loop, even though today's execution shape looks similar either way.

## 3. New code

**A new split-fn constructor, mirroring the existing R×C precedent structurally** — this is the one
piece of genuinely new aminx code beyond small reduction/selection helpers, and it earns its place by
directly mirroring an established, proven pattern rather than inventing a new one:

- **`make_multistate_candidate_logits_split_fn(model)`** (new; name is a TBD — mirror
  `make_batched_conditional_logits_split_fn`'s naming convention exactly, swapping "batched" for
  whatever term best signals "state axis, not replicate axis," at implementation time) — structurally
  identical to `make_batched_conditional_logits_split_fn` (`conditional_logits.py:337-439`): same
  `encode_fn`/`decode_fn` inner kernel from `make_encoding_conditional_logits_split_fn`, same
  `_plan_axis_strategy` + `make_axis_dispatch_via_xtrax` composition — the **only** change is which
  `AxisSpec` drives the outer axis (`N_STATES` instead of `N_REPLICATES`) and what the outer input
  represents (k distinct reference structures, not R noise-replicate keys of one structure). Returns
  `(batched_encode_fn_over_states, batched_decode_fn_over_states_and_candidates)` with the same
  R×C-style call shape, just S×C.

Given this, the remaining genuinely new logic is small:

1. **A small NLL-from-logits reduction** applied to the new function's `(S, C, L, 21)` output —
   either `compute_pseudo_perplexity` after reshaping to its rigid 4-leading-dim convention
   (`[batch, samples, noise, temp, seq_len, 21]`, `host/logit_aggregation.py:14-53`), or a direct ~4-line
   equivalent (`-jnp.sum(one_hot(seq) * log_softmax(logits, axis=-1), axis=(-1,-2))`, axis-generic,
   doesn't need the rigid wrapper). Produces `(S, C)` per-candidate-per-state scores in one shot — no
   loop needed even for this step, since the state axis is already materialized as a real array
   dimension by the SafeMap(tile=1) dispatch, not iterated in Python.
2. **`average_cross_state_scores(per_state_scores: Float[Array, "S C"]) -> Float[Array, "C"]`** — a
   small, pure-JAX reduction, genuinely `Stacked[S] -> Out[O]` (states → one score per candidate).
   Simple elementwise mean over axis 0.
3. **`select_mbr_candidates(mean_scores, sequences, top_k=1) -> selected sequences/indices`** — selects
   the **lowest**-scoring candidates (lower NLL = better, per §1's scoring-direction note) via
   `jnp.argsort`/`jnp.argmin` over the candidate axis — **not** argmax; get the direction right, it is
   easy to invert by mistake.

Composition (pseudocode — no Python loop over either axis; both dispatch through xtrax):

```python
def mbr_rerank(model, state_structures, candidate_sequences, replicate_keys, top_k=1):
  # state_structures: stacked (coords, mask, residue_index, chain_index) over the S state axis --
  # matches N_STATES semantics (heterogeneous shapes allowed; this is NOT an InferenceBundle).
  # replicate_keys here plays the same role batched_encode_fn's replicate_keys does today, but keyed
  # over states, not backbone-noise replicates -- naming TBD at implementation time (§8).
  batched_encode_fn, batched_decode_fn = make_multistate_candidate_logits_split_fn(model)  # new, mirrors R×C

  encodings = batched_encode_fn(state_structures, replicate_keys)  # dispatched over N_STATES via xtrax
  logits = batched_decode_fn(encodings, candidate_sequences)  # (S, C, L, 21) -- C dispatched via xtrax too
  per_state_scores = nll_from_logits(logits, candidate_sequences)  # (S, C) -- §3 item 1, no loop
  mean_score = average_cross_state_scores(per_state_scores)  # (C,)
  return select_mbr_candidates(mean_score, candidate_sequences, top_k=top_k)
```

Both axes go through `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax` — the state axis resolves to
`SafeMap(tile=1)` (heterogeneous), the candidate axis resolves to `Vmap`/`SafeMap(tile=N)` depending on
`C` and the memory budget, exactly as `batched_decode_fn` already does today. §6's acceptance criteria
are written against this version. **Implementer note:** confirm at implementation time that
`make_encoding_conditional_logits_split_fn`'s `encode_fn` doesn't assume anything replicate-specific
(e.g. a shared `backbone_noise` scalar across the batched axis) that wouldn't make sense across
genuinely different structures — it shouldn't, since it's documented as generic to "one structure in,"
but this needs eyes-on verification, not assumption, before treating the mirror as exact.

## 4. Where this lives — NOT `StageSet.axis_boundaries`

MBR reranking's actual input is **already-sampled sequences from a completed production run** (e.g.
tev_design's necklace P2 job, stored zarr output) being re-scored against the k reference states — this
is inherently a **post-hoc batch utility**, not a per-decode-call `StageSet` stage. `axis_boundaries`
exists to extend a *live* decode/`StageSet` pipeline; MBR reranking runs entirely after sampling is
already complete and files are already written.

**Decision:** ship `mbr_rerank` (composition function) plus the new
`make_multistate_candidate_logits_split_fn` (§3) as standalone functions (new module, e.g.
`aminx.sampling.mbr_consensus` — implementer's choice, mirror the nearest existing sibling module's
layout), reusing `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax` and the already-registered
`N_STATES`/`N_CANDIDATES` `AxisSpec`s for both axes, mirroring the R×C pattern rather than inventing a
new dispatch idiom. `StageSet.axis_boundaries` remains unpopulated by this work — it is a real, distinct
extension point for a *future* live-decode use case, not this one. Do not force this task to be that use
case's first occupant merely because the slot exists and is otherwise idle.

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

- **Given** k reference-state structures with GENUINELY DIFFERENT shapes (e.g. 1LVB/1LVM's extra
  water/hetero chains vs. reac1/reac2's absence of them — the real heterogeneity already confirmed to
  exist across this project's actual reference states), and a set of already-sampled candidate sequences
  (arbitrary count C).
- **When** `mbr_rerank` is called with these inputs.
- **Then** it runs to completion and produces correct per-state scores for every state — this is the
  test that actually exercises `N_STATES.heterogeneous=True` and confirms `SafeMap(tile=1)` handles
  varying per-state shapes correctly, not just same-shape states (a test using k copies of one state
  would not catch a heterogeneity-handling bug).

- **Given** k reference-state structures and a set of already-sampled candidate sequences from a
  completed production run.
- **When** `mbr_rerank`'s output is compared against k×C separate manual calls to `aminx.score()` (one
  call per state per candidate, S=1 each, `multi_state_temperature=1.0`, `state_weights=None`) averaged
  by hand.
- **Then** results are numerically equivalent — i.e. a unit test must assert this, not just "runs
  without error." This cross-checks two structurally different code paths (the new xtrax-dispatched
  S×C composition vs. `aminx.score()`'s own single-call encode/decode/score) against each other — a
  stronger check than testing either path in isolation.

- **Given** the same candidate set scored via the new S×C composition with `C` candidates in one call
  vs. `C` separate calls each with a single candidate.
- **When** results are compared.
- **Then** per-candidate scores are identical regardless of batch size (a real test of "arbitrary
  cardinality" on the candidate axis, not just "runs for one specific C") — run this at at least two
  different C values (e.g. C below and above whatever `default_batch_size`/memory threshold triggers
  `BatchPlanner`'s Vmap→SafeMap demotion) to confirm the SafeMap-chunked path also gives identical
  results, not just Vmap.

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

- **Given** the finished `mbr_rerank`/`make_multistate_candidate_logits_split_fn` module.
- **When** grepped for imports and for the literal string `for ` at the top level of any function body.
- **Then** it imports `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax` and the `N_STATES`/`N_CANDIDATES`
  `AxisSpec`s — contains **no** hand-written `for`/`while` loop over either the state or candidate axis
  anywhere in the implementation (the state axis's practical sequentiality must come from
  `SafeMap(tile=1)`, not a Python loop that bypasses xtrax) — and does **not** touch `types/stages.py`'s
  `axis_boundaries` field or `inference/decode/autoregressive.py`'s live AR-loop fusion call site.

## 7. Assumptions

- The k reference states (1LVB, 1LVM, reac1-reac, reac2-reac per tev_design's necklace campaign) are
  already available as the raw, per-state `(coords, mask, residue_index, chain_index)` arrays the new
  `make_multistate_candidate_logits_split_fn`'s `batched_encode_fn`-equivalent takes, stacked/collected
  over the `N_STATES` axis (heterogeneous shapes allowed — **not** an `InferenceBundle`, and **not**
  required to be uniform-shape the way `N_REPLICATES`' backbone-noise keys are) — this spec does not
  address how tev_design constructs those; that's existing, working infra (`canonical_bundle.npz` per
  tev_design's `build_canonical_bundle.py`), which will need a small extraction step to pull out the
  per-array, per-state fields this function wants.
- `BatchPlanner`'s Vmap/SafeMap strategy selection for the candidate axis is assumed to require no
  special handling beyond passing the full candidate array — verified structurally (§1) but not yet
  run against a real large-C candidate set from an actual necklace production output; §6's acceptance
  criteria close this gap with real tests across the Vmap/SafeMap boundary AND across genuinely
  heterogeneous state shapes.
- `_plan_axis_strategy`'s generic (`AxisSpec`-parameterized, not candidate-specific) design is assumed to
  work correctly when applied to `N_STATES` the same way it already works for `N_CANDIDATES`/`N_REPLICATES`
  — the function's own signature and docstring support this (§1), but it has not yet been exercised
  against `N_STATES` in any existing test; this is the one point in this design that most needs a real
  run before being trusted, since it's the newest application of an old pattern.

## 8. TBDs (only genuinely open items remain — see §1/§2/§3 for what independent review resolved)

- Exact new module path/name (`aminx.sampling.mbr_consensus` vs. `aminx.host.mbr_rerank` vs. other) —
  implementer's call, follow the nearest sibling module's naming convention at time of implementation.
- Exact name for `make_multistate_candidate_logits_split_fn` (placeholder name used in §3) and for its
  "replicate_keys-equivalent-but-really-states" parameter — should read clearly as "states," not reuse
  "replicate" terminology, per this project's descriptive-naming convention; implementer's call at
  write time, not a correctness question.
  Neither of these has correctness or scope implications — genuinely fine to leave to implementation time.

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
- **State-axis dispatch question** — corrected a further time (Provenance, fourth round): a raw Python
  loop over states was NOT acceptable even though states genuinely need independent (non-fused) scores —
  independence is an argument against fusion, not an argument for bypassing xtrax's dispatch layer. The
  state axis is dispatched through the already-registered `N_STATES` `AxisSpec` (heterogeneous,
  `default_batch_size=1`) via the same `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax` idiom already
  used for candidates — resolving to `SafeMap(tile=1)`, not a hand-written loop. There is no Python `for`
  loop over either axis anywhere in this design.
