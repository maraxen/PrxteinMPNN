---
title: MBR post-hoc consensus reranking — states via genuine jax.vmap over tev_design's already-canonicalized (padded, uniform-shape) bundle reusing existing ConditionalDecode/_VmapEncode machinery, candidates via xtrax AxisSpec/BatchPlanner on N_CANDIDATES; not a new axis_boundaries Fuse, not a Python loop over either axis, not SafeMap over a genuinely ragged array (impossible)
status: Draft, fifth revision (supersedes a factually-incorrect contemplex brainstorm artifact — see "Provenance" below for the full correction history)
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

**Fifth round (independent Opus review, loaded `/using-jax` and `/using-xtrax` first):** the fourth
round's fix is itself infeasible, and infeasible in a way that invalidates the entire mechanism, not
just a detail. `SafeMap`/`jax.vmap` (`src/aminx/utils/safe_map.py:44-52`) require a **stacked array with
a uniform leading axis** — they slice a rectangular array; they do not ragged-iterate Python objects.
States with genuinely different shapes (the fourth round's own stated premise) **cannot be assembled
into such an array at all** — `SafeMap(tile=1)` was never going to work regardless of tile size. Worse:
the R×C precedent this design claimed to mirror is not actually analogous — `batched_encode_fn`
(`conditional_logits.py:378-405`) maps over `replicate_keys` (a homogeneous key array) with the single
structure **closed over as a constant**; it never varies the structure, so heterogeneity never arises
there. "Mirror R×C, swap the axis" was structurally unsound from the start for an axis over
*different structures*, not repeated draws of one structure.

The actual fix, found by checking how aminx **already** solves this elsewhere rather than inventing a
new mechanism: `GeometryBundle.coords` is `[S, L, 4, 3]` with a **static, padded** `n_canonical`
(`types/bundles.py:57-63`); `inference/encode.py`'s `_VmapEncode` already vmaps over S on this padded,
uniform-shape stack (genuinely, not SafeMap-over-ragged); and — checked directly during this
correction — **tev_design's own `build_canonical_bundle.py:28` already sets `N_CANONICAL = 214` and pads
every reference state's arrays to that fixed size before they ever reach aminx**
(`can_coords = np.zeros((N_CANONICAL, N_ATOM37, 3), ...)`, line 237). So the states this design actually
receives are **not** ragged by the time they arrive — `N_STATES`'s `heterogeneous=True` in the tiling
registry is a conservative, general declaration covering the axis abstractly (some hypothetical future
caller might not canonicalize first), not a fact about this concrete input. A genuine `jax.vmap` over
states is legitimate here, reusing aminx's own existing, tested encode machinery — no SafeMap, no
padding logic to invent, no heterogeneity problem to solve, because tev_design already solved it
upstream. §1–§3 are rewritten a third time around this.

One more thing this round found, reusable rather than needing new invention:
`ConditionalDecode.__call__` (`inference/decode/conditional.py:79-155`) already computes
`logits_stack = _project_logits(self.model, decoded)` (line 150) — genuine **per-state, unfused**
logits via `jax.vmap` over the padded bundle — immediately BEFORE its two fusion calls
(`_apply_logit_transform`, `_apply_tie_group_fuse`, lines 154-155). This is exactly the "independent
per-state result computed via real vmap, no python loop, no fusion" this design needs, and it's an
intermediate value inside existing, working code, built from pure, explicitly-vmappable helpers
(`_decode_one_step`, `_project_logits`, `inference/decode/_kernel.py:16-108`, both docstring-documented
as "Can be vmapped over states"). No need to hack `stage_set` with an identity transform (which risked
breaking `_apply_tie_group_fuse`, per the second-round finding) — a small new function that mirrors
`ConditionalDecode.__call__` up to and including the `logits_stack` line, and simply doesn't call the
two fusion lines, sidesteps that risk entirely by construction.

Minor, non-blocking observation from this round, worth recording so it isn't lost: `N_STATES.name` is
`"n_states"`, but `make_axis_dispatch_via_xtrax`'s heterogeneity guard checks against the literal string
`"state"` (`tiling/dispatch.py`) — a name mismatch that means the guard doesn't actually fire for this
axis. Non-fatal today (`_plan_axis_strategy` never emits `Scan`, the only strategy that guard exists to
reject), but worth fixing independently of this spec so the safety check isn't silently dead for the one
axis most likely to need it.

## 1. What already exists (verified, reusable as-is)

- **`build_canonical_bundle.py:28`** (tev_design side) — `N_CANONICAL = 214`. Every reference state's
  arrays are padded to this fixed size (`can_coords = np.zeros((N_CANONICAL, N_ATOM37, 3), ...)`, line
  237) before they reach aminx. **This is what makes the state axis genuinely uniform-shape in
  practice** — the heterogeneity concern is already resolved upstream, by existing infra, not something
  this design needs to solve.
- **`inference/encode.py`'s `_VmapEncode`** (`encode.py:36-...`) — encodes S states via a genuine
  `jax.vmap`, already used in production by `sample_autoregressive` and `score_conditional` (per this
  module's own docstring, lines 1-9). Takes a padded, uniform-shape `InferenceBundle`/`GeometryBundle`.
  **Reused as-is for encoding** — no new encode code needed.
- **`ConditionalDecode.__call__`** (`inference/decode/conditional.py:79-155`) — computes
  `logits_stack = _project_logits(self.model, decoded)` (line 150) — genuine **per-state, unfused**
  logits via `jax.vmap` over the padded bundle (`decoded = self.state_iterator(per_state_fn,
  per_state_inputs, in_axes=0)`, line 148) — **immediately before** its two fusion calls
  (`_apply_logit_transform` line 154, `_apply_tie_group_fuse` line 155). This intermediate value is
  exactly "independent per-state result, computed via real vmap, no fusion" — the thing MBR needs.
- **`_decode_one_step`, `_project_logits`** (`inference/decode/_kernel.py:16-108`) — the pure helpers
  `ConditionalDecode` calls internally to produce `logits_stack`. Both docstring-documented as "Can be
  vmapped over states" — explicitly designed to be reused/composed this way, not private implementation
  detail that happens to be reusable.
- **`N_CANDIDATES`** (`tiling/axes.py:131-137`) — already registered, homogeneous (candidates for a
  given bead share sequence length `L`), consumed today by `make_batched_conditional_logits_split_fn`'s
  `batched_decode_fn`. Still the right mechanism for the candidate axis — this round's correction is
  about the state axis, not the candidate axis (rounds 3's fix there still holds).
- **`_plan_axis_strategy`** (`sampling/conditional_logits.py:287-334`) — resolves any `AxisSpec` to
  `Vmap()`/`SafeMap(tile=N)` via `BatchPlanner`. Reused for the candidate axis exactly as in round 3;
  **not** reused for the state axis in this round's design (see below — the state axis now uses a
  direct `jax.vmap` over the already-padded bundle, matching how `_VmapEncode`/`ConditionalDecode`
  already do it, rather than routing through `BatchPlanner` a second time for an axis that's already
  handled by existing, proven machinery).
- **`compute_pseudo_perplexity`** (`host/logit_aggregation.py:14-53`) — pure JAX, correct as-is.
  **Real integration detail, not a blocker:** its signature hardcodes a rigid 4-leading-dim shape
  convention (`[batch, samples, noise, temp, seq_len, 21]` in) inherited from a different pipeline (the
  stage3 sampling grid) — the new function's `(S, C, L, 21)` output (§3) has 2 leading dims, not 4;
  either reshape before calling it, or write the ~4-line NLL reduction directly (axis-generic, doesn't
  need the rigid wrapper). Flag this at implementation time rather than assuming direct compatibility.
- **`aminx.score()`** — documented for context only (establishes the lower-is-better scoring
  convention, §2), not used as this design's mechanism: it takes one candidate per call and internally
  runs `ConditionalDecode`'s fusion, neither of which fit here.

## 2. The actual mismatch this spec resolves, and why the state axis is genuine `vmap`, not `SafeMap`-over-ragged

MBR reranking needs the **average of k independently-computed per-state scores**, not a score against
an already-state-fused logit tensor. `ConditionalDecode` always fuses (`_apply_logit_transform` then
`_apply_tie_group_fuse`, `conditional.py:154-155`) — passing it an identity/no-op transform to suppress
fusion would risk breaking `_apply_tie_group_fuse`, which almost certainly assumes already-reduced input
(second-round finding). So this design does not use `ConditionalDecode` at all; it reuses the
pre-fusion `logits_stack` computation pattern directly (§1, §3).

**On the state axis specifically:** an earlier version of this correction assumed states are
irreducibly heterogeneous (different shapes) and tried to dispatch them through xtrax's `BatchPlanner`
as a `SafeMap(tile=1)`. That's impossible in principle — `SafeMap`/`jax.vmap` require a genuinely
uniform-shape stacked array to operate on at all; they cannot ragged-iterate differently-shaped Python
objects (fifth-round finding, verified against `utils/safe_map.py`). The actual resolution: **states
aren't ragged by the time they reach this design** — tev_design's `build_canonical_bundle.py` already
pads every state to `N_CANONICAL=214` (§1). `N_STATES.heterogeneous=True` in the tiling registry is a
conservative declaration covering the axis abstractly, for hypothetical callers who might not
canonicalize first — it is not a fact about this concrete, already-padded input. A genuine `jax.vmap`
over states is legitimate and is exactly what `_VmapEncode`/`ConditionalDecode` already do in production
today. This design reuses that same mechanism rather than inventing padding logic or heterogeneous-axis
handling that isn't needed.

## 3. New code

**A new function that mirrors `ConditionalDecode.__call__` up to its pre-fusion intermediate, then
stops** — this is the one piece of genuinely new aminx code beyond small reduction/selection helpers,
and it earns its place by reusing the exact existing pure helpers (`_decode_one_step`, `_project_logits`)
that already compute the value this design needs, rather than fusing further like `ConditionalDecode`
does:

- **`decode_states_unfused(model, encodings, sequence_oh, ar_mask, key, decode_step, state_iterator)`**
  (new; name is a TBD) — living alongside `ConditionalDecode` in `inference/decode/` (same package, so
  it can reuse `_decode_one_step`/`_project_logits` from `_kernel.py` the same way `ConditionalDecode`
  does). Body: the same `per_state_fn`/`state_iterator(...)`/`_project_logits(...)` sequence as
  `ConditionalDecode.__call__` lines 108-150 — **and returns `logits_stack` directly**, never calling
  `_apply_logit_transform`/`_apply_tie_group_fuse`. Output shape `(S, L, 21)` — genuine per-state,
  unfused logits, computed via the SAME `jax.vmap`-based `state_iterator` `ConditionalDecode` already
  uses (typically `VmapIterator`, matching `score_conditional.py`'s existing usage) — no new vmap
  machinery to write, just a shorter call chain than `ConditionalDecode`'s.
- **Candidate batching, reusing round 3's fix as-is**: wrap the above in the same `_plan_axis_strategy`
  + `make_axis_dispatch_via_xtrax` composition over `N_CANDIDATES` that `make_batched_conditional_logits_split_fn`'s
  `batched_decode_fn` already uses (`conditional_logits.py:412-437`) — i.e. dispatch
  `decode_states_unfused` once per candidate (or per candidate chunk, if `BatchPlanner` selects
  `SafeMap`), varying `sequence_oh`, reusing the SAME `encodings` (computed once via `_VmapEncode`/
  `score_conditional.encode()`, §1 — encoding doesn't depend on sequence, only on structure, so it's
  correctly computed once and reused across all C candidates, mirroring exactly how
  `make_encoding_conditional_logits_split_fn`'s "encode once, decode many" split already works for R).

Given this, the remaining genuinely new logic is small:

1. **A small NLL-from-logits reduction** applied to the resulting `(C, S, L, 21)` (or `(S, C, L, 21)`,
   ordering is an implementation choice) logits — either `compute_pseudo_perplexity` after reshaping to
   its rigid 4-leading-dim convention (`host/logit_aggregation.py:14-53`), or a direct ~4-line
   equivalent (axis-generic, doesn't need the rigid wrapper). Produces per-candidate-per-state scores in
   one shot — no loop, since both axes are already real array dimensions by this point.
2. **`average_cross_state_scores(per_state_scores: Float[Array, "S C"]) -> Float[Array, "C"]`** — a
   small, pure-JAX reduction, genuinely `Stacked[S] -> Out[O]` (states → one score per candidate).
   Simple elementwise mean over axis 0.
3. **`select_mbr_candidates(mean_scores, sequences, top_k=1) -> selected sequences/indices`** — selects
   the **lowest**-scoring candidates (lower NLL = better, per §1's scoring-direction note) via
   `jnp.argsort`/`jnp.argmin` over the candidate axis — **not** argmax; get the direction right, it is
   easy to invert by mistake.

Composition (pseudocode — no Python loop over either axis; states via direct `jax.vmap` reusing existing
production machinery, candidates via xtrax `BatchPlanner`):

```python
def mbr_rerank(model, canonical_bundle, candidate_sequences, prng_key, top_k=1):
  # canonical_bundle: the padded, N_CANONICAL=214-uniform multi-state InferenceBundle/GeometryBundle
  # tev_design's build_canonical_bundle.py already produces -- genuinely vmappable over S, no
  # heterogeneous-axis handling needed (§1, §2).
  encode_fn = ...  # score_conditional.encode() / _VmapEncode, reused as-is -- no new code
  encodings = encode_fn(model, prng_key, canonical_bundle, config)  # (S, L, H) -- real jax.vmap over S

  # Candidate axis: dispatch decode_states_unfused over N_CANDIDATES via xtrax, exactly as
  # batched_decode_fn already does for R x C (round 3's fix, reused unchanged here).
  strategy = _plan_axis_strategy(N_CANDIDATES, candidate_sequences.shape[0], None, activation_bytes_per_element=...)
  iterator = make_axis_dispatch_via_xtrax(strategy, axis=N_CANDIDATES.name)
  def _decode_one_candidate(seq_oh):
    return decode_states_unfused(model, encodings, seq_oh, ar_mask, prng_key, decode_step, state_iterator)
  logits = iterator(_decode_one_candidate, candidate_sequences_one_hot)  # (C, S, L, 21) -- C via xtrax

  per_state_scores = nll_from_logits(logits, candidate_sequences)  # (C, S) or (S, C) -- §3 item 1, no loop
  mean_score = average_cross_state_scores(per_state_scores)  # (C,)
  return select_mbr_candidates(mean_score, candidate_sequences, top_k=top_k)
```

Neither axis uses a Python loop: states go through a genuine `jax.vmap` (reusing `_VmapEncode`'s and
`ConditionalDecode`'s existing, production-proven mechanism, made legitimate by tev_design's upstream
`N_CANONICAL` padding), and candidates go through xtrax's `BatchPlanner`-driven `Vmap`/`SafeMap` dispatch
(reusing round 3's fix unchanged). §6's acceptance criteria are written against this version.
**Implementer note:** this pseudocode elides real argument-plumbing detail (exact `InferenceConfig`/
`AutoRegressiveMask`/`decode_step` wiring) that a from-scratch implementation will need to get right by
reading `ConditionalDecode.__call__` and `score_conditional.encode()`/`kernel()` directly at
write time — treat this as a structural sketch of the composition, not literal final code.

## 4. Where this lives — NOT `StageSet.axis_boundaries`

MBR reranking's actual input is **already-sampled sequences from a completed production run** (e.g.
tev_design's necklace P2 job, stored zarr output) being re-scored against the k reference states — this
is inherently a **post-hoc batch utility**, not a per-decode-call `StageSet` stage. `axis_boundaries`
exists to extend a *live* decode/`StageSet` pipeline; MBR reranking runs entirely after sampling is
already complete and files are already written.

**Decision:** ship `mbr_rerank` (composition function) plus the new `decode_states_unfused` (§3) as
standalone functions (new module, e.g. `aminx.sampling.mbr_consensus` — implementer's choice, mirror the
nearest existing sibling module's layout), reusing existing production encode machinery for the state
axis and `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax` + `N_CANDIDATES` for the candidate axis.
`StageSet.axis_boundaries` remains unpopulated by this work — it is a real, distinct extension point for
a *future* live-decode use case, not this one. Do not force this task to be that use case's first
occupant merely because the slot exists and is otherwise idle.

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

- **Given** the k reference states, canonicalized via tev_design's `build_canonical_bundle.py`
  (`N_CANONICAL=214`, uniform shape), and a set of already-sampled candidate sequences (arbitrary count
  C).
- **When** `mbr_rerank` is called with these inputs.
- **Then** it runs to completion via genuine `jax.vmap` over states (not a Python loop, not a
  ragged-array operation) and produces correct per-state scores for every state — confirm this with a
  test using states that have DIFFERENT pre-canonicalization residue counts (e.g. 1LVB/1LVM vs.
  reac1/reac2), verifying the padding step is what makes vmapping legitimate, not an assumption.

- **Given** `decode_states_unfused`'s per-state output for a SINGLE state (S=1 slice).
- **When** compared against `ConditionalDecode.__call__`'s fused output for that same single state
  (trivially, fusion over one element is a no-op for `ArithmeticMeanLogits`/regardless of parameters —
  the S=1 case, not the general multi-state case).
- **Then** results are numerically equivalent — this ties the new pre-fusion function back to the
  existing, trusted, already-tested `ConditionalDecode` path, rather than trusting a from-scratch
  reimplementation on its own. This is a stronger, more direct check than comparing against
  `aminx.score()` (which wraps additional encode/decode logic of its own) — it isolates exactly the
  claim this design depends on: that stopping before the two fusion calls doesn't change anything else.

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

- **Given** the finished `mbr_rerank`/`decode_states_unfused` module.
- **When** grepped for imports and for the literal string `for ` / `while ` at the top level of any
  function body.
- **Then** the state axis is processed via `jax.vmap` (reusing `_VmapEncode`/`ConditionalDecode`'s
  existing iteration machinery, e.g. `state_iterator`/`VmapIterator`) and the candidate axis via
  `_plan_axis_strategy`/`make_axis_dispatch_via_xtrax` on `N_CANDIDATES` — contains **no** hand-written
  `for`/`while` loop over either axis anywhere in the implementation — and does **not** touch
  `types/stages.py`'s `axis_boundaries` field or `inference/decode/autoregressive.py`'s live AR-loop
  fusion call site.

## 7. Assumptions

- The k reference states (1LVB, 1LVM, reac1-reac, reac2-reac per tev_design's necklace campaign) arrive
  as tev_design's existing `canonical_bundle.npz`-style padded representation (`build_canonical_bundle.py`,
  `N_CANONICAL=214`), already uniform-shape and vmappable — **not** raw, ragged per-state arrays. This
  spec does not address the tev_design-side construction of that bundle; it's existing, working infra.
  If a future caller ever wants to feed this design genuinely un-canonicalized, differently-shaped
  states directly, that's out of scope here and would need real heterogeneous-axis handling (padding
  via `bucket_boundaries`, or similar) — not assumed to work by extension of this design.
- `decode_states_unfused`'s reuse of `_decode_one_step`/`_project_logits` (private helpers in
  `inference/decode/_kernel.py`) is assumed safe because `ConditionalDecode` itself already does exactly
  this, in the same package — not reaching across a public API boundary. Confirm at implementation time
  that no other invariant `ConditionalDecode.__call__` relies on between encode and `logits_stack` is
  silently skipped by the new function (§6's cross-check against `ConditionalDecode` at S=1 is the real
  test of this, not this assumption alone).
- `BatchPlanner`'s Vmap/SafeMap strategy selection for the candidate axis is assumed to require no
  special handling beyond passing the full candidate array — verified structurally (§1, reusing round
  3's already-working `N_CANDIDATES` dispatch unchanged) but not yet run against a real large-C
  candidate set from an actual necklace production output; §6's acceptance criteria close this gap.

## 8. TBDs (only genuinely open items remain)

- Exact new module path/name (`aminx.sampling.mbr_consensus` vs. `aminx.host.mbr_rerank` vs. other) —
  implementer's call, follow the nearest sibling module's naming convention at time of implementation.
- Exact name and final signature for `decode_states_unfused` (placeholder name used in §3) — implementer's
  call at write time, following `ConditionalDecode`'s existing parameter naming closely since it mirrors
  that function's body almost exactly, minus the final two fusion calls.
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
- **State-axis dispatch question** — corrected twice more after the fourth round's fix was itself
  infeasible (Provenance, fifth round): a raw Python loop over states was correctly rejected (fourth
  round), but the replacement — dispatching through `_plan_axis_strategy` to get `SafeMap(tile=1)` on a
  "heterogeneous" axis — doesn't work either: `SafeMap`/`jax.vmap` require a genuinely uniform-shape
  stacked array, and cannot ragged-iterate differently-shaped states at all, at any tile size. The real
  resolution: states aren't actually ragged by the time they reach this design — tev_design's
  `build_canonical_bundle.py` already pads every state to `N_CANONICAL=214` — so a genuine `jax.vmap`
  over states (reusing `_VmapEncode`/`ConditionalDecode`'s existing production mechanism directly,
  §1/§3) is legitimate and requires no new dispatch layer at all for the state axis. `N_STATES`'s
  `heterogeneous=True` registry flag is a conservative declaration for the axis in the abstract, not a
  fact about this already-canonicalized input.
