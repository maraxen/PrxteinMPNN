# Finding: `n_samples` axis planning silently decoupled from actual runtime sample count

**Task:** 260706_samples-axis-cardinality-mismatch · **Status:** DRAFT — recon complete, remediation not yet chosen · **Scope:** aminx only, `host/kernel_dispatch.py` / `host/plan.py` / `run/specs.py`. **Not part of EPIC #1541** — this predates the xtrax migration and would exist identically with the old, retired local `BatchPlanner`; it is a gap in how aminx calls its own planner, not in the planner itself.

**REVISED 2026-07-07** — `.praxia/docs/specs/260707_xtrax-migration-gap-audit-runspec-scaffolding.md`
(Finding D) found the trace below describes the *legacy* dispatch path (`use_unified_driver=False`,
`_safe_map(..., batch_size=samples_bs)`). The *default* path (`use_unified_driver=True`, the default
everywhere it's set) dispatches via `_dispatch_axis`, whose `Vmap` branch is an **unconditional**
`jax.vmap(body)(xs)` — no `batch_size`/`num_elements` check of any kind, not even the defeatable
`batch_size == 0` condition this document analyzes. The default-path variant is more severe and more
likely to be hit in practice than what's described below. Read this document for the root-cause
history and empirical-confirmation methodology; read the 260707 document for the full severity picture
before scoping remediation.

## Discovery context

Found while implementing EPIC #1541 P4's T4.0b (renaming aminx's vendored
`async_indexed_stream` to `chunk_int_range` — see
`.praxia/docs/specs/260706_epic1541-p4-runner-hostsinks-scoping.md`). When asked to explain
what the renamed function does, an initial answer described it as "purely a host-side
bookkeeping helper... no JAX tracing involved" — accurate for the function itself, but
misleading about its role: it directly determines the sample count fed into a call whose
correctness depends on that count matching what the axis planner verified. Follow-up
pressure ("how will we leverage axis planning and dedup if this is just a dumb Python loop")
prompted tracing the actual call chain rather than re-asserting the same claim, which
surfaced this.

## The finding

Two `SamplingSpecification` fields (`run/specs.py:518-519`) govern what is conceptually the
same axis, from two independent, uncoordinated places:

```python
samples_batch_size: int = 16       # feeds the axis planner's memory-budget check
samples_chunk_size: int | None = None  # feeds the actual runtime sample count per call
```

No validator, `__post_init__`, or cross-check anywhere ties them together — confirmed by
reading `SamplingSpecification`'s full field list.

**Trace, in `_sample_batch` (`host/kernel_dispatch.py:100-181`):**

1. `batch_plan = make_sampling_planner(spec)` (line 114) — builds the `n_samples` axis's
   Vmap/SafeMap decision using `cardinality = spec.samples_batch_size` (`host/plan.py:206`).
   This is the value the joint-budget memory check verifies fits the GPU.
2. `samples_bs = extract_batch_sizes(batch_plan)[1]` (line 116) — the resulting decision,
   translated to `_safe_map`'s convention (`0` = Vmap/no-chunking, per `_legacy_batch_size`,
   `host/plan.py:223-238`).
3. `target_num_samples = resolve_target_samples(spec, chunk_sample_count, grid_lineage)`
   (line 121) — resolves to `chunk_sample_count` if given, else `spec.run_spec.sampling.num_samples`
   (`host/plan.py:299-339`). **Computed independently of step 1**, from a different input.
4. `sample_keys = compute_sample_keys(base_key, target_num_samples, ...)` (lines 176-181) —
   an array of exactly `target_num_samples` elements (`host/plan.py:264-296`,
   `np.arange(target_num_samples, ...)`).
5. `_safe_map(_run_one_sample, sample_keys, batch_size=samples_bs)` (lines 434, 531) —
   dispatches `sample_keys` (sized by step 3/4's `target_num_samples`) using `samples_bs`
   (sized by step 1/2's `samples_batch_size`).

`safe_map`'s dispatch logic (`utils/safe_map.py:49`):
```python
if batch_size is None or batch_size == 0 or num_elements <= batch_size:
    # vmap everything at once — no chunking
```
If the plan decided **Vmap** for `n_samples` (`samples_bs == 0`, chosen because
`samples_batch_size` comfortably fit the memory budget), this branch fires
**unconditionally** — `batch_size == 0` short-circuits the check regardless of
`num_elements`. Whatever `sample_keys` actually contains (which can be far larger than
`samples_batch_size`) gets vmapped in a single call, with no re-check against the memory
budget the planner was supposed to enforce for that cardinality.

**`make_sampling_planner(spec)` is also called fresh on every single `_sample_batch`
invocation** (line 114) — there is no caching or per-run plan reuse for this axis set, and
each call uses the same static `spec.samples_batch_size`, blind to whatever
`chunk_sample_count` arrives in that specific call.

## Why this doesn't affect the other three axes

- **`n_temperatures`/`n_noises`**: cardinality at plan time
  (`len(spec.temperature)`/`len(spec.backbone_noise)`, `host/plan.py:209-212`) and the actual
  arrays used later (`temperatures`/`noises` at `kernel_dispatch.py:123-124`) are **both
  derived from the same source fields within the same call** — structurally guaranteed to
  match, no independent knob exists.
- **`n_structures`**: **verified safe**, by a different mechanism than `n_temperatures`/
  `n_noises`. Cardinality at plan time is `spec.batch_size` (`host/plan.py:203`), and
  `protein_iterator` (which yields the actual `batched_ensemble`) is constructed via
  `create_protein_dataset(..., batch_size=spec.batch_size, ...)` (`host/prep.py:102-104`) —
  the **same** config field drives both. Since the last batch of any batched iterator can
  only be smaller than the requested batch size (never larger — a ragged final batch, not
  an oversized one), `batched_ensemble.coordinates.shape[0]` is bounded above by the same
  `spec.batch_size` the plan used as `n_structures`'s cardinality. The direction of possible
  divergence (actual ≤ planned) is the safe direction — it can only make Vmap *more*
  conservative than needed, never less. No fix needed here.
- **`n_samples`** is the only axis with two independently-settable spec fields
  (`samples_batch_size` for planning, `samples_chunk_size`/`num_samples` for the actual
  runtime count) and zero code connecting them.

## Failure mode specifics

Only manifests when **both**:
1. The plan decides **Vmap** for `n_samples` (i.e. `samples_batch_size` comfortably fits the
   memory budget) — SafeMap decisions are **not** affected, because
   `jax.lax.map(..., batch_size=tile)` bounds per-iteration memory by `tile` regardless of
   the total array size; chunking still happens correctly there.
2. The actual `target_num_samples` at runtime exceeds `samples_batch_size`.

Given `samples_batch_size` defaults to 16 (small — plausibly fits most GPU budgets → Vmap
is a likely choice) and real sampling runs commonly request well over 16 total samples per
structure, this looks like a plausible-to-hit default-configuration gap, not an exotic edge
case — though this pass did not check test coverage or production run configs to confirm
it has actually fired in practice.

## Empirical confirmation (2026-07-06)

Reproduced live rather than trusting the code trace alone (matching this project's
"verify the measurement pipeline" discipline):

```
spec = SamplingSpecification(inputs="test.pdb", model_family="ligandmpnn")
# samples_batch_size: 16, samples_chunk_size: None, num_samples: 1

plan = make_sampling_planner(spec)
# n_samples decision: strategy=Vmap batch_size=1 cardinality=16
#   reasoning="joint-budget: Vmap retained (final estimate 40 B <= budget 3435973836 B)"
# extract_batch_sizes -> samples_bs = 0   (legacy Vmap sentinel)

keys = jax.random.split(jax.random.key(0), 500)   # target_num_samples=500, >> 16
jax.eval_shape(lambda ks: safe_map(fn, ks, batch_size=0), keys)
# -> output shape (500,), fn traced ONCE at scalar per-element shape
# -> confirmed: safe_map genuinely vmaps all 500 in a single call, zero chunking
```

Confirms every step of the trace, not just the theoretical `safe_map` branch: realistic
defaults *do* produce a Vmap decision (a 40-byte estimate trivially fits any real memory
budget), and that decision, applied to an array 31× larger than the cardinality it was
verified against, genuinely skips chunking entirely rather than falling back to some safe
default.

## Root cause: when and how this was introduced

Traced via `git log -S` on `run/specs.py` (project's older name, `prxteinmpnn`, at the time):

- `samples_batch_size` was introduced in `f0f9340` (2025-10-28, "Enhance sampling
  specifications with batch size parameters for samples and noise") wired **directly** as
  `jax.lax.map(noise_map_fn, keys, batch_size=spec.samples_batch_size)` — at that point it
  was unconditionally safe: `jax.lax.map` always chunks into groups of the given
  `batch_size`, regardless of the total array length, with no Vmap-vs-chunk *decision*
  involved at all.
- The gap was introduced in `5ca2abf` (2026-05-07, "Wave 2 — BatchPlanner advisory logging +
  active n_structures dispatch"), which replaced that direct, always-safe call with
  planner-driven dispatch ("batch_size from planner"). This is the commit that turned
  `samples_batch_size` from a raw, always-applied chunk size into a *cardinality input to a
  Vmap-vs-SafeMap decision* — a semantic change that silently dropped the "always bound
  memory regardless of actual count" guarantee for the Vmap branch.
- That commit's own message states: **"Parity gate: 26/26 pass. Outputs numerically
  identical — safe_map and vmap are equivalent for independent-per-element functions."**
  This explains why the gap has stood unnoticed: the parity gate that validated this change
  checked *numerical correctness* (chunked vs. unchunked results, which are indeed identical
  for independent-per-element work — that's `safe_map`'s whole design point), not *memory
  behavior*. A correctness-focused test suite would never have caught this; it isn't a
  wrong-answer bug, it's a memory-safety-guarantee bug, invisible to output comparison.

## Still open

- Whether this has already caused an observed problem (OOM, silently-larger-than-budgeted
  memory use) in an actual run — not checked; would need prior incident logs / cluster job
  history if such exist. The empirical repro above only confirms the mechanism fires with
  realistic defaults, not that it has been hit in production.
- What `samples_batch_size` was originally intended to mean *now*, given the historical
  finding above: the pre-`5ca2abf` semantics (an always-applied chunk size, safe by
  construction) is arguably what should be restored, rather than treating it as a
  cardinality input to a Vmap/SafeMap decision at all — this bears directly on which
  remediation direction (below) is most faithful to the field's original intent.

## Possible remediation directions (not chosen — a design decision, not made here)

- **A.** Make the planner's `n_samples` cardinality reflect the actual runtime count
  (`target_num_samples`) at each `_sample_batch` call, rather than the static
  `samples_batch_size` config value — closes the gap at the source, and is arguably the most
  faithful fix given the root-cause history above: it restores the pre-`5ca2abf` property
  that memory safety is verified against the *real* count, not a separately-configured
  proxy for it. Changes what `samples_batch_size` means going forward (possibly obsoletes it
  as a separate field — if the plan is always built from the real count, there may be
  nothing left for `samples_batch_size` to configure) and re-opens the question of whether
  recomputing a plan per-call (already happening today, per line 114) is the right
  performance shape at all.
- **B.** Add a defensive runtime check in `_sample_batch` (or in `extract_batch_sizes`) that
  raises or forces a downgrade if `target_num_samples` diverges from the cardinality the
  plan was built for and the decision is Vmap — cheaper to implement, doesn't fix the
  underlying two-independent-knobs design, just guards against the dangerous combination.
- **C.** Add cross-field validation to `SamplingSpecification` (or rename/merge the two
  fields) so `samples_batch_size` and `samples_chunk_size`/`num_samples` can't silently
  diverge in the first place.

This document stops at the finding and options — no remediation has been implemented, and
no direction has been chosen. Next step is a decision on priority (this is a correctness/
memory-safety gap, arguably higher priority than P4's in-flight naming cleanup, but that's
a scheduling call) and which remediation direction to pursue before any code changes.
