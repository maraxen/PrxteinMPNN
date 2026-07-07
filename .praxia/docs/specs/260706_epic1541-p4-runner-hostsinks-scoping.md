# Spec: EPIC #1541 P4 scoping — inference-runner / host-sinks slice

**Task:** 260706_epic1541-p4-scoping · **Status:** CONVERGED — challenger + defender review complete, no blocking objections remain · **Scope:** aminx only; consumes xtrax 0.4.0a1 as a fixed, already-shipped dependency.

**Supersedes-in-part** the P4 section of `260611_aminx-xtrax-refactor.md`'s task DAG ("T4.1: move generic output_sinks/streaming_host/plan to xtrax; SPLIT runner").

**Revision note (2026-07-06):** this document's first draft was adversarially challenged
(`praxia-spec-challenger`, task `260706_epic1541-p4-scoping`). The challenge found the
centerpiece claim materially wrong — not a disagreement about judgment, a factual error I
then independently reproduced myself (see Finding 3, Reasons A/B). This revision corrects
it rather than defending it, and additionally closes the two suggestion-severity gaps the
review raised (C6 — "SPLIT runner" dismissal was asserted, not evidenced; C7 — no explicit
comparison against xtrax's I/O primitives) with real evidence instead of assertions. A
second review round (`praxia-spec-defender`) independently re-verified every corrected
claim against source and returned `overall_verdict: converged` — see Finding 5's closing
note for the one remaining suggestion-level polish item that survived that round.

## Motivation

P3's planner piece taught a specific lesson: don't evaluate "does this look like it should
move to xtrax" from the original spec's prose — check what xtrax's *actual, current* API
looks like, and check whether aminx's code is structurally compatible with it before
assuming a migration is straightforward. This spec applies that discipline to P4. The first
draft did this for four of five findings but got the fifth (the centerpiece) wrong by not
applying the discipline rigorously enough to its own conclusion — see below.

## Recon findings (2026-07-06, verified against installed `xtrax==0.4.0a1` and current aminx source)

### 1. Boundary abstractions — already done, nothing to migrate

`Fuse`/`Tap`/`Sink`/`AxisBoundary` are already sourced from `xtrax.stages.boundaries`
(`src/aminx/types/boundaries.py:5`). Predates this session. No action.

### 2. `xtrax.run.resolver`/`xtrax.run.sink` — checked, no seam here either

For completeness (the first draft asserted this without checking): `xtrax.run.resolver`
holds `RuntimeBundle` (a thin `iterator: VmapIterator | ... | None; model: eqx.Module`
dataclass) and `InputResolver` (`(spec, bundle) -> FeatureBatch`, a `runtime_checkable
Protocol`) — this is the same `RuntimeBundle`/`InputResolver` pair
`260630_runtimebundle-inputresolver-compose-not-subclass.md` already scoped for aminx's
*not-yet-built* `#1910`, unrelated to the sampling runner. `xtrax.run.sink.SinkSpec` is a
3-field routing-config dataclass (`output_dir`/`format`/`flush_every`) — a config shape, not
an implementation. `host/kernel_dispatch.py:_sample_batch(spec, batched_ensemble: Protein,
plan: InferencePlan, ...)` receives `batched_ensemble` **already resolved** — it does not do
spec-to-bundle input resolution itself, so `InputResolver`'s pattern has no call site to
attach to inside this function. Conclusion holds, now on evidence: **"SPLIT runner" is a
separate, aminx-internal architectural question** (if still wanted at all), not an xtrax
migration target, and not scoped further here.

### 3. `StageBundle`/`StageSet` — NOT viable to adopt, for two independent reasons (revised)

**What the first draft claimed:** "3 of `StageSet`'s 10 fields aren't `Optional[Callable]`;
the other 7 could go into a genuine `StageBundle` subclass." **This is wrong.** Adversarial
review (and my own independent verification, reproduced below) found:

**Reason A — every field fails, not 3 of 10.** `StageBundle.__init_subclass__`
(`xtrax/stages/bundle.py:32-70`) accepts a field only if it is `X | None` where `X` is
*literally* `Callable` or `Callable[...]` (`_is_callable_type`, `bundle.py:10-19`). Checked
each of `StageSet`'s 10 fields (`src/aminx/types/stages.py:353-362`) against that rule:

| Field | Annotation | Passes `StageBundle`'s rule? |
|---|---|---|
| `logit_transform` | `BatchLogitFn \| None` | No — `BatchLogitFn` is `class BatchLogitFn(Protocol)` (`inference/logits.py:21`), not literal `Callable` |
| `ar_logit_transform` | `BatchLogitFn \| None` | No — same |
| `decode_step` | `ConditionalDecodeStep \| UnconditionalDecodeStep \| None` | No — 3-way union, `len(args) != 2` at `bundle.py:52` |
| `sample_step` | `Any \| None` | No — `_is_callable_type(Any)` is `False` |
| `tie_group_fuse` | `TieGroupFuseFn \| None` | No — `class TieGroupFuseFn(Protocol)` (`inference/logits.py:334`) |
| `encoder_sink` | `tuple[EncoderSinkFn, ...]` | No — not even a union |
| `decoder_sink` | `tuple[DecoderSinkFn, ...]` | No — same |
| `encoding_fusion` | `EncodingFusionFn \| None` | No — `class EncodingFusionFn(Protocol)` (`types/stages.py:115`) |
| `decoding_fusion` | `DecodingFusionFn \| None` | No — `class DecodingFusionFn(Protocol)` (`types/stages.py:128`) |
| `axis_boundaries` | `dict[str, AxisBoundary]` | No — not a union |

**Zero of ten fields would pass.** The 7 fields the first draft called "pure" are Protocols,
a 3-way union, or `Any` — none is literal `Callable`. A `StageBundle` subclass holding those
7 fields verbatim would `TypeError` at class-definition time for every single one.

**Reason B — independent of Reason A, and more fundamental: PEP 563 defeats the validator
entirely, for *any* field.** Both `src/aminx/types/stages.py:14` and
`src/aminx/inference/logits.py:8` have `from __future__ import annotations`. Under PEP 563,
`cls.__annotations__` holds **strings**, not type objects — confirmed empirically:

```python
from __future__ import annotations
from typing import Optional, Callable
class Foo:
    x: Optional[Callable] = None
print(Foo.__annotations__)        # {'x': 'Optional[Callable]'}
print(type(Foo.__annotations__['x']))  # <class 'str'>
```

`StageBundle.__init_subclass__` reads raw `cls.__annotations__` (`bundle.py:40`) and calls
`get_origin(field_type)` on each value (`bundle.py:45`) — **never** `typing.get_type_hints()`
to resolve the strings first. Confirmed: `get_origin('BatchLogitFn | None')` returns `None`
regardless of what the string actually says, so the validator's `else` branch
(`bundle.py:65-70`, "must be Optional[Callable]") fires unconditionally. **A `StageBundle`
subclass cannot be defined inside any module using `from __future__ import annotations` —
not because of what the fields are, but because the validator never sees resolved types at
all.** This would apply even to a hypothetical field that *was* genuinely `Callable | None`.

**Consequence:** `StageBundle` is not adoptable by `StageSet` as it exists today, full stop
— not "compose instead of subclass" (the first draft's fallback), but *no current path*,
short of one of:
(a) an upstream xtrax change to `StageBundle.__init_subclass__` (resolve annotations via
`get_type_hints()`; accept Protocols/`Any`/bounded unions, not just literal `Callable`) —
real feature work against xtrax, analogous to what P3's planner piece needed, not attempted
here;
(b) hosting a `StageBundle` subclass in a *new* module with bare `Callable`/`Callable[...]`
annotations and no `from __future__ import annotations`, which discards the Protocol/union
typing that currently documents real invariants (e.g. `decode_step`'s union *is* the
topology-inference signal — see `StageSet`'s own docstring at `types/stages.py:289-292`).

Neither is "compose, don't subclass" in the sense the `260630` `RuntimeBundle` precedent
used that phrase (there, composition was a clean, working alternative to a blocked
subclass). Here, composition doesn't dodge the blocker — `StageBundle`, subclassed
*anywhere* by anyone in a PEP-563 module, hits Reason B regardless of which fields it holds.

**Benefit check (missing from the first draft, per review objection C4):** what would adopting
`StageBundle` actually buy aminx? `StageBundle` provides exactly two methods:
`active_stages()` and `has_stage()` (`bundle.py:97-104`), simple list/bool checks over
non-None fields — behavior `StageSet`'s own topology-inference docstring already documents
and its callers already implement ad hoc (checking `is not None` directly). No aminx code
currently calls anything resembling `active_stages`/`has_stage`. **The benefit is
approximately zero.**

**Blast radius, corrected (per review objection C3):** the first draft said "~10+ call sites."
Actual count — grep for the 10 field-access patterns
(`.logit_transform`/`.ar_logit_transform`/`.decode_step`/`.sample_step`/`.tie_group_fuse`/
`.encoder_sink`/`.decoder_sink`/`.encoding_fusion`/`.decoding_fusion`/`.axis_boundaries`)
across `src/`: **52 occurrences across 12 files** —
`host/{averaging,kernel_dispatch,plan,runner,stage_adapter}.py`,
`inference/{driver,logits}.py`,
`inference/decode/{_base,autoregressive,conditional,unconditional}.py`, `types/stages.py`.
This includes `host/plan.py`'s `eqx.tree_at(lambda s: s.encoding_fusion, ...)` /
`s.decoding_fusion` / `s.decode_step` path lambdas (`plan.py:781-815`) — PyTree-path
expressions that would need rewriting under any restructure, and are exactly the kind of
site where a JIT-recompile regression would hide.

**Given near-zero benefit, a genuine (not just difficult) structural blocker, and a
52-occurrence blast radius: `StageSet` stays exactly as it is. This is the default
conclusion, not a contingent fallback (see revised Off-ramp below).**

### 4. Concrete sinks stay local — same pattern as `pad.py`/`bucketing.py` in P3

`output_sinks.py`'s concrete sink classes (`StreamingTensorStagingSink`,
`EncoderIntermediateStagingSink`, `JacobianAccumulationSink`) and `streaming.py`'s
HDF5/ArrayRecord writers (`_sample_streaming`, `_sample_streaming_arrayrecord`,
`_sample_streaming_averaged`) are protein/MPNN-domain-specific: keyed by
structure/noise/chunk indices, writing protein-specific `DesignPayload`/`DesignMetadata`
records. No generic xtrax equivalent exists (checked `xtrax.run.sink.SinkSpec` — a routing
config, not an implementation) or would make sense to move these to. Domain business logic
built on `io_callback`, not a tiling-style generic primitive.

`streaming_host.py`'s two generic-looking helpers, named explicitly (per review C7, which
found the first draft's "trivial, not worth migrating" dismissal never named what it was
dismissing): `sink_barrier()` (`streaming_host.py:18-20`, a bare `jax.effects_barrier()`
wrapper) and `iter_streaming_chunks()` (`:31-33`, wraps the vendored `async_indexed_stream`
— see Finding 5). xtrax's I/O layer (`xtrax.engine.io`/`xtrax.io`) is entirely
`asyncio`-coroutine-based, backing the *training* `Engine.fit()` loop — there is no
`jax.effects_barrier`-style synchronization primitive in xtrax to compare `sink_barrier`
against, because xtrax's async layer isn't synchronizing JAX `io_callback` effects at all,
it's bounding concurrent Python coroutines in a plain training loop. Different problem
domain, not a smaller version of the same thing. Both helpers stay local, un-migrated,
alongside the concrete sinks they serve.

### 5. Naming collision — real, but the recommended fix was wrong (revised)

aminx has its own **reimplementation in the style of** ("jaxbeans-style," per its own
docstring — not literally vendored/copied, a first-draft wording error the review caught)
`BoundedCallbackHandler` and `async_indexed_stream` in
`src/aminx/utils/_vendored_callbacks.py`. xtrax has classes with the *identical names* in
`xtrax.io` / `xtrax.engine.io`, solving different problems:

| | aminx's | xtrax's |
|---|---|---|
| `BoundedCallbackHandler` | Sync FIFO backlog, `max_pending`, raises on overflow | Async semaphore-bounded concurrent coroutine executor, backs `xtrax.engine.Engine.fit()`'s training-loop callback dispatch |
| `async_indexed_stream` | Sync generator chunking an int range: `(total, chunk_size) → (chunk_start, chunk_count)` | Async generator prefetching from a blocking iterable via a background thread: `(index, item)` |

Not a migration target — confirmed different problems, matching the first draft's
conclusion. **But the recommended fix (rename) was wrong for `BoundedCallbackHandler`
specifically: it has zero call sites anywhere in `src/`** (grep finds only its own
definition) — dead code, not a landmine anyone will trip over by using it. The honest fix is
**delete**, not rename. `async_indexed_stream` has exactly one real call site
(`host/streaming_host.py:15,33`, not "~2-3" as the first draft estimated) and is worth
renaming for clarity, since it's actually used and actually collides.

## Revised task list

- **T4.0a**: delete aminx's unused `BoundedCallbackHandler` from `_vendored_callbacks.py`.
  Zero call sites; confirmed via grep before deletion, same discipline as `carry_shape.py`'s
  retirement in P3.
- **T4.0b**: rename `async_indexed_stream` (and update its one call site in
  `streaming_host.py`) to something that doesn't collide with xtrax's differently-behaved
  same-named function — e.g. `chunk_int_range`. Mechanical, no behavior change.
- **`StageSet`/`StageBundle`**: **not a task.** Recorded as a stay-local decision (see Finding 3)
  — not deferred, not blocked, closed. If a future need arises to actually adopt
  `StageBundle`, it requires an upstream xtrax change to the validator first (Reason B is not
  aminx-side fixable at all); that would be new xtrax feature work, scoped separately if it
  ever becomes worth doing, the same way P3's planner piece got a real xtrax feature added
  when the case for it was concrete.
- **"SPLIT runner"**: not scoped here (Finding 2) — a separate, aminx-internal architectural
  question if still wanted, with no connection to xtrax's `run`/`stages` modules found.
- **Concrete sinks / streaming writers**: stay local, no action (Finding 4).

## Definition of done (for this scoping pass)

T4.0a/T4.0b are small enough to just implement directly once this revision is accepted —
no further design decision needed for either. Everything else in P4's original scope is now
closed as a documented stay-local/out-of-scope decision, not pending work.

## Off-ramp

Not conditional this time: `StageSet` stays as a plain `eqx.Module`, permanently, by
default — given a benefit of approximately zero (two unused helper methods) against a
confirmed structural blocker (Reason B applies regardless of field types) and a
52-occurrence blast radius. `StageBundle` existing in xtrax does not obligate aminx to use
it; the case for the fit here is negative, not merely disproportionate.
