# Spec: EPIC #1541 P4 scoping — inference-runner / host-sinks slice

**Task:** 260706_epic1541-p4-scoping · **Status:** DRAFT, pending review · **Scope:** aminx only; consumes xtrax 0.4.0a1 as a fixed, already-shipped dependency (no new xtrax feature requests identified — contrast with P3's planner piece, which needed one).

**Supersedes-in-part** the P4 section of `260611_aminx-xtrax-refactor.md`'s task DAG ("T4.1: move generic output_sinks/streaming_host/plan to xtrax; SPLIT runner"). That line was written before this scoping pass; recon below found the real shape of the work is materially different — smaller in the "move to xtrax" dimension, but with one genuinely invasive piece (`StageSet`/`StageBundle`) the original line didn't anticipate the difficulty of.

## Motivation

P3's planner piece taught a specific lesson: don't evaluate "does this look like it should move to xtrax" from the original spec's prose — check what xtrax's *actual, current* API looks like, and check whether aminx's code is structurally compatible with it before assuming a migration is straightforward. This spec applies that same discipline to P4, which the epic's task DAG describes only at the level of "move generic output_sinks/streaming_host/plan to xtrax; SPLIT runner" — three different claims bundled into one line, none previously checked against xtrax 0.4.0a1's real contents.

## Recon findings (2026-07-06, verified against installed `xtrax==0.4.0a1` and current aminx source)

### 1. Boundary abstractions — already done, nothing to migrate

`Fuse`/`Tap`/`Sink`/`AxisBoundary` are already sourced from `xtrax.stages.boundaries`
(`src/aminx/types/boundaries.py:5`). This predates the current session. No action.

### 2. A real naming collision, not a migration opportunity

aminx has its own **vendored** (from "jaxbeans," per its own module docstring — not from
xtrax) `BoundedCallbackHandler` and `async_indexed_stream` in
`src/aminx/utils/_vendored_callbacks.py`. xtrax has classes with the *identical names* in
`xtrax.io` / `xtrax.engine.io`, but they solve different problems:

| | aminx's (vendored) | xtrax's |
|---|---|---|
| `BoundedCallbackHandler` | Sync FIFO backlog, `max_pending`, raises on overflow (`submit(item)` appends to a list) | Async semaphore-bounded *concurrent coroutine* executor (`async def submit(coro)`, `asyncio.Semaphore`) |
| `async_indexed_stream` | Sync generator chunking an int range: `(total, chunk_size) → (chunk_start, chunk_count)` pairs | Async generator prefetching from a blocking iterable via a background thread: `(index, item)` pairs |

xtrax's versions back `xtrax.engine.Engine.fit()`'s async callback dispatch (the *training*
loop) — unrelated to the JAX `io_callback` JIT-boundary staging aminx's `output_sinks.py`
actually does. The original spec's line ("reconcile aminx io_callback drain vs xtrax
BoundedCallbackHandler") assumed these needed reconciling; they don't — they're unrelated
mechanisms that happen to share names. **This is not a migration target**, but the naming
collision is a real landmine for future readers/maintainers (someone will eventually assume
they're interchangeable and get it wrong, the same way I almost did here).

### 3. `StageBundle`/`StageSet` — the one genuinely open, genuinely invasive piece

xtrax has `xtrax.stages.bundle.StageBundle`: a "typed bag of optional callable stage
slots," where `__init_subclass__` **strictly validates every annotated field is
`Optional[Callable]`** and raises `TypeError` at class-definition time otherwise. This
matches the original spec's own sub-decision ("StageBundle = wrap-aminx-StageSet")
verbatim in *intent*.

`aminx.types.stages.StageSet` (`src/aminx/types/stages.py:280`) is a plain `eqx.Module`,
never wired to `StageBundle`. Checking whether it even *could* subclass `StageBundle` as
currently structured: **it cannot.** Three of `StageSet`'s ten fields are not
`Optional[Callable]`:

- `encoder_sink: tuple[EncoderSinkFn, ...]`
- `decoder_sink: tuple[DecoderSinkFn, ...]` (static)
- `axis_boundaries: dict[str, AxisBoundary]` (static)

`StageBundle.__init_subclass__` would reject all three immediately if `StageSet` declared
`class StageSet(StageBundle)` — this fires at class-definition/import time, not
instantiation, so it's not a subtle runtime bug, it's a hard blocker verified by reading
`xtrax/stages/bundle.py`'s validation logic directly.

This codebase already has a **direct precedent** for exactly this shape of problem:
`.praxia/docs/decisions/260630_runtimebundle-inputresolver-compose-not-subclass.md` (aminx's
planned `RuntimeBundle` vs `xtrax.run.resolver.RuntimeBundle`) concluded **compose, don't
subclass** — for a *different* root cause (frozen-dataclass-vs-`eqx.Module` metaclass
conflict, not a validation rule). `StageSet`/`StageBundle` are both `eqx.Module`-based, so
the metaclass issue doesn't apply here — but the *same resolution* (compose, not subclass)
is still the likely answer, now for a validation-rule reason instead: `StageSet` would hold
an inner `StageBundle` instance for its 7 pure-`Optional[Callable]` fields
(`logit_transform`, `ar_logit_transform`, `decode_step`, `sample_step`, `tie_group_fuse`,
`encoding_fusion`, `decoding_fusion`), keeping the 3 container-typed fields
(`encoder_sink`/`decoder_sink`/`axis_boundaries`) as `StageSet`'s own, separate concern.

**This is the one piece of P4 with real blast radius.** `StageSet` is constructed and read
throughout the entire decode/inference pipeline (`make_stage_set`, `kernel_dispatch.py`,
every mode class in `inference/decode/`, topology-inference rules keyed on which fields are
`None`). A composition refactor touches every one of those call sites' field access pattern
(`stage_set.decode_step` → `stage_set.bundle.decode_step`, or similar), not just a type
annotation.

### 4. Concrete sinks stay local — same pattern as `pad.py`/`bucketing.py` in P3

`output_sinks.py`'s concrete sink classes (`StreamingTensorStagingSink`,
`EncoderIntermediateStagingSink`, `JacobianAccumulationSink`) and `streaming.py`'s
HDF5/ArrayRecord writers (`_sample_streaming`, `_sample_streaming_arrayrecord`,
`_sample_streaming_averaged`) are protein/MPNN-domain-specific: keyed by structure/noise/
chunk indices, writing protein-specific `DesignPayload`/`DesignMetadata` records. No
generic xtrax equivalent exists or would make sense to move these to — this is domain
business logic built on top of `io_callback`, not a tiling-style generic primitive.
`streaming_host.py` (41 lines) is a thin wrapper around `jax.effects_barrier()` plus the
vendored `async_indexed_stream` — trivial, not worth migrating on its own, though its use
of the collision-prone name (finding #2) should be fixed alongside it.

### 5. "SPLIT runner" — a separate, aminx-internal question, not an xtrax migration

The original spec's own wording ("move ... to xtrax; **SPLIT runner**") bundles two
different verbs. Nothing found in this recon suggests `kernel_dispatch.py`'s `_sample_batch`
has a natural xtrax destination — "splitting" it (if still wanted) is an aminx-internal
architectural refactor question, orthogonal to anything xtrax offers. Not scoped further
here; flagged as a separate decision if still desired.

## Revised task list

Given the above, P4's real remaining scope is much narrower than the original DAG line, but
concentrated almost entirely in one nontrivial piece:

- **T4.0 (new)**: rename aminx's vendored `BoundedCallbackHandler`/`async_indexed_stream`
  (`_vendored_callbacks.py`, and their use in `streaming_host.py`) to names that don't
  collide with xtrax's differently-behaved same-named classes — e.g.
  `HostBackpressureQueue`/`chunk_range` or similar. Mechanical, no behavior change, low
  risk, contained to `_vendored_callbacks.py` + its ~2-3 call sites.
- **T4.1 (revised)**: `StageSet` composes an inner `xtrax.stages.bundle.StageBundle`
  (built as a genuine subclass of `StageBundle` holding only the 7 pure-`Optional[Callable]`
  fields) rather than being flattened into one. Needs: (a) a design decision on the exact
  composition shape and field-access migration pattern before touching the ~10+ call sites
  across `inference/decode/`, `inference/logits.py`, `host/plan.py`,
  `host/kernel_dispatch.py`; (b) verification that PyTree structure changes (if any) don't
  change JIT-recompile behavior — this is the one place a real parity/recompile gate is
  warranted, narrower than the original T4.3's "5 hotspots" framing (io_callback drain,
  ArrayRecord/HDF5 read, and the planner are untouched by this specific piece).
- **Not scoped / deferred**: `output_sinks.py`/`streaming.py`'s concrete sinks (stay local,
  no action), "SPLIT runner" (separate architectural question, not part of this migration).

## Definition of done (for this scoping pass, not for P4 itself)

This document is a scoping pass, not an implementation plan. Before any code changes:
confirm the `StageSet`/`StageBundle` composition shape (a follow-up design decision,
likely its own short spec or decision doc mirroring `260630`'s format), and confirm whether
T4.0's rename and T4.1's composition should land as one PR or two (T4.0 has near-zero risk
and no dependency on T4.1; they don't need to be sequenced together).

## Off-ramp

If the `StageSet` composition refactor's call-site blast radius, once fully inventoried,
looks disproportionate to the benefit (topology-inference rules, JIT-recompile risk across
the whole decode pipeline, for a class that already works correctly as a plain `eqx.Module`
with no consumer confusion) — the honest fallback is: leave `StageSet` as-is, permanently,
and record that decision the same way `bucketing.py`/`pad.py`/the old planner's stay-local
pieces were recorded. `StageBundle` existing in xtrax does not, by itself, obligate aminx to
use it if the fit is poor.
