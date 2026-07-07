# Finding: `n_samples` axis planning silently decoupled from actual runtime sample count

**Task:** 260706_samples-axis-cardinality-mismatch · **Status:** DRAFT — recon complete, remediation not yet chosen · **Scope:** aminx only, `host/kernel_dispatch.py` / `host/plan.py` / `run/specs.py`. **Not part of EPIC #1541** — this predates the xtrax migration and would exist identically with the old, retired local `BatchPlanner`; it is a gap in how aminx calls its own planner, not in the planner itself.

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
- **`n_structures`**: cardinality at plan time is `spec.batch_size`
  (`host/plan.py:203`), and the actual structure count is
  `batched_ensemble.coordinates.shape[0]` (`kernel_dispatch.py:127`) — whether these two can
  diverge depends on whether `protein_iterator`'s own batching is driven by the same
  `spec.batch_size` field. **Not verified in this pass** — flagged as an open question, not
  asserted safe, but structurally a tighter (single-config-field) coupling than the
  `n_samples` case, which has two independent fields by design.
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

## Not yet determined

- Whether this has already caused an observed problem (OOM, silently-larger-than-budgeted
  memory use) — not checked in this pass; would need a look at prior incident logs / test
  failures / cluster job history if such exist.
- Whether `n_structures`'s coupling via `protein_iterator` is actually as safe as it looks
  structurally, or has its own version of this gap — not traced in this pass.
- What `samples_batch_size` was originally intended to mean: a genuine "how many samples to
  vmap together for memory reasons" knob (in which case it should scale with or be derived
  from whatever the actual per-call count is), or a vestigial config field nobody varies
  from its default in practice (in which case the simplest fix might be collapsing the two
  fields into one).

## Possible remediation directions (not chosen — a design decision, not made here)

- **A.** Make the planner's `n_samples` cardinality reflect the actual runtime count
  (`target_num_samples`) at each `_sample_batch` call, rather than the static
  `samples_batch_size` config value — closes the gap at the source, but changes what
  `samples_batch_size` means (possibly obsoletes it as a separate field) and re-opens the
  question of whether recomputing a plan per-call (already happening today, per line 114)
  is the right performance shape at all.
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
