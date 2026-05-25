# COMP-UNIFIED: Encoder Fusion via InferencePlan Composability

**Date:** 2026-05-25  
**Ticket:** COMP-UNIFIED  
**Branch:** `refactor-full`  
**Status:** SPEC REVISED — oracle critique addressed (2026-05-25)

---

## Summary

Eliminate the `if spec.average_node_features:` branch in `runner.py` and the
parallel averaged-path functions (`_sample_non_streaming_averaged`,
`_sample_streaming_averaged`, `_sample_batch_averaged`). Averaging becomes an
encoder-fusion operation expressed as a differently-constructed `encode_fn`
closure inside `InferenceComponents`. The `_sample_batch` dispatch accepts
`plan: InferencePlan` (replacing the current `model + stage_set` pair), calls
`plan.encode()` once per structure, then vmaps `plan.decode()` over samples.

A new optional `encoder_sink: EncoderSinkFn | None` slot in `StageSet` allows
downstream consumers to register an `io_callback`-based hook that stages encoder
intermediates (averaged features, multi-state encodings) for inspection or
persistence without coupling the kernel to HDF5 I/O.

---

## Motivation

### Current pain points

1. **Duplicated dispatch logic.** Three parallel paths exist for regular,
   averaged, and streaming-averaged sampling. All share >80% identical structure
   but diverge on encode strategy. Bug fixes must be applied to each.

2. **Hardcoded averaging.** `spec.average_node_features` is a boolean branch
   inside the host runner, not a composable property of the pipeline. There is
   no way to plug in a custom encoder fusion strategy.

3. **`_sample_batch` is opaque to `InferencePlan`.** The function takes raw
   `model + stage_set` and rebuilds inference structure internally. It cannot
   leverage the `encode_fn` closure already on `InferencePlan`.

4. **No staged encoder intermediates.** When averaging, the fused encoder
   features are discarded after decoding. There is no hook for saving them to
   an HDF5 or staging them for downstream analysis.

---

## Design

### 1. `EncoderSinkFn` protocol + `encoder_sink` slot in `StageSet`

Add to `src/prxteinmpnn/types/stages.py`:

```python
class EncoderSinkFn(Protocol):
    """Optional side-effect hook called after encode().

    Fires io_callback to stage encoder intermediates for persistence or inspection.
    Implementations MUST call jax.experimental.io_callback with ordered=False.
    """
    def __call__(
        self,
        enc: EncoderOutput,
        batch_idx: jax.Array,    # jnp.int32 scalar
        structure_idx: jax.Array, # jnp.int32 scalar
    ) -> None: ...
```

Add to `StageSet`:

```python
class StageSet(eqx.Module):
    logit_transform: BatchLogitFn | None = None
    ar_logit_transform: BatchLogitFn | None = None
    decode_step: ConditionalDecodeStep | UnconditionalDecodeStep | None = None
    sample_step: Any | None = None
    tie_group_fuse: TieGroupFuseFn | None = None
    encoder_sink: EncoderSinkFn | None = None  # NEW
```

The new field defaults to `None` (no-op). Plain Python `None` is static in
equinox, so this does not change trace shape for existing `StageSet` instances
that leave it unset.

**Topology rule addition (docstring):**
- If `encoder_sink is not None` → fire after `plan.encode()` per structure;
  `io_callback` ordering is `ordered=False`.

### 2. `IoCallbackEncoderSink` in `output_sinks.py`

```python
class IoCallbackEncoderSink(eqx.Module):
    """Stages encoder node/edge features via io_callback to active sink."""

    def __call__(
        self,
        enc: EncoderOutput,
        batch_idx: jax.Array,
        structure_idx: jax.Array,
    ) -> None:
        jax.experimental.io_callback(
            _dispatch_encoder_intermediate_io,
            None,
            jnp.int32(batch_idx),
            jnp.int32(structure_idx),
            enc.node_features,
            enc.edge_features,
            ordered=False,
        )
```

Add `_dispatch_encoder_intermediate_io` callback and
`EncoderIntermediateStagingSink` to `output_sinks.py`, following the same
`ContextVar`-based pattern as `StreamingTensorStagingSink`:

```python
_ENCODER_STAGING_SINK: ContextVar[EncoderIntermediateStagingSink | None] = \
    ContextVar("_ENCODER_STAGING_SINK", default=None)

def active_encoder_staging_sink() -> EncoderIntermediateStagingSink | None:
    return _ENCODER_STAGING_SINK.get()

@contextmanager
def encoder_sink_session() -> Generator[EncoderIntermediateStagingSink, None, None]:
    sink = EncoderIntermediateStagingSink()
    token = _ENCODER_STAGING_SINK.set(sink)
    try:
        yield sink
    finally:
        _ENCODER_STAGING_SINK.reset(token)

def _dispatch_encoder_intermediate_io(
    batch_idx: np.ndarray,
    structure_idx: np.ndarray,
    node_features: np.ndarray,
    edge_features: np.ndarray,
) -> None:
    sink = active_encoder_staging_sink()
    if sink is None:
        return
    sink.stage(int(batch_idx), int(structure_idx), node_features, edge_features)
```

`EncoderIntermediateStagingSink` stores items under `(batch_idx, structure_idx)` key.
`take_encoder_intermediates(batch_idx, structure_idx)` drains the entry or raises
`RuntimeError` if missing (matching `take_staging_sequences_logits` contract).

**Composition rule (R4 fix):** `encoder_sink_session()` and
`streaming_tensor_sink_session()` use separate ContextVars and can nest safely.
Both can be active simultaneously — they manage independent staging stores.
T1 must include a test confirming both can be active in the same `with` block.

### 3. `make_averaging_encode_fn` in `averaging.py`

Replace `get_averaged_encodings` + `make_encoding_sampling_split_fn` with a
single clean factory:

```python
def make_averaging_encode_fn(
    base_encode_fn: Callable[[InferenceBundle, PRNGKeyArray, InferenceConfig], EncoderOutput],
    spec: SamplingSpecification,
) -> Callable[[InferenceBundle, PRNGKeyArray, InferenceConfig], EncoderOutput]:
    """Wrap base_encode_fn to average encoder output over noise levels.

    Uses jax.lax.map (not Python for-loop) over noise levels to avoid N
    separate JAX trace compilations. Returns a closure with the same signature
    as base_encode_fn; callers are unaware of averaging.

    The returned closure:
    1. Builds N perturbed bundles (one per noise level in spec.backbone_noise)
    2. Maps base_encode_fn over them via jax.lax.map
    3. Averages node_features and edge_features across the noise axis
    4. Returns a single EncoderOutput with averaged features; neighbor_indices
       and mask taken from the first noise level (shape-invariant).

    PRECONDITION (R1): base_encode_fn must be a pure JAX-traceable function of
    (bundle, key, config). _apply_backbone_noise must produce a bundle with
    IDENTICAL pytree structure to its input — only the coordinates array changes.
    Any field that varies conditionally on noise value (e.g., a Python-level
    conditional on bundle state) will cause jax.lax.map to fail with a pytree
    structure mismatch. The implementer MUST verify this with an end-to-end
    jax.lax.map test over ≥2 noise levels using a real make_encode_fn(model)
    (not a stub). See T2 gate requirements.
    """
    noise_levels = jnp.asarray(spec.backbone_noise, dtype=jnp.float32)

    def averaging_encode_fn(
        bundle: InferenceBundle,
        key: PRNGKeyArray,
        config: InferenceConfig,
    ) -> EncoderOutput:
        def encode_at_noise(noise: jax.Array) -> EncoderOutput:
            noisy_bundle = _apply_backbone_noise(bundle, noise, key)
            return base_encode_fn(noisy_bundle, key, config)

        stacked: EncoderOutput = jax.lax.map(encode_at_noise, noise_levels)
        return EncoderOutput(
            node_features=jnp.mean(stacked.node_features, axis=0),
            edge_features=jnp.mean(stacked.edge_features, axis=0),
            neighbor_indices=stacked.neighbor_indices[0],
            mask=stacked.mask[0],
        )

    return averaging_encode_fn
```

**Key design note:** `ar_mask` is NOT included in `EncoderOutput` (stays at 4
fields). The driver re-derives `ar_mask` from `bundle.conditioning.ar_mask`
at decode time. The 5th element of the old `get_averaged_encodings` tuple is
discarded. T2 must assert `bundle.conditioning.ar_mask` is identical across
noise levels (shape and values) to confirm discard is safe.

**`jax.lax.map` rationale:** Using a Python for-loop or `jax.vmap` over noise
levels causes N separate trace compilations when noise levels change across
calls. `jax.lax.map` maps a single compiled function body over a leading axis,
keeping compile cost O(1).

### 4. Update `make_inference_plan` in `plan.py`

Do NOT introduce new `_make_base_encode_fn` / `_make_driver` helpers. Wrap the
existing `encode_fn` that `make_inference_plan` already constructs (via
`make_encode_fn(model, use_rolling_state=...)`) directly:

```python
def make_inference_plan(
    model: ModelProtocol,
    spec: SamplingSpecification,
) -> InferencePlan:
    stage_set = make_stage_set(model, spec)
    base_encode_fn = make_encode_fn(model, use_rolling_state=spec.use_rolling_state)

    if spec.average_node_features:
        encode_fn = make_averaging_encode_fn(base_encode_fn, spec)
    else:
        encode_fn = base_encode_fn

    # driver and other components unchanged from current make_inference_plan
    components = InferenceComponents(
        encode_fn=encode_fn,
        driver=<existing driver>,
        stage_set=stage_set,
    )
    return InferencePlan(model=model, components=components)
```

The `spec.average_node_features` check lives only here. All callers downstream
receive an `InferencePlan` that is opaque to averaging. The implementer reads
the actual current `make_inference_plan` body to fill in `<existing driver>`
without introducing new abstractions.

### 5. Restructure `_sample_batch` in `kernel_dispatch.py`

**New signature:**
```python
def _sample_batch(
    spec: SamplingSpecification,
    batched_ensemble: Protein,
    plan: InferencePlan,            # replaces model + stage_set
    *,
    canonical_structure_ids: Sequence[str] | None = None,
    batch_structure_ids: Sequence[str] | None = None,
    chunk_sample_start: int | None = None,
    chunk_sample_count: int | None = None,
    batch_idx: int = 0,
    structure_batch_count: int = -1,
    emit_structure_batch_io: bool = True,
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
```

**New `_call_kernel` inner closure:**

```python
def _call_kernel(key_samples, structure_idx, noise_val, temp_val):
    # Extract single structure
    c = batched_ensemble.coordinates[structure_idx]
    m = batched_ensemble.mask[structure_idx]
    ri = batched_ensemble.residue_index[structure_idx]
    ci = batched_ensemble.chain_index[structure_idx]
    fm = fixed_mask_for_vmap[structure_idx]
    ft = fixed_tokens_for_vmap[structure_idx]

    bundle, config = build_inference_bundle(
        coords=c, mask=m, residue_index=ri, chain_index=ci,
        backbone_noise=noise_val,
        fixed_mask=fm, fixed_tokens=ft,
        bias=..., tie_group_map=..., state_weights=...,
        ligand_coords=..., ligand_atom_types=..., ligand_mask=...,
        structure_mapping=..., temperature=temp_val,
        mode="sample_ar", use_rolling_state=spec.use_rolling_state,
        inference=True,
    )

    # Encode once per (structure, noise, temp)
    encode_key = jax.random.fold_in(key_samples[0], structure_idx)
    enc = plan.encode(bundle, encode_key, config)

    # Optional: fire encoder_sink if wired
    if plan.stage_set.encoder_sink is not None:
        plan.stage_set.encoder_sink(enc, jnp.int32(batch_idx), structure_idx)

    # Decode over samples
    def _run_one_sample(k):
        res = plan.decode(enc, bundle, k, config)
        return res.sequence, res.logits

    return _safe_map(_run_one_sample, key_samples, batch_size=samples_bs)
```

**STE strategy preservation (R3 fix — DEFER):** The current `resolve_kernel_fn`
dispatches `"straight_through"` to a closure wrapping `score_conditional_kernel`
and returning a `SampleResult`-shaped tuple. This is NOT a `sample_step`
(signature: `(logits, key) → tokens`) — it is a full kernel replacement with a
different decoder topology (teacher-forced + argmax). Relocating STE into
`make_stage_set.sample_step` would require restructuring the driver, which is
out of scope for COMP-UNIFIED.

**Decision: keep `resolve_kernel_fn` intact for STE.** The refactor in T4 must
preserve the existing `resolve_kernel_fn` call-path for `"straight_through"`.
When `plan.decode()` is implemented, it must check `spec.sampling_strategy` and
route `straight_through` through the existing closure, not through
`stage_set.sample_step`. STE integration into the composable pipeline is a
separate follow-up ticket (`COMP-STE`).

**Noise loop — no shape-collapsing optimization (R2 fix):** When
`spec.average_node_features`, the outer noise dispatch loop in `_sample_batch`
still iterates over `spec.backbone_noise`. `plan.encode()` internally applies
averaging over those same noise levels, so each iteration of the outer loop
produces an identical `enc` (idempotent but not wasteful — only one
`plan.encode()` call exists per `_call_kernel` invocation, so the averaging
happens once-per-call regardless). The outer loop is over structures × noises ×
temps; the `enc` produced is averaged, but the noise axis dimension in the
output tensor is preserved. Do NOT change `noises` to `jnp.zeros(1)` — this
would silently change the output tensor shape from `(B, D, T, N, L)` to
`(B, 1, T, N, L)`, breaking HDF5 schema and parity expectations.

For v2: if the noise axis is provably redundant under averaging, collapse it
explicitly with a reshape+broadcast after dispatch — not by silently changing
the dispatch dimensions.

### 6. Simplify `runner.py`

Remove lines 167–170:
```python
# REMOVE:
if spec.average_node_features:
    if spec.output_h5_path:
        return _sample_streaming_averaged(spec, protein_iterator, model, _sample_batch_averaged)
    return _sample_non_streaming_averaged(spec, protein_iterator, model)
```

Remove `_sample_non_streaming_averaged` function (lines 275–354).

Both `_sample_streaming` and the non-streaming loop now receive `plan` instead
of `model`:

```python
bound_sample_batch = functools.partial(_sample_batch, plan=plan)
```

The non-streaming loop call site:
```python
_, _, pseudo_perplexity = _sample_batch(
    spec, batched_ensemble, plan,
    stage_set=plan.stage_set,  # REMOVED — plan carries stage_set
    ...
)
```

### 7. Deprecate `_sample_streaming_averaged`

In `streaming.py`, mark `_sample_streaming_averaged` as deprecated:

```python
def _sample_streaming_averaged(*args, **kwargs):
    """Deprecated: use _sample_streaming with an averaging InferencePlan instead."""
    import warnings
    warnings.warn(
        "_sample_streaming_averaged is deprecated and will be removed in a future release. "
        "Use sample() with average_node_features=True (routed via unified _sample_streaming).",
        DeprecationWarning, stacklevel=2,
    )
    return _original_sample_streaming_averaged(*args, **kwargs)
```

### 8. Deprecation stubs for `_sampling_averaged.py` + `averaging.py` legacy exports

Five functions get deprecation stubs:
- `_sampling_averaged._internal_sample_averaged`
- `_sampling_averaged._sample_batch_averaged`
- `averaging.get_averaged_encodings`
- `averaging.make_encoding_sampling_split_fn`
- `averaging.make_encoding_conditional_logits_split_fn`

Each stub:
```python
def <name>(*args, **kwargs):
    import warnings
    warnings.warn("<name> is deprecated; use make_averaging_encode_fn instead.",
                  DeprecationWarning, stacklevel=2)
    return _original_<name>(*args, **kwargs)
```

Original implementations remain in place (not deleted) to avoid breakage in
any external callers. Scheduled for deletion in the next major release.

---

## File Touch List

| File | Change |
|------|--------|
| `src/prxteinmpnn/types/stages.py` | Add `EncoderSinkFn` protocol; add `encoder_sink` field to `StageSet` |
| `src/prxteinmpnn/host/output_sinks.py` | Add `IoCallbackEncoderSink`, `EncoderIntermediateStagingSink`, `_dispatch_encoder_intermediate_io`, `active_encoder_staging_sink`, `encoder_sink_session`, `take_encoder_intermediates` |
| `src/prxteinmpnn/host/averaging.py` | Add `make_averaging_encode_fn`; deprecation stubs for legacy functions |
| `src/prxteinmpnn/host/plan.py` | Update `make_inference_plan` to wrap `encode_fn` when `spec.average_node_features` |
| `src/prxteinmpnn/host/kernel_dispatch.py` | Restructure `_sample_batch`: `model + stage_set` → `plan: InferencePlan`; new `_call_kernel` with `plan.encode()` + optional `encoder_sink` |
| `src/prxteinmpnn/host/runner.py` | Remove `if spec.average_node_features` branch; remove `_sample_non_streaming_averaged`; pass `plan` to all call sites |
| `src/prxteinmpnn/host/streaming.py` | Deprecate `_sample_streaming_averaged`; update `_sample_streaming` to accept `plan` |
| `src/prxteinmpnn/host/_sampling_averaged.py` | Deprecation stubs for `_internal_sample_averaged`, `_sample_batch_averaged` |
| `tests/host/test_comp_unified_encoder_fusion.py` | New test file (see T8) |

---

## Task Breakdown

### T1: `EncoderSinkFn` + `encoder_sink` slot + sink infrastructure

**Files:** `types/stages.py`, `host/output_sinks.py`

- Add `EncoderSinkFn` protocol to `stages.py`; add to `__all__`
- Add `encoder_sink: EncoderSinkFn | None = None` to `StageSet`; update docstring topology rules
- Add to `output_sinks.py`:
  - `EncoderIntermediateStagingSink` class (stages under `(batch_idx, structure_idx)` key)
  - `_ENCODER_STAGING_SINK` ContextVar
  - `active_encoder_staging_sink()` accessor
  - `encoder_sink_session()` context manager
  - `_dispatch_encoder_intermediate_io(batch_idx, structure_idx, node_features, edge_features)` callback
  - `take_encoder_intermediates(batch_idx, structure_idx)` drain function
  - `IoCallbackEncoderSink(eqx.Module)` with `__call__(enc, batch_idx, structure_idx)` → `io_callback`

**Gate:**
- `isinstance(IoCallbackEncoderSink(), EncoderSinkFn)` is True (runtime_checkable)
- `EncoderSinkFn` uses same `TYPE_CHECKING`-guarded import pattern as existing protocols in `stages.py` (no forward-ref issues)
- `eqx.tree_at` round-trips a `StageSet` with `encoder_sink=None` (default) unchanged
- Grep `StageSet(` in codebase; all usages are keyword-arg form and absorb new default field cleanly
- Both `encoder_sink_session()` and `streaming_tensor_sink_session()` can be active simultaneously in the same `with` block (separate ContextVars)

### T2: `make_averaging_encode_fn` in `averaging.py`

**Files:** `host/averaging.py`

- Add `_apply_backbone_noise(bundle, noise, key) -> InferenceBundle` helper (perturbs coords)
- Add `make_averaging_encode_fn(base_encode_fn, spec) -> Callable`
  - Uses `jax.lax.map` over `noise_levels = jnp.asarray(spec.backbone_noise)`
  - Returns `EncoderOutput` with `mean(node_features)`, `mean(edge_features)`, `[0]` for nei/mask
- Add deprecation stubs for `get_averaged_encodings`, `make_encoding_sampling_split_fn`, `make_encoding_conditional_logits_split_fn`

**Gate (R1/R7/R8 required):**
1. Single noise level: averaged output == base_encode_fn output (jnp.allclose)
2. End-to-end `jax.lax.map` test with ≥3 noise levels using a REAL `make_encode_fn(model)` (not a stub) — must complete without pytree structure mismatch
3. `bundle.conditioning.ar_mask` is identical (shape + values) across all noise levels — confirms ar_mask discard is safe
4. `_apply_backbone_noise` produces a bundle with IDENTICAL pytree structure to input (only coords array changes; assert `jax.tree_util.tree_structure(noisy) == jax.tree_util.tree_structure(original)`)

### T3: Update `make_inference_plan`

**Files:** `host/plan.py`

- Add `_make_base_encode_fn(model) -> Callable` (wraps `model.encode` or equivalent)
- Add `_make_driver(model) -> Callable` (wraps existing driver dispatch)
- Update `make_inference_plan` to branch on `spec.average_node_features` as shown above

**Gate:** `make_inference_plan(model, spec_with_avg=True).encode(bundle, key, config)` returns
`EncoderOutput` with same shape as single-noise encode. Verify no new helper
functions (`_make_base_encode_fn`, `_make_driver`) were introduced — the body
wraps the existing `encode_fn` construction directly.

### T4: Restructure `_sample_batch`

**Files:** `host/kernel_dispatch.py`

- Change signature: `model, stage_set` → `plan: InferencePlan`
- Restructure `_call_kernel` as described in §5 above
- Do NOT change noise dispatch dimensions (see R2 fix in §5)
- Preserve `resolve_kernel_fn` call for STE; `plan.decode()` must route
  `"straight_through"` through the existing closure (see R3 fix in §5)
- Keep all io_callback emission logic (COMP-NEW) unchanged

**Gate:**
- Existing tests in `tests/host/test_sampling_tensor_batch_io.py` pass (monkeypatches updated to target `plan.encode` / `plan.decode`)
- `_sample_batch(spec_with_averaging, batch, plan_avg, ...)` output shape == `_sample_batch(spec_no_avg, batch, plan_no_avg, ...)` output shape
- `sampling_strategy="straight_through"` path still exercises `resolve_kernel_fn` (not `stage_set.sample_step`)

### T5: Simplify `runner.py`

**Files:** `host/runner.py`

- Remove `if spec.average_node_features:` block (lines 167–170)
- Remove `_sample_non_streaming_averaged` function
- Update non-streaming loop: `_sample_batch(spec, batch, plan, ...)` — drop `stage_set=plan.stage_set`
- Update streaming: `functools.partial(_sample_batch, plan=plan)` (drop `stage_set=plan.stage_set`)
- Update import block: remove `_sample_batch_averaged`, `_sample_non_streaming_averaged` usages
- Also remove stale import of `make_encoding_sampling_split_fn` from `averaging` (line 24 area)

**Gate:** `runner.sample(spec_with_avg=True)` takes the non-streaming path and
produces output with same shape as before. Verify `_sample_non_streaming_averaged`
does not appear anywhere in the module after cleanup.

### T6: Deprecate `_sample_streaming_averaged`

**Files:** `host/streaming.py`

- Wrap `_sample_streaming_averaged` in deprecation warning stub
- Existing body preserved as `_original_sample_streaming_averaged`

### T7: Deprecation stubs in `_sampling_averaged.py`

**Files:** `host/_sampling_averaged.py`

- Stubs for `_internal_sample_averaged`, `_sample_batch_averaged`

### T8: Tests

**File:** `tests/host/test_comp_unified_encoder_fusion.py`

Required tests:

1. `test_make_averaging_encode_fn_single_noise_matches_base` — with 1 noise
   level, averaged fn output == base fn output (exact equality via jnp.allclose);
   uses a REAL `make_encode_fn(model)` stub, not a pure Python mock
2. `test_make_averaging_encode_fn_lax_map_end_to_end` — with ≥3 noise levels,
   `jax.lax.map` completes without pytree structure mismatch; output has same
   shape as base fn output (averaging reduces noise axis); uses real `make_encode_fn`
3. `test_apply_backbone_noise_preserves_pytree_structure` — assert
   `jax.tree_util.tree_structure(noisy_bundle) == jax.tree_util.tree_structure(original_bundle)`
4. `test_ar_mask_invariant_across_noise_levels` — `bundle.conditioning.ar_mask`
   is identical (jnp.array_equal) across all noise perturbations (confirms ar_mask
   discard is safe in `EncoderOutput`)
5. `test_make_inference_plan_avg_wraps_encode_fn` — `spec.average_node_features=True`
   → `plan.components.encode_fn` is the averaging wrapper (inspect closure type)
6. `test_make_inference_plan_no_avg_base_encode_fn` — `spec.average_node_features=False`
   → `plan.components.encode_fn` is the bare `make_encode_fn` result (no extra wrapping)
7. `test_sample_batch_accepts_plan` — `_sample_batch(spec, batch, plan, ...)` does
   not raise; monkeypatch `plan.encode` + `plan.decode` with stubs
8. `test_encoder_sink_fires_when_wired` — `IoCallbackEncoderSink` wired in
   `stage_set.encoder_sink`; after `_sample_batch` + effects_barrier, `take_encoder_intermediates`
   returns staged data
9. `test_encoder_sink_no_op_when_none` — `stage_set.encoder_sink=None` (default);
   no sink activation; no `RuntimeError`
10. `test_encoder_sink_session_composes_with_streaming_sink` — both
    `encoder_sink_session()` and `streaming_tensor_sink_session()` active simultaneously;
    no interference
11. `test_runner_averaged_path_no_longer_branches` — after T5, call
    `runner.sample(spec_with_avg=True)`; assert `_sample_non_streaming_averaged`
    is not in the call stack (inspect via mock or source-level check)
12. `test_deprecation_warning_legacy_averaged_fns` — calling each of the 5 deprecated
    stubs emits `DeprecationWarning` with appropriate message
13. **PARITY TEST (R7 — required):** `test_averaged_path_parity_legacy_vs_unified`
    — pick a small protein structure (seq_len ≤ 20); run `_sample_batch_averaged`
    (legacy) and `_sample_batch(spec, batch, plan_avg)` (unified); assert output
    sequences and logits agree within tolerance (`rtol=1e-4, atol=1e-5`). This
    test confirms the refactor is semantically equivalent, not just shape-correct.

---

## Invariants (do not change)

- `InferenceBundle` and sub-bundles — JIT boundary, untouchable
- `StageSet` field names and semantics (`logit_transform`, `ar_logit_transform`,
  `decode_step`, `sample_step`, `tie_group_fuse`) — new `encoder_sink` only
- Kernel math (scatter logic, scan layouts in `driver.py`) — untouched
- `SamplerFn` / `ScoreFn` top-level signatures — unchanged
- `io_callback` ordering: always `ordered=False` per project policy
- `EncoderOutput` field count: stays at 4 (no `ar_mask`)

---

## Open Questions

1. **`_apply_backbone_noise` placement**: Should it live in `averaging.py` or
   `bundle_builder.py`? Leaning toward `bundle_builder.py` since noise perturbation
   is a bundle transformation, not an averaging concern. T2 implementer to decide.

2. **`encoder_sink` and `ordered=False`**: The `io_callback` for `encoder_sink`
   fires during `_call_kernel` per-structure, not at the end of the full batch.
   This means encoder intermediates arrive out of order if `structure_batch_count > 1`.
   `EncoderIntermediateStagingSink.take_encoder_intermediates()` should not enforce
   ordering. Caller must collect all intermediates after `effects_barrier()`.

3. **COMP-STE follow-up**: STE relocation into the composable pipeline (wiring
   through `stage_set.sample_step`) is deferred from COMP-UNIFIED. When STE
   integration is tackled, the key structural challenge is that STE is a full
   kernel replacement (teacher-forced decode), not a sampling rule. It likely
   warrants a separate `decode_step` variant rather than a `sample_step`.

---

## Dependencies

- COMP-NEW (DONE) — `streaming_tensor_sink_session` / `take_staging_sequences_logits` pattern
- COMP-532 (DONE) — `build_inference_bundle` splits into `(bundle, config)` return
- COMP-533 (DONE) — `make_stage_set` in `inference/logits.py`; `_sample_batch` accepts `stage_set`
- COMP-534 (DONE) — `make_inference_plan(model, spec)` called once in `runner.py`
- COMP-535 (DONE) — `plan.encode()` / `plan.decode()` on `InferencePlan`

All dependencies complete. No blockers.
