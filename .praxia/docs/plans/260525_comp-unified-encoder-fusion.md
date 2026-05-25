# COMP-UNIFIED: Encoder Fusion via InferencePlan Composability

**Date:** 2026-05-25  
**Ticket:** COMP-UNIFIED  
**Branch:** `refactor-full`  
**Status:** SPEC v3 — architecture corrected (2026-05-25)

---

## Summary

Eliminate the `if spec.average_node_features:` branch in `runner.py` and the
parallel averaged-path functions (`_sample_non_streaming_averaged`,
`_sample_streaming_averaged`, `_sample_batch_averaged`). Averaging becomes a
composable `EncodingFusionFn` stage wired into `StageSet.encoding_fusion`. When
set, the dispatch topology in `_sample_batch` restructures: the noise axis maps
over **encode only**, the fusion reduces D encoded outputs to 1, then decode runs
once on the fused `EncoderOutput`.

A new optional `encoder_sink: EncoderSinkFn | None` slot allows io_callback-based
staging of per-noise-level encoder intermediates before fusion.

---

## Motivation

### Current pain points

1. **Duplicated dispatch logic.** Three parallel paths exist for regular,
   averaged, and streaming-averaged sampling. All share >80% identical structure
   but diverge at encoding. Bug fixes must be applied to each.

2. **Hardcoded averaging.** `spec.average_node_features` is a boolean branch
   inside the host runner. There is no way to plug in a custom encoding fusion
   strategy (e.g., K cluster representatives, weighted ensemble).

3. **`_sample_batch` is opaque to `InferencePlan`.** Takes raw `model + stage_set`,
   cannot leverage the `encode_fn` / `decode` structure already on `InferencePlan`.

4. **No staged encoder intermediates.** Fused encoder features are discarded after
   decoding with no hook for persistence or downstream analysis.

---

## Architecture

### Key invariant: noise is not a Python loop

The noise axis in `_sample_batch` is dispatched via `_safe_map` — a `jax.vmap`
or `jax.lax.map` over a JAX array. `noise_val` inside `_call_kernel` is a traced
scalar, not a Python loop variable. This means:

**Without fusion:** The noise `_safe_map` maps over (encode + decode) atomically.
Output shape: `(B, D, T, N, L)`.

**With fusion:** The noise `_safe_map` maps over **encode only**. After producing
`stacked_enc: EncoderOutput` with leading D axis, `encoding_fusion(stacked_enc)`
reduces D → 1. Decode then runs once on the fused `EncoderOutput`. Output shape:
`(B, 1, T, N, L)` — noise_dim=1 is semantically correct (one fused encoding per
structure, not D separate ones). The existing transpose `(0,3,1,2,4)` still
applies; `D=1` passes through unchanged.

---

## Design

### 1. `EncodingFusionFn` protocol + `encoding_fusion` slot in `StageSet`

Add to `src/prxteinmpnn/types/stages.py`:

```python
class EncodingFusionFn(Protocol):
    """Reduce D encoded outputs (from noise dispatch) to a single fused EncoderOutput.

    Called after encoding at D noise levels, before decoding. Receives a stacked
    EncoderOutput with a leading D axis (one entry per noise level) and returns
    a single EncoderOutput with that axis removed.

    Implementations:
    - ArithmeticMeanEncodingFusion: element-wise mean over D
    - Future: weighted mean, cluster representatives (K=1 centroid), etc.
    """
    def __call__(self, stacked: EncoderOutput) -> EncoderOutput: ...
```

Add to `StageSet`:

```python
class StageSet(eqx.Module):
    logit_transform: BatchLogitFn | None = None
    ar_logit_transform: BatchLogitFn | None = None
    decode_step: ConditionalDecodeStep | UnconditionalDecodeStep | None = None
    sample_step: Any | None = None
    tie_group_fuse: TieGroupFuseFn | None = None
    encoder_sink: EncoderSinkFn | None = None   # NEW — fires per noise-level encoding
    encoding_fusion: EncodingFusionFn | None = None  # NEW — reduces D encodes → 1
```

Both new fields default to `None`. `None` is static in equinox; existing
`StageSet` instantiations without these fields pick up defaults cleanly.

**Topology rule additions (docstring):**
- If `encoding_fusion is not None` → restructured dispatch: noise axis maps encode
  only; fusion reduces before decode; `noise_dim=1` in output.
- If `encoder_sink is not None` → fires `io_callback` per noise-level encoding,
  inside the encode `_safe_map`, before fusion.

### 2. `ArithmeticMeanEncodingFusion` in `averaging.py`

```python
class ArithmeticMeanEncodingFusion(eqx.Module):
    """Average D encoded outputs element-wise over the leading noise axis."""

    def __call__(self, stacked: EncoderOutput) -> EncoderOutput:
        return EncoderOutput(
            node_features=jnp.mean(stacked.node_features, axis=0),
            edge_features=jnp.mean(stacked.edge_features, axis=0),
            neighbor_indices=stacked.neighbor_indices[0],
            mask=stacked.mask[0],
        )
```

No internal noise loop. No `_apply_backbone_noise`. The D encodings come from
the existing noise dispatch; this class only performs the reduction.

`ar_mask` is NOT in `EncoderOutput` (stays at 4 fields). The driver re-derives
it from `bundle.conditioning.ar_mask` at decode time. Confirm `ar_mask` is
noise-invariant (depends only on structure topology, not perturbed coordinates)
as part of T2 gate.

### 3. `EncoderSinkFn` protocol + `encoder_sink` slot + sink infrastructure

Add `EncoderSinkFn` to `src/prxteinmpnn/types/stages.py`:

```python
class EncoderSinkFn(Protocol):
    """Optional side-effect hook called once per noise-level encoding.

    When encoding_fusion is set, fires D times per structure (once per noise
    level), before fusion. When encoding_fusion is None, fires once per
    (structure, noise, temp) inside the standard _call_kernel.
    Implementations MUST use io_callback with ordered=False.
    """
    def __call__(
        self,
        enc: EncoderOutput,
        batch_idx: jax.Array,     # jnp.int32 scalar
        structure_idx: jax.Array, # jnp.int32 scalar
        noise_idx: jax.Array,     # jnp.int32 scalar — position in noise dispatch
    ) -> None: ...
```

Add to `output_sinks.py` — same `ContextVar` pattern as `StreamingTensorStagingSink`:

```python
class IoCallbackEncoderSink(eqx.Module):
    def __call__(self, enc, batch_idx, structure_idx, noise_idx) -> None:
        jax.experimental.io_callback(
            _dispatch_encoder_intermediate_io, None,
            jnp.int32(batch_idx), jnp.int32(structure_idx), jnp.int32(noise_idx),
            enc.node_features, enc.edge_features,
            ordered=False,
        )

_ENCODER_STAGING_SINK: ContextVar[EncoderIntermediateStagingSink | None] = \
    ContextVar("_ENCODER_STAGING_SINK", default=None)

def active_encoder_staging_sink() -> EncoderIntermediateStagingSink | None: ...
def encoder_sink_session() -> ContextManager[EncoderIntermediateStagingSink]: ...
def _dispatch_encoder_intermediate_io(batch_idx, structure_idx, noise_idx,
                                       node_features, edge_features) -> None: ...
def take_encoder_intermediates(batch_idx, structure_idx, noise_idx) -> tuple: ...
```

`EncoderIntermediateStagingSink` stages under `(batch_idx, structure_idx, noise_idx)` key.

**Composition:** `encoder_sink_session()` and `streaming_tensor_sink_session()`
use separate `ContextVar`s; both can be active simultaneously.

### 4. Update `make_inference_plan` in `plan.py`

Do NOT introduce new helper functions. Wrap the existing `encode_fn` construction
directly and wire `ArithmeticMeanEncodingFusion` into `stage_set.encoding_fusion`:

```python
def make_inference_plan(model, spec) -> InferencePlan:
    stage_set = make_stage_set(model, spec)

    if spec.average_node_features:
        stage_set = eqx.tree_at(
            lambda s: s.encoding_fusion,
            stage_set,
            ArithmeticMeanEncodingFusion(),
        )

    # encode_fn, driver unchanged from current make_inference_plan body
    components = InferenceComponents(
        encode_fn=<existing encode_fn>,
        driver=<existing driver>,
        stage_set=stage_set,
    )
    return InferencePlan(model=model, components=components)
```

The `spec.average_node_features` check lives only here. Downstream is opaque to it.

### 5. Restructure `_sample_batch` in `kernel_dispatch.py`

**New signature** — `model + stage_set` → `plan: InferencePlan`:

```python
def _sample_batch(
    spec, batched_ensemble, plan: InferencePlan, *,
    canonical_structure_ids=None, batch_structure_ids=None,
    chunk_sample_start=None, chunk_sample_count=None,
    batch_idx=0, structure_batch_count=-1, emit_structure_batch_io=True,
) -> tuple[ProteinSequence, Logits, jax.Array | None]:
```

**Two dispatch topologies based on `plan.stage_set.encoding_fusion`:**

`plan.stage_set.encoding_fusion is not None` is a static Python-level condition
(None vs. module, known at trace time) — safe to branch on.

**Path A — no fusion (existing topology, minimal change):**

```python
def _call_kernel(key_samples, structure_idx, noise_val, temp_val):
    bundle, config = build_inference_bundle(..., backbone_noise=noise_val, ...)
    encode_key = jax.random.fold_in(base_key, structure_idx)
    enc = plan.encode(bundle, encode_key, config)

    if plan.stage_set.encoder_sink is not None:
        plan.stage_set.encoder_sink(enc, jnp.int32(batch_idx), structure_idx,
                                     jnp.int32(0))  # noise_idx=0 in standard path

    def _run_one_sample(k):
        res = plan.decode(enc, bundle, k, config)
        return res.sequence, res.logits

    return _safe_map(_run_one_sample, key_samples, batch_size=samples_bs)

# Outer dispatch unchanged: structures → noises → temps → _call_kernel
sampled_sequences, sampled_logits = _safe_map(_dispatch_structure, ...)
# shape: (B, D, T, N, L)
```

**Path B — with fusion:**

```python
def _call_structure_fused(structure_idx):
    # extract structure data as before ...

    # Step 1: encode at each noise level, optionally fire encoder_sink
    def encode_at_noise(noise_val_and_idx):
        noise_val, noise_idx = noise_val_and_idx
        bundle, config = build_inference_bundle(..., backbone_noise=noise_val, ...)
        encode_key = jax.random.fold_in(base_key, structure_idx)
        enc = plan.encode(bundle, encode_key, config)
        if plan.stage_set.encoder_sink is not None:
            plan.stage_set.encoder_sink(enc, jnp.int32(batch_idx),
                                         structure_idx, jnp.int32(noise_idx))
        return enc

    noise_with_idx = (noises, jnp.arange(len(spec.backbone_noise)))
    stacked_enc = _safe_map(encode_at_noise, noise_with_idx, batch_size=noises_bs)

    # Step 2: fuse D encoded outputs → 1
    fused_enc = plan.stage_set.encoding_fusion(stacked_enc)

    # Step 3: decode (bundle for conditioning uses noise=0.0 — coords irrelevant)
    def _dispatch_temp(temp_val):
        decode_bundle, decode_config = build_inference_bundle(
            ..., backbone_noise=jnp.float32(0.0), temperature=temp_val, ...)

        def _run_one_sample(k):
            res = plan.decode(fused_enc, decode_bundle, k, decode_config)
            return res.sequence, res.logits

        return _safe_map(_run_one_sample, sample_keys, batch_size=samples_bs)

    return _safe_map(_dispatch_temp, temperatures, batch_size=temps_bs)

fused_seqs, fused_logits = _safe_map(
    _call_structure_fused, jnp.arange(batch_size), batch_size=structures_bs,
)
# shape: (B, T, N, L) — no noise axis
# Expand to (B, 1, T, N, L) for schema consistency with Path A
sampled_sequences = fused_seqs[:, None, ...]
sampled_logits = fused_logits[:, None, ...]
```

After this, the existing transpose `(0, 3, 1, 2, 4)` applies to both paths:
- Path A: `(B, D, T, N, L)` → `(B, N, D, T, L)`
- Path B: `(B, 1, T, N, L)` → `(B, N, 1, T, L)` — noise_dim=1, schema documented

**STE — defer (COMP-STE):** `resolve_kernel_fn("straight_through")` returns a full
kernel replacement (teacher-forced + argmax), not a `sample_step`. Do not relocate
into `make_stage_set`. `plan.decode()` must preserve the existing `resolve_kernel_fn`
routing for `"straight_through"`. STE composability is a separate follow-up.

### 6. Simplify `runner.py`

Remove:
```python
if spec.average_node_features:
    if spec.output_h5_path:
        return _sample_streaming_averaged(spec, protein_iterator, model, _sample_batch_averaged)
    return _sample_non_streaming_averaged(spec, protein_iterator, model)
```

Remove `_sample_non_streaming_averaged`. Pass `plan` everywhere:

```python
bound_sample_batch = functools.partial(_sample_batch, plan=plan)
# non-streaming loop:
_, _, pseudo_perplexity = _sample_batch(spec, batched_ensemble, plan, ...)
```

Remove stale imports: `_sample_batch_averaged`, `_sample_non_streaming_averaged`,
`make_encoding_sampling_split_fn`.

### 7. Deprecate `_sample_streaming_averaged`

Wrap in `DeprecationWarning` stub; preserve body as `_original_sample_streaming_averaged`.

### 8. Deprecation stubs

Five legacy functions get `DeprecationWarning` stubs, originals preserved:
- `_sampling_averaged._internal_sample_averaged`
- `_sampling_averaged._sample_batch_averaged`
- `averaging.get_averaged_encodings`
- `averaging.make_encoding_sampling_split_fn`
- `averaging.make_encoding_conditional_logits_split_fn`

---

## File Touch List

| File | Change |
|------|--------|
| `src/prxteinmpnn/types/stages.py` | Add `EncodingFusionFn` + `EncoderSinkFn` protocols; add `encoding_fusion` + `encoder_sink` fields to `StageSet` |
| `src/prxteinmpnn/host/output_sinks.py` | Add `IoCallbackEncoderSink`, `EncoderIntermediateStagingSink`, `_dispatch_encoder_intermediate_io`, `active_encoder_staging_sink`, `encoder_sink_session`, `take_encoder_intermediates` |
| `src/prxteinmpnn/host/averaging.py` | Add `ArithmeticMeanEncodingFusion`; deprecation stubs for legacy functions |
| `src/prxteinmpnn/host/plan.py` | Update `make_inference_plan`: wire `ArithmeticMeanEncodingFusion` into `stage_set.encoding_fusion` when `spec.average_node_features` |
| `src/prxteinmpnn/host/kernel_dispatch.py` | Restructure `_sample_batch`: `model + stage_set` → `plan`; two dispatch paths based on `stage_set.encoding_fusion` |
| `src/prxteinmpnn/host/runner.py` | Remove averaged branch + `_sample_non_streaming_averaged`; pass `plan` everywhere; remove stale imports |
| `src/prxteinmpnn/host/streaming.py` | Deprecate `_sample_streaming_averaged`; update `_sample_streaming` to accept `plan` |
| `src/prxteinmpnn/host/_sampling_averaged.py` | Deprecation stubs for `_internal_sample_averaged`, `_sample_batch_averaged` |
| `tests/host/test_comp_unified_encoder_fusion.py` | New test file (see T8) |

---

## Task Breakdown

### T1: `EncodingFusionFn` + `EncoderSinkFn` protocols + `StageSet` slots + sink infrastructure

**Files:** `types/stages.py`, `host/output_sinks.py`

- Add `EncodingFusionFn` protocol to `stages.py`; add to `__all__`
- Add `EncoderSinkFn` protocol to `stages.py`; add to `__all__`
- Add `encoding_fusion: EncodingFusionFn | None = None` to `StageSet`
- Add `encoder_sink: EncoderSinkFn | None = None` to `StageSet`; update topology rules docstring
- Add to `output_sinks.py`:
  - `EncoderIntermediateStagingSink` (stages under `(batch_idx, structure_idx, noise_idx)`)
  - `_ENCODER_STAGING_SINK` ContextVar + `active_encoder_staging_sink()` + `encoder_sink_session()`
  - `_dispatch_encoder_intermediate_io(batch_idx, structure_idx, noise_idx, node_features, edge_features)`
  - `take_encoder_intermediates(batch_idx, structure_idx, noise_idx)` — raises `RuntimeError` if missing
  - `IoCallbackEncoderSink(eqx.Module)` — calls `io_callback(_dispatch_encoder_intermediate_io, ..., ordered=False)`

**Gate:**
- `isinstance(IoCallbackEncoderSink(), EncoderSinkFn)` is True (`runtime_checkable`)
- `isinstance(ArithmeticMeanEncodingFusion(), EncodingFusionFn)` is True (check after T2)
- Grep `StageSet(` — all call sites use kwargs; new fields absorbed by defaults
- `eqx.tree_at` round-trip on a default `StageSet()` succeeds
- `encoder_sink_session()` + `streaming_tensor_sink_session()` simultaneously active — no interference

### T2: `ArithmeticMeanEncodingFusion` in `averaging.py`

**Files:** `host/averaging.py`

- Add `ArithmeticMeanEncodingFusion(eqx.Module)` as described in §2
- Add deprecation stubs for `get_averaged_encodings`, `make_encoding_sampling_split_fn`,
  `make_encoding_conditional_logits_split_fn`

**Gate:**
1. `ArithmeticMeanEncodingFusion()(stacked_enc)` where `stacked_enc` is built from
   a single noise level → output equals the un-stacked encoding (`jnp.allclose`)
2. `ArithmeticMeanEncodingFusion()(stacked_enc)` over D≥3 noise levels → output shape
   equals single-noise shape (D axis reduced)
3. `bundle.conditioning.ar_mask` is identical across noise perturbations for any
   given structure — confirms discarding from `EncoderOutput` is safe

### T3: Update `make_inference_plan`

**Files:** `host/plan.py`

- When `spec.average_node_features`: use `eqx.tree_at` to wire `ArithmeticMeanEncodingFusion()`
  into `stage_set.encoding_fusion` (do not introduce new helper functions)
- `encode_fn` and `driver` in `InferenceComponents` unchanged

**Gate:** `make_inference_plan(model, spec_avg=True).stage_set.encoding_fusion` is
an `ArithmeticMeanEncodingFusion` instance. `make_inference_plan(model, spec_avg=False).stage_set.encoding_fusion` is `None`.

### T4: Restructure `_sample_batch`

**Files:** `host/kernel_dispatch.py`

- Change signature: `model + stage_set` → `plan: InferencePlan`
- Implement Path A (no fusion) and Path B (with fusion) as described in §5
- Path B: `_safe_map(encode_at_noise, ...)` → `encoding_fusion(stacked_enc)` → decode
- `encoder_sink` fires inside the encode `_safe_map` in both paths (noise_idx tracked)
- Preserve `resolve_kernel_fn` for STE (see §5)
- All COMP-NEW `io_callback` emission logic unchanged

**Gate:**
- Existing `tests/host/test_sampling_tensor_batch_io.py` passes (update monkeypatches to target `plan`)
- `plan.stage_set.encoding_fusion is None` → output `shape[1] == len(spec.backbone_noise)`
- `plan.stage_set.encoding_fusion is not None` → output `shape[1] == 1`
- `sampling_strategy="straight_through"` still routes through `resolve_kernel_fn`

### T5: Simplify `runner.py`

**Files:** `host/runner.py`

- Remove `if spec.average_node_features:` block
- Remove `_sample_non_streaming_averaged`
- Pass `plan` to all `_sample_batch` call sites; remove `stage_set=plan.stage_set`
- Remove stale imports: `_sample_batch_averaged`, `_sample_non_streaming_averaged`,
  `make_encoding_sampling_split_fn`

**Gate:** `runner.sample(spec_avg=True)` completes; output `shape[1] == 1`.
No reference to `_sample_non_streaming_averaged` anywhere in module.

### T6: Deprecate `_sample_streaming_averaged`

**Files:** `host/streaming.py`

Deprecation wrapper stub; original body preserved as `_original_sample_streaming_averaged`.

### T7: Deprecation stubs in `_sampling_averaged.py`

**Files:** `host/_sampling_averaged.py`

Stubs for `_internal_sample_averaged`, `_sample_batch_averaged`.

### T8: Tests

**File:** `tests/host/test_comp_unified_encoder_fusion.py`

1. `test_arithmetic_mean_fusion_single_noise` — stacked from 1 noise → output equals base encoding
2. `test_arithmetic_mean_fusion_multi_noise_shape` — D≥3 noise levels → D axis reduced; output same shape as single
3. `test_ar_mask_invariant_across_noise_levels` — `bundle.conditioning.ar_mask` unchanged by noise perturbation
4. `test_make_inference_plan_wires_fusion_when_avg` — `spec.average_node_features=True` → `plan.stage_set.encoding_fusion` is `ArithmeticMeanEncodingFusion`
5. `test_make_inference_plan_no_fusion_when_no_avg` — `spec.average_node_features=False` → `plan.stage_set.encoding_fusion is None`
6. `test_sample_batch_path_a_noise_dim` — plan without fusion → `output.shape[1] == len(spec.backbone_noise)`
7. `test_sample_batch_path_b_noise_dim_1` — plan with `ArithmeticMeanEncodingFusion` → `output.shape[1] == 1`
8. `test_encoder_sink_fires_d_times_in_path_b` — `IoCallbackEncoderSink` wired; after `_sample_batch` + `effects_barrier()`, sink has D entries for the structure
9. `test_encoder_sink_no_op_when_none` — default `None` → no `RuntimeError`
10. `test_encoder_sink_session_composes_with_streaming_sink` — both contexts simultaneously active
11. `test_runner_averaged_path_removed` — `runner.sample` source contains no reference to `_sample_non_streaming_averaged`
12. `test_deprecation_warning_legacy_fns` — each of 5 deprecated stubs emits `DeprecationWarning`
13. **PARITY (required):** `test_averaged_path_parity_legacy_vs_unified` — small structure (seq_len ≤ 20); `_sample_batch_averaged` (legacy) vs `_sample_batch(plan_with_fusion)` (unified); sequences and logits agree within `rtol=1e-4, atol=1e-5`

---

## Invariants (do not change)

- `InferenceBundle` and sub-bundles — JIT boundary, untouchable
- `StageSet` existing 5 fields — only `encoding_fusion` + `encoder_sink` added
- Kernel math (scatter logic, scan layouts in `driver.py`) — untouched
- `SamplerFn` / `ScoreFn` top-level signatures — unchanged
- `io_callback` ordering: always `ordered=False`
- `EncoderOutput` field count: stays at 4 (no `ar_mask`)

---

## Open Questions

1. **`_call_structure_fused` encode key:** `encode_key = jax.random.fold_in(base_key, structure_idx)` is used for all D noise-level encodes. If the encoder uses the key for stochastic noise injection internally, all D encodes share the same key. This is the existing behaviour in Path A (one encode per `_call_kernel`). If D independent keys are needed, fold in `noise_idx` too: `jax.random.fold_in(jax.random.fold_in(base_key, structure_idx), noise_idx)`. T4 implementer to decide based on encoder internals.

2. **`encoder_sink` and ordering:** Fires D times per structure inside the noise `_safe_map`, with `ordered=False`. Intermediates arrive out of order. `take_encoder_intermediates` does not enforce ordering. Caller collects all after `effects_barrier()`.

3. **COMP-STE follow-up:** STE is a teacher-forced full kernel replacement, not a `sample_step`. Deferred. Likely warrants a separate `decode_step` variant when tackled.

---

## Dependencies

- COMP-NEW (DONE) — `streaming_tensor_sink_session` / io_callback emission in `_sample_batch`
- COMP-532 (DONE) — `build_inference_bundle` → `(bundle, config)`
- COMP-533 (DONE) — `make_stage_set` in `inference/logits.py`
- COMP-534 (DONE) — `make_inference_plan(model, spec)` in `runner.py`
- COMP-535 (DONE) — `plan.encode()` / `plan.decode()` on `InferencePlan`

All dependencies complete. No blockers.
