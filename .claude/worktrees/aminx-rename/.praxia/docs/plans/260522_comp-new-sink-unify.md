# COMP-NEW: Unify Result-Sink Topology

**Branch:** `refactor-full`
**Status:** Spec — do not implement until reviewed

---

## Background and motivation

The codebase has two paths for collecting sampled sequences and logits from device to host:

- **Non-streaming** (`runner.py`): appends raw JAX device arrays to `all_sequences` per batch, then concatenates after the loop.
- **Streaming** (`streaming.py`): expects io_callback to stage tensors into `StreamingTensorStagingSink`, then drains via `take_staging_sequences_logits` after `jax.effects_barrier()`.

The streaming path is **currently broken**: `_sample_batch` in `kernel_dispatch.py` never calls `_dispatch_sampling_tensor_batch_io` and never emits any io_callback for sequences or logits, despite accepting `emit_structure_batch_io: bool` in its signature. The streaming drain therefore always raises:

```
RuntimeError: Streaming tensor sink missing entry for key=...
```

The goal of this task is to:
1. Wire the missing io_callback emission into `_sample_batch`.
2. Rewire the non-streaming path in `runner.py` to drain via the same sink after each batch.
3. Fix the emit-gating of the structure-batch scalar marker in both chunk loops in `streaming.py`.
4. Add targeted tests proving the emission and drain contract.

Averaged paths (`_sample_batch_averaged`, `_sample_non_streaming_averaged`, `_sample_streaming_averaged`) are explicitly out of scope and must not be touched.

---

## Key contract: the io_callback key tuple

Every io_callback emission and every sink drain must agree on the key triple `(batch_idx, chunk_start, chunk_count)`. The contract is:

- `chunk_start` = `chunk_sample_start` when provided (campaign/streaming chunked path), or `0` when `chunk_sample_start is None` (non-chunked path).
- Non-campaign streaming (HDF5): pass `chunk_sample_start=key_chunk_start` explicitly from the caller so the emit key and drain key agree. See T3 §Non-campaign HDF5 path below.
- Non-streaming (runner.py): `chunk_start` is always `0` (no chunking); drain likewise uses `0`.

This contract must be maintained or the `RuntimeError` will recur in grid-mode non-campaign streaming.

---

## Scope boundaries

**In scope:**
- `src/aminx/host/kernel_dispatch.py`
- `src/aminx/host/runner.py`
- `src/aminx/host/streaming.py` (two chunk loops: HDF5 campaign + ArrayRecord campaign)
- `tests/host/test_sampling_tensor_batch_io.py` (create; see T4)

**Out of scope (explicit — do not touch):**
- `_sample_batch_averaged`, `_sample_non_streaming_averaged`, `_sample_streaming_averaged`
- Any model math or JAX tracing layout (`jax.lax.scan`, `lax.map`)
- HDF5 / ArrayRecord adapter internals beyond what's already wired
- Scoring path

---

## T1 — Add tensor io_callback emission to `_sample_batch`

**File:** `src/aminx/host/kernel_dispatch.py`

### Current state

`_sample_batch` ends at lines 211–224 with:

```python
# 6. Post-process (transpose to expected output shape: [batch, samples, noise, temp, seq_len])
# current: [B, D, T, N, L] -> desired: [B, N, D, T, L]
sampled_sequences = jnp.transpose(sampled_sequences, (0, 3, 1, 2, 4))
sampled_logits = jnp.transpose(sampled_logits, (0, 3, 1, 2, 4, 5))

# 7. IO & Metadata
if spec.compute_pseudo_perplexity:
    mask = batched_ensemble.mask
    if mask is None:
        mask = jnp.ones(batched_ensemble.coordinates.shape[:2], dtype=jnp.float32)
    pseudo_perplexity = compute_pseudo_perplexity(sampled_logits, sampled_sequences, mask)
    return sampled_sequences, sampled_logits, pseudo_perplexity

return sampled_sequences, sampled_logits, None
```

The `emit_structure_batch_io` parameter (line 87) is never used.

### Required change

After the transpose block (after line 214), before the perplexity block, insert a new section `# 7. io_callback emission`:

```python
# 7. io_callback emission — stage tensors to active sink (if any)
_effective_chunk_start = chunk_sample_start if chunk_sample_start is not None else 0

jax.experimental.io_callback(
    _dispatch_sampling_tensor_batch_io,
    None,
    jnp.int32(batch_idx),
    jnp.int32(structure_batch_count),
    jnp.int32(_effective_chunk_start),
    jnp.int32(target_num_samples),
    sampled_sequences,
    sampled_logits,
    ordered=False,
)

if emit_structure_batch_io:
    jax.experimental.io_callback(
        _noop_sampling_structure_batch_io,
        None,
        jnp.int32(batch_idx),
        jnp.int32(structure_batch_count),
        ordered=False,
    )
```

The perplexity block (step 8) and both `return` statements remain unchanged. The existing `return sampled_sequences, sampled_logits, pseudo_perplexity` and `return sampled_sequences, sampled_logits, None` stay — the non-streaming path will continue to use return-path values for its existing `all_sequences.append` until T2 rewires it.

### Import additions

Extend the existing `from aminx.host._sampling_helper import (...)` tuple to add the two new names:

```python
from aminx.host._sampling_helper import (
    _broadcast_per_structure,
    _dispatch_sampling_tensor_batch_io,
    _noop_sampling_structure_batch_io,
    _prepare_fixed_controls,
    _prepare_ligand_context,
)
```

Also add `import jax.experimental` at the top of the file with the other JAX imports.

### io_callback semantics

- `ordered=False` is required for both calls. Do not use `ordered=True`.
- Second positional arg to `io_callback` is `result_shape_dtypes`; pass `None` for void callbacks.
- `jax.effects_barrier()` is NOT called inside `_sample_batch`. It is the caller's responsibility.
- When no `streaming_tensor_sink_session` is active, `_dispatch_sampling_tensor_batch_io` checks `active_sampling_staging_sink() is None` and returns early (graceful no-op).

---

## T2 — Rewire non-streaming path in `runner.py`

**File:** `src/aminx/host/runner.py`

### Required change

1. Add imports:

```python
from aminx.host.output_sinks import streaming_tensor_sink_session, take_staging_sequences_logits
```

2. Wrap the entire non-streaming loop with `streaming_tensor_sink_session()`:

```python
with streaming_tensor_sink_session():
    for batch_idx, batched_ensemble in enumerate(protein_iterator):
        batch_size = batched_ensemble.coordinates.shape[0]
        batch_structure_ids = _structure_ids_for_batch(
            canonical_structure_ids,
            structure_offset=structure_offset,
            batch_size=batch_size,
        )
        target_for_batch = resolve_target_samples(spec, None, grid_lineage)
        _, _, pseudo_perplexity = _sample_batch(
            spec,
            batched_ensemble,
            model,
            stage_set=plan.stage_set,
            canonical_structure_ids=canonical_structure_ids,
            batch_structure_ids=batch_structure_ids,
            batch_idx=batch_idx,
            structure_batch_count=structure_batch_count,
        )
        StreamingBatchHost.sink_barrier()   # per-batch barrier (moved inside loop)
        sampled_sequences_np, sampled_logits_np = take_staging_sequences_logits(
            batch_idx,
            0,                        # chunk_start always 0 in non-streaming path
            target_for_batch,
        )
        all_sequences.append(jnp.asarray(sampled_sequences_np))
        if spec.return_logits and all_logits is not None:
            all_logits.append(jnp.asarray(sampled_logits_np))
        if pseudo_perplexity is not None:
            all_pseudo_perplexities.append(pseudo_perplexity)
        resolved_structure_ids.extend(batch_structure_ids)
        structure_offset += batch_size
# post-loop StreamingBatchHost.sink_barrier() — REMOVE (now per-batch inside loop)
```

3. Remove the post-loop `StreamingBatchHost.sink_barrier()` call.

**Rationale:** Per-batch barrier flushes all pending io_callbacks (tensor staging + scalar marker). Moving it inside the loop ensures sequences are drained before the next batch. Removing the post-loop call is safe because the final batch's per-loop barrier already covers it.

**Note on numpy→jnp roundtrip:** `jnp.asarray(sampled_sequences_np)` introduces a device→host→device roundtrip compared to the previous direct device-array append. This is intentional — it matches the streaming path's behavior. Downstream code (`pad_to_max`, `jnp.concatenate`) is unchanged.

---

## T3 — Wire `emit_structure_batch_io` in streaming chunk loops

**File:** `src/aminx/host/streaming.py`

There are **three** call sites that need the fix. All currently pass no `emit_structure_batch_io` argument (defaults to `True`, so scalar marker fires on every chunk — violating PR3a).

### T3.a — Non-campaign HDF5 path (line ~123)

This path has no chunking — one call per batch. Add `chunk_sample_start` to align emit/drain keys when `grid_lineage` is present, and make `emit_structure_batch_io=True` explicit:

```python
_, _, pseudo_perplexity = sample_batch_fn(
    spec,
    batched_ensemble,
    model,
    canonical_structure_ids=canonical_structure_ids,
    batch_structure_ids=batch_structure_ids,
    chunk_sample_start=key_chunk_start,      # NEW — aligns emit/drain key
    chunk_sample_count=key_chunk_count,       # NEW
    batch_idx=batch_idx,
    structure_batch_count=structure_batch_count_stream,
    emit_structure_batch_io=True,             # explicit (was default)
)
```

The drain call below already uses `key_chunk_start` and `key_chunk_count` and is unchanged.

Note: `chunk_sample_count` is intentionally omitted from the new kwargs — both `_sample_batch` (emit side) and `take_staging_sequences_logits` (drain side) derive `chunk_count` via `resolve_target_samples(spec, None, grid_lineage)`, guaranteeing key agreement without threading an extra parameter.

### T3.b — HDF5 campaign chunk loop (lines ~178–190)

Per PR3a, emit scalar marker only on the last chunk. Rewrite:

```python
chunks = list(StreamingBatchHost.iter_chunks(total_num_samples, chunk_size))
for chunk_idx, (chunk_start, chunk_count) in enumerate(chunks):
    chunk_sample_start = sample_start + chunk_start
    is_last_chunk = chunk_idx == len(chunks) - 1
    _, _, pseudo_perplexity = sample_batch_fn(
        spec,
        batched_ensemble,
        model,
        canonical_structure_ids=canonical_structure_ids,
        batch_structure_ids=batch_structure_ids,
        chunk_sample_start=chunk_sample_start,
        chunk_sample_count=chunk_count,
        batch_idx=batch_idx,
        structure_batch_count=structure_batch_count_stream,
        emit_structure_batch_io=is_last_chunk,     # NEW
    )
```

### T3.c — ArrayRecord campaign chunk loop (lines ~320–332)

Same fix as T3.b for `_sample_streaming_arrayrecord`:

```python
chunks = list(StreamingBatchHost.iter_chunks(total_num_samples, chunk_size))
for chunk_idx, (chunk_start, chunk_count) in enumerate(chunks):
    chunk_sample_start = sample_start + chunk_start
    is_last_chunk = chunk_idx == len(chunks) - 1
    _, _, _ = sample_batch_fn(
        spec,
        batched_ensemble,
        model,
        canonical_structure_ids=canonical_structure_ids,
        batch_structure_ids=batch_structure_ids,
        chunk_sample_start=chunk_sample_start,
        chunk_sample_count=chunk_count,
        batch_idx=batch_idx,
        structure_batch_count=structure_batch_count_ar,
        emit_structure_batch_io=is_last_chunk,     # NEW
    )
```

---

## T4 — Tests

**File:** `tests/host/test_sampling_tensor_batch_io.py` (new file; `tests/host/` already has `__init__.py`)

### Fixture strategy

`_sample_batch` calls `make_sampling_planner`, `_prepare_fixed_controls`, `_prepare_ligand_context`, and `build_inference_bundle` BEFORE the kernel is ever invoked. A `resolve_kernel_fn` monkeypatch alone does not bypass these. Use TWO monkeypatches:

1. **Patch `build_inference_bundle`** (in `kernel_dispatch` module namespace) to return a fixed `(bundle, config)` pair, bypassing all pre-kernel infrastructure:

```python
from unittest.mock import MagicMock

def _stub_build_inference_bundle(*args, **kwargs):
    return MagicMock(name="bundle"), MagicMock(name="config")
```

Monkeypatch target: `aminx.host.kernel_dispatch.build_inference_bundle`

2. **Patch `resolve_kernel_fn`** to return a stub kernel returning fixed-shape `SampleResult`:

```python
from aminx.inference.sample_autoregressive import SampleResult

def _make_stub_kernel(seq_len: int, vocab: int = 21):
    def _stub(model, prng_key, bundle, config, stage_set):
        import jax.numpy as jnp
        seq = jnp.zeros((seq_len,), dtype=jnp.int32)
        logits = jnp.zeros((seq_len, vocab), dtype=jnp.float32)
        return SampleResult(sequence=seq, logits=logits)
    return _stub
```

Monkeypatch target: `aminx.host.kernel_dispatch.resolve_kernel_fn`

With both patches active, a `SamplingSpecification` with minimal fields and a simple `Protein` namedtuple with correct array shapes are sufficient. Use:

```python
spec = SamplingSpecification(
    inputs=["/tmp/test.pdb"],
    checkpoint_id="ckpt_001",
    num_samples=2,
    temperature=[1.0],
    backbone_noise=[0.0],
    sampling_strategy="temperature",
    compute_pseudo_perplexity=False,
)
```

And a `Protein` namedtuple (or `MagicMock` with `.coordinates.shape` → `(1, 10, 4)`, `.mask.shape` → `(1, 10)`, `.residue_index.shape` → `(1, 10)`, `.chain_index.shape` → `(1, 10)`, `.mapping` → `None`).

### Test 1 — direct unit test of sink staging

```
test_dispatch_tensor_io_callback_stages_to_active_sink
```

With `streaming_tensor_sink_session()` active, directly call `_dispatch_sampling_tensor_batch_io(...)` with numpy arrays. Then call `take_staging_sequences_logits(0, 0, 4)` and assert returned arrays have expected shapes. Pure host-side test — no JAX tracing.

### Test 2 — io_callback stages when sink is active

```
test_sample_batch_stages_when_sink_active
```

Monkeypatch `resolve_kernel_fn` to return stub kernel. Build minimal `SamplingSpecification` and fake `Protein` namedtuple with shapes `(1, 10, 4)`, `(1, 10)` etc. Inside `streaming_tensor_sink_session()`:

```python
_, _, _ = _sample_batch(spec, batched_ensemble, model, stage_set=stage_set, batch_idx=0, structure_batch_count=1)
jax.effects_barrier()
seqs, logits = take_staging_sequences_logits(0, 0, spec.num_samples)
assert seqs is not None
assert logits is not None
```

Assert no `RuntimeError`.

### Test 3 — no error without active sink

```
test_sample_batch_does_not_raise_without_active_sink
```

Same setup. Call `_sample_batch` WITHOUT `streaming_tensor_sink_session()`. Call `jax.effects_barrier()`. Assert no error.

### Test 4 — `emit_structure_batch_io=False` suppresses scalar marker

```
test_emit_structure_batch_io_false_skips_scalar_marker
```

Monkeypatch `_noop_sampling_structure_batch_io` in the `kernel_dispatch` module namespace (the local reference held by `_sample_batch` after import) to count invocations. Call `_sample_batch(..., emit_structure_batch_io=False)`, `jax.effects_barrier()` — assert call count == 0. Then call with `emit_structure_batch_io=True`, `jax.effects_barrier()` — assert call count == 1.

**Monkeypatch target:** Patch `aminx.host.kernel_dispatch._noop_sampling_structure_batch_io` (the symbol in the `kernel_dispatch` module namespace, where `_sample_batch` holds a reference after import).

---

## Implementation order

1. **T1** (`kernel_dispatch.py`) — adds emission; streaming path unbreaks. Non-streaming path is unaffected.
2. **T3** (`streaming.py`) — fixes emit-gating for scalar marker. Depends on T1 so `emit_structure_batch_io` is consumed.
3. **T2** (`runner.py`) — rewires non-streaming to drain via sink. Depends on T1.
4. **T4** (tests) — Test 1 has no dependencies; Tests 2–4 depend on T1.

Commit after each task: `fix(COMP-NEW T1): add tensor io_callback to _sample_batch`, etc.

---

## Verification gates

COMP-NEW is done when:

1. Streaming HDF5 path (non-campaign and campaign) no longer raises `RuntimeError: Streaming tensor sink missing entry`.
2. Streaming ArrayRecord path likewise passes.
3. Non-streaming `sample()` returns identical shapes/dtypes to before.
4. `emit_structure_batch_io=False` suppresses scalar marker io_callback (Test 4 passes).
5. All four T4 tests pass.
6. Existing tests pass without regression:

```bash
uv run pytest tests/host/test_sampling_tensor_batch_io.py tests/host/ tests/sampling/test_sample.py -q
```

---

## Risk table

| Risk | Mitigation |
|---|---|
| Emit/drain key mismatch in non-campaign HDF5 + grid_lineage | T3.a adds `chunk_sample_start=key_chunk_start` to align keys |
| ArrayRecord campaign chunk loop emits scalar marker on every chunk | T3.c adds `emit_structure_batch_io=is_last_chunk` |
| MagicMock model breaks T4 tests 2–4 | All tests monkeypatch `resolve_kernel_fn` to return fixed-shape stub |
| numpy→jnp roundtrip in non-streaming path | `jnp.asarray` preserves dtype/shape; downstream logic unchanged |
| Monkeypatching `_noop_sampling_structure_batch_io` misses io_callback reference | Patch the symbol in `kernel_dispatch` module namespace |
| Post-loop `sink_barrier()` removal leaves open effects | Per-batch barrier covers the final batch; no effects left open |

---

## Reference: relevant source locations

| Symbol | File | Lines |
|---|---|---|
| `_sample_batch` | `src/aminx/host/kernel_dispatch.py` | 75–224 |
| `resolve_kernel_fn` | `src/aminx/host/kernel_dispatch.py` | 41–72 |
| Non-streaming loop | `src/aminx/host/runner.py` | 183–208 |
| `_sample_streaming` HDF5 non-campaign block | `src/aminx/host/streaming.py` | 120–158 |
| `_sample_streaming` HDF5 campaign chunk loop | `src/aminx/host/streaming.py` | 178–196 |
| `_sample_streaming_arrayrecord` chunk loop | `src/aminx/host/streaming.py` | ~320–332 |
| `_dispatch_sampling_tensor_batch_io` | `src/aminx/host/_sampling_helper.py` | 385–407 |
| `_noop_sampling_structure_batch_io` | `src/aminx/host/_sampling_helper.py` | 356–373 |
| `streaming_tensor_sink_session` | `src/aminx/host/output_sinks.py` | 104–116 |
| `take_staging_sequences_logits` | `src/aminx/host/output_sinks.py` | 119–129 |
| `active_sampling_staging_sink` | `src/aminx/host/output_sinks.py` | 132–134 |
| `StreamingTensorStagingSink` | `src/aminx/host/output_sinks.py` | 44–90 |
