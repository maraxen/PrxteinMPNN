# Sprint `refactor-phase5-sink-unify-20260507` (Phase 5g — host sink unify)

**Signed-off scope:** Remove redundant second D2H on sampling **streaming** paths (HDF5 + ArrayRecord campaign) by draining host tensors recorded in the PR4 tensor `io_callback` after each `jax.effects_barrier()`, keyed by `(batch_idx, chunk_sample_start, chunk_sample_count)` so campaign chunk loops stay unambiguous when `batch_idx` repeats.

**Non-goals:** `OUTPUT_SINKS` / `DesignSink` registry; scoring paths; `ordered=True`; per-chunk tensor hook inside the chunk loop (defer).

**Implementation:** Single trace-stable `_dispatch_sampling_tensor_batch_io` registered with `jax.experimental.io_callback`; when `ContextVar` sink is active (streaming loops only), record NumPy payloads under the composite key; otherwise delegate to `_noop_sampling_tensor_batch_io` (preserves test monkeypatches). Extend tensor hook operands with `chunk_start` / `chunk_count` scalars (matches streaming loop contract). Perplexity remains device-returned + `np.asarray` in slice 1 (`compute_pseudo_perplexity` stays after concat per `TODO_io_callback.txt`).

**Verification:** `uv run ty check`, `uv run ruff check .`, `PYTHONPATH=src uv run pytest tests/streaming/test_sampling_*.py tests/streaming/test_io_callback_ordering.py -q`; filtered log under `.agents/verification_logs/`.

**Plan-auditor:** NEEDS_WORK addressed via this sprint doc + keyed queue + stable dispatch symbol.
