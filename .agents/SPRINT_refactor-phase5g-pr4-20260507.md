# Sprint refactor-phase5-sampling-stream-20260507 (Phase 5g PR4)

**Goal:** Mirror Phase 5g PR3b (scoring tensor `io_callback`) on `run/sampling.py` `_sample_batch`.

**Land:**
- `_noop_sampling_tensor_batch_io` + `jax.experimental.io_callback(..., ordered=False)` after intra-batch `jnp.concatenate` of chunk sequences/logits; `stop_gradient` on operands.
- Structure-batch scalar hook unchanged (PR3a): only when `emit_structure_batch_io`.
- Tensor hook runs every `_sample_batch` exit (including campaign intermediate chunks).
- Streaming HDF5 / ArrayRecord: comments note PR4 + **second D2H** vs `np.asarray` until sink unify.

**Tests:** `tests/streaming/test_sampling_tensor_batch_io_callback.py`

**Verification:** `.agents/verification_logs/sprint_phase5g_pr4_sampling_tensor_20260507.filtered.txt`

**task_id:** `refactor-phase5-sampling-stream-20260507`
