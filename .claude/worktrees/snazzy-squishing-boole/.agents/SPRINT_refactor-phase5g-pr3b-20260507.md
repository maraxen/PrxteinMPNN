# Sprint: Phase 5g PR3b — scoring tensor `io_callback`

**task_id:** `refactor-phase5-pr3b-sprint-20260507`  
**Plan gate:** plan-auditor **PASS** (2026-05-07)  
**Status:** **landed** in-repo (2026-05-07); verification logs under `.agents/verification_logs/sprint_phase5g_pr3b_20260507.*`.

## Goal

Add optional host-visible **tensor** D2H via `jax.experimental.io_callback(..., ordered=False)` for per-batch `scores` and `logits` in `src/prxteinmpnn/run/scoring.py`, alongside existing scalar structure-batch markers (PR2b/c). Default no-op hook; tests monkeypatch.

## Hook

- `_noop_scoring_tensor_batch_io(batch_idx, batch_count, scores_host, logits_host) -> None`
- Call sites pass `jax.lax.stop_gradient` on all four operands.
- **Never** `ordered=True`.

## Code touchpoints

1. `_score_standard_mode` — inside traced `_compute`, after scalar `io_callback`.
2. `_score_batch_averaged` — after scalar `io_callback`, before `return`.
3. `_score_streaming` — inside `_compute`, after scalar `io_callback`. Keep `effects_barrier()` and HDF5 `np.asarray` unchanged (**double D2H** acceptable for this slice; comment/TODO).

## Tests

- New: `tests/streaming/test_scoring_tensor_batch_io_callback.py`
- Standard + averaged paths: accumulate host tensors from hook; after `score()`, `allclose` vs `result["scores"]` / `result["logits"]` using `get_tolerances(jnp.float32)`.

## Verification

```bash
LOG=.agents/verification_logs/sprint_phase5g_pr3b_20260507.log
mkdir -p "$(dirname "$LOG")"
(uv run pytest tests/streaming/test_scoring_batch_io_callback.py \
  tests/streaming/test_scoring_tensor_batch_io_callback.py && \
  uv run ty check && \
  uv run ruff check .) >"$LOG" 2>&1
grep -E '(PASSED|FAILED|ERROR|All checks passed|Found [[:digit:]]+ error)' "$LOG" \
  > .agents/verification_logs/sprint_phase5g_pr3b_20260507.filtered.txt || true
```

## Out of scope

HDF5 vs ArrayRecord adapter unification; removing redundant streaming `np.asarray`.
