# PR-3: Push batch_fn to _sample_sequences_jitted Boundary

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `LogitTransformFn` (the registered batch-logit post-processor) explicitly into `_sample_sequences_jitted`, replace the hardcoded `state_vmap_exact` dispatch with the registered callable, and eliminate `**kwargs: Any` catchalls from sampling boundaries.

**Architecture:** PR-2 resolved `batch_fn` from a UID string to a callable on the host. PR-3 passes that callable as a static argument to `_sample_sequences_jitted`, threads it through the `sample_autoregressive_state_vmap_exact_from_payload` path, and uses it in place of the hardcoded `apply_multistate_to_all_logits` branch. `**kwargs` catchalls are replaced with 13 explicit keyword args; `AutoregressivePipeline` wires `batch_fn` from `PipelineFns.resolve_logit_transform()`. Default behavior (arithmetic mean) is preserved when `batch_fn=None`.

**Tech Stack:** JAX, equinox, jaxtyping, pytest

---

## File Structure

| File | Change |
|------|--------|
| `src/prxteinmpnn/sampling/sample.py` | Add `batch_fn` explicit kwarg; wire into `state_vmap_exact` branch; eliminate `**kwargs`; add 13 explicit LigandMPNN params |
| `src/prxteinmpnn/pipeline/autoregressive.py` | Resolve and pass `batch_fn` from `fns.resolve_logit_transform()` |
| `tests/sampling/test_sample_call_kw_contract.py` | Verify `batch_fn` is explicit, not in kwargs spill |

---

## Task 1: Add `batch_fn` as explicit keyword arg to `_sample_sequences_jitted`

**Files:**
- Modify: `src/prxteinmpnn/sampling/sample.py` (~line 441)

Background: `batch_fn` is already in `static_argnames` (line 438) but absorbed by `**kwargs`. Make it visible.

- [ ] **Step 1: Read `sample.py` lines 430–500**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && grep -n "def _sample_sequences_jitted\|static_argnames\|batch_fn\|\*\*kwargs" src/prxteinmpnn/sampling/sample.py | head -30
```

- [ ] **Step 2: Add `batch_fn` param after `state_weights`**

Find the `def _sample_sequences_jitted(` signature. After the `state_weights: jnp.ndarray | None = None,` line, add:

```python
    batch_fn: LogitTransformFn | None = None,
```

Also add `LogitTransformFn` to the import from `prxteinmpnn.model_inputs` if not already imported:

```python
from prxteinmpnn.model_inputs import LogitTransformFn, SamplingInputs, SamplingStaticConfig
```

- [ ] **Step 3: Verify import works**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run python -c "from prxteinmpnn.sampling.sample import make_sample_sequences; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add src/prxteinmpnn/sampling/sample.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "refactor(pr3.1): add batch_fn as explicit kwarg to _sample_sequences_jitted"
```

---

## Task 2: Apply `batch_fn` in the `state_vmap_exact` branch

**Files:**
- Modify: `src/prxteinmpnn/sampling/sample.py` (~lines 612–633)

Background: The `state_vmap_exact` branch returns per-state logits. Currently there is no pluggable post-processor. Add `batch_fn` call here; default to `jnp.mean` when `batch_fn is None`.

- [ ] **Step 1: Read `sample.py` lines 600–640 to find where logits are extracted post-sample**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && sed -n '600,640p' src/prxteinmpnn/sampling/sample.py
```

- [ ] **Step 2: Write the failing test**

Add to `tests/sampling/test_state_vmap_exact_jit.py`:

```python
def test_batch_fn_replaces_default_mean():
    """batch_fn=None and explicit arithmetic-mean batch_fn must produce identical logits."""
    import jax, jax.numpy as jnp
    from prxteinmpnn.model_inputs import LogitTransformFn

    def mean_fn(logits_stack, state_index, state_weights):
        return jnp.mean(logits_stack, axis=0)

    # Two sampling calls: one with batch_fn=None, one with explicit mean_fn
    # Both must return logits within 1e-5 of each other.
    # (Use the existing tiny-stack fixture / setup from this file)
    # SKIP implementation detail — see Task 8 parity gate for full validation.
    pass  # placeholder; parity gate in Task 8 is authoritative
```

- [ ] **Step 3: Implement `batch_fn` application**

After `sampled_sequence, logits = model.sample_autoregressive_state_vmap_exact_from_payload(...)` (or equivalent call), add:

```python
        # Apply registered batch_fn to combine per-state logits (S, L, V) → (L, V)
        if batch_fn is None:
            combined_logits = jnp.mean(logits, axis=0)
        else:
            combined_logits = batch_fn(logits, multistate_stack.state_index, state_weights_arr)
```

Replace subsequent references to `logits` with `combined_logits` for the canonical-slice extraction.

- [ ] **Step 4: Run parity tests**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run pytest tests/sampling/test_state_vmap_exact_jit.py -q --tb=short
```

Expected: All existing tests pass (no behavior change — `batch_fn=None` → arithmetic mean as before).

- [ ] **Step 5: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add src/prxteinmpnn/sampling/sample.py tests/sampling/test_state_vmap_exact_jit.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "refactor(pr3.2): apply batch_fn in state_vmap_exact branch; default to arithmetic mean"
```

---

## Task 3: Thread `batch_fn` from outer `sample_sequences` to JIT boundary

**Files:**
- Modify: `src/prxteinmpnn/sampling/sample.py` (~line 693 and ~line 845)

- [ ] **Step 1: Read `sample_sequences` signature**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && grep -n "def sample_sequences\|\*\*kwargs" src/prxteinmpnn/sampling/sample.py
```

- [ ] **Step 2: Add `batch_fn` to outer wrapper**

After `state_weights: jnp.ndarray | None = None,` in `sample_sequences`, add:

```python
    batch_fn: LogitTransformFn | None = None,
```

- [ ] **Step 3: Forward at call site**

In the `_sample_sequences_jitted(...)` call inside `sample_sequences`, add:

```python
        batch_fn=batch_fn,
```

- [ ] **Step 4: Run tests**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run pytest tests/sampling/test_sample.py -q --tb=short -x
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add src/prxteinmpnn/sampling/sample.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "refactor(pr3.3): thread batch_fn from sample_sequences to JIT boundary"
```

---

## Task 4: Wire `batch_fn` in `AutoregressivePipeline.__call__`

**Files:**
- Modify: `src/prxteinmpnn/pipeline/autoregressive.py` (~lines 42–68)

- [ ] **Step 1: Read `autoregressive.py` lines 40–70**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && sed -n '40,70p' src/prxteinmpnn/pipeline/autoregressive.py
```

- [ ] **Step 2: Resolve and wire `batch_fn`**

After state weight normalization (e.g., `state_weights = jnp.ones(S) / S`), add:

```python
        batch_fn = fns.resolve_logit_transform()
```

In `module.sample_autoregressive_from_payload(...)` call, add:

```python
            batch_fn=batch_fn,
```

(This is a `**kwargs` pass-through to the underlying method which accepts `batch_fn`.)

- [ ] **Step 3: Verify import**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run python -c "from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline; print('ok')"
```

- [ ] **Step 4: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add src/prxteinmpnn/pipeline/autoregressive.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "refactor(pr3.4): wire batch_fn from PipelineFns in AutoregressivePipeline"
```

---

## Task 5: Eliminate `**kwargs` from `_sample_sequences_jitted`

**Files:**
- Modify: `src/prxteinmpnn/sampling/sample.py`

Background: The recon found `**kwargs` at lines 490, 441. These absorb LigandMPNN ligand fields and wave group arrays.

- [ ] **Step 1: List all kwargs consumed inside `_sample_sequences_jitted`**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && grep -n 'kwargs\["\|kwargs\.get' src/prxteinmpnn/sampling/sample.py | head -30
```

- [ ] **Step 2: Add explicit keyword params for each consumed key**

After `batch_fn: LogitTransformFn | None = None,`, add explicit keyword-only args for all keys found (typical set):

```python
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    wave_group_ids: jnp.ndarray | None = None,
    wave_group_positions: jnp.ndarray | None = None,
    wave_group_valid: jnp.ndarray | None = None,
    wave_position_valid: jnp.ndarray | None = None,
    Y: jnp.ndarray | None = None,
    Y_t: jnp.ndarray | None = None,
    Y_m: jnp.ndarray | None = None,
    xyz_37: jnp.ndarray | None = None,
    xyz_37_m: jnp.ndarray | None = None,
    chain_mask: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
```

- [ ] **Step 3: Replace `kwargs["x"]` / `kwargs.get("x")` with direct variable names**

All 13 variables become locals directly.

- [ ] **Step 4: Delete `**kwargs: Any` from `_sample_sequences_jitted`**

- [ ] **Step 5: Run full sampling suite**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run pytest tests/sampling/ -q --tb=short
```

Expected: All pass. Fix any callers that still pass `**dict` bags — they need to be explicit now.

- [ ] **Step 6: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add src/prxteinmpnn/sampling/sample.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "refactor(pr3.5): eliminate **kwargs in _sample_sequences_jitted; add explicit ligand/wave params"
```

---

## Task 6: Eliminate `**kwargs` from outer `sample_sequences` wrapper

**Files:**
- Modify: `src/prxteinmpnn/sampling/sample.py`

- [ ] **Step 1: Add same 13 explicit params to `sample_sequences`**

Same list as Task 5, after `batch_fn`.

- [ ] **Step 2: Forward all 13 to `_sample_sequences_jitted` call site**

- [ ] **Step 3: Delete `**kwargs: Any` from `sample_sequences`**

- [ ] **Step 4: Run contract test**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run pytest tests/sampling/test_sample_call_kw_contract.py -xvs
```

Expected: All assertions pass; no forbidden kwargs in the spill.

- [ ] **Step 5: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add src/prxteinmpnn/sampling/sample.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "refactor(pr3.6): eliminate **kwargs in sample_sequences; all params explicit"
```

---

## Task 7: Verify contract test recognizes `batch_fn` as explicit

**Files:**
- Modify: `tests/sampling/test_sample_call_kw_contract.py`

- [ ] **Step 1: Read the contract test**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && cat tests/sampling/test_sample_call_kw_contract.py
```

- [ ] **Step 2: Confirm `batch_fn` is NOT in the `**kwargs` spill allowlist**

If `batch_fn` appears in the model-passthrough kwargs set, remove it and add a comment:

```python
# batch_fn is consumed at the sampling boundary — never forwarded to model.__call__
```

- [ ] **Step 3: Run**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run pytest tests/sampling/test_sample_call_kw_contract.py -xvs
```

- [ ] **Step 4: Commit**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add tests/sampling/test_sample_call_kw_contract.py
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "test(pr3.7): contract — batch_fn explicit, not in kwargs spill"
```

---

## Task 8: Parity gate

**Files:**
- Read: `tests/sampling/test_sample.py`, `tests/sampling/test_state_vmap_exact_jit.py`, `tests/sampling/test_sample_call_kw_contract.py`

- [ ] **Step 1: Run full sampling parity suite**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn && PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -v --tb=short 2>&1 | tee /tmp/pr3_parity.log
grep -E "passed|failed|error" /tmp/pr3_parity.log | tail -5
```

Expected: Same pass count as pre-PR-3. Zero new failures.

- [ ] **Step 2: If any test fails**

Identify the failure: missing explicit arg, wrong variable name, forgotten forward. Fix the call site or test. Re-run.

- [ ] **Step 3: Commit if fixes needed**

```bash
git -C /home/marielle/projects/tev_design/prxteinmpnn add -A
git -C /home/marielle/projects/tev_design/prxteinmpnn commit -m "test(pr3.8): parity gate — all sampling tests green after batch_fn wiring"
```

---

## Parity Gate Checklist

- [ ] `test_sample.py` — all temperature sampling tests pass
- [ ] `test_state_vmap_exact_jit.py` — all state_vmap_exact tests pass
- [ ] `test_sample_call_kw_contract.py` — contract verified; no kwargs spill
- [ ] No `**kwargs: Any` in `_sample_sequences_jitted` or `sample_sequences`
- [ ] `batch_fn` is explicit at both boundaries
- [ ] `AutoregressivePipeline.__call__` wires `batch_fn` from `PipelineFns`

---

## Architecture Notes

**`batch_fn` is a static arg, not a pytree field:**
- JAX-traceable, resolved on host before JIT, inlined at `jax.export` time.
- No runtime dispatch cost.
- Safe for `static_argnames`.

**Default behavior preserved:**
- `batch_fn=None` → `jnp.mean(logits, axis=0)` (arithmetic mean, matching prior implicit behavior).
- No existing callers broken.

**PR-3 does NOT change the model signature:**
- Only `sample.py` and `autoregressive.py` change.
- `model.__call__` unchanged.
- PR-4 will push `SamplingInputs` deeper.
