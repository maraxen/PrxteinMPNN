# Sprint A: ARLogitTransformFn — AR Wave Scan LogitTransform Wiring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire a new `ARLogitTransformFn` protocol through the autoregressive wave-parallel scan, replacing the hardcoded `combine_logits_multistate_idx` call inside the `contrib()` closure, so callers can supply custom per-position multistate fusion logic.

**Architecture:** `ARLogitTransformFn` operates on `(S, V)` per decode position — one position at a time — which is the natural granularity of the AR scan body. This is distinct from `LogitTransformFn` which operates on `(S, L, V)` (full sequence). Both coexist. `PipelineFns` gains an optional `ar_logit_transform_uid`. When `ar_logit_transform_fn=None`, the existing `combine_logits_multistate_idx` default path is preserved exactly.

**Tech Stack:** JAX, jaxtyping, equinox, cloudpickle (UID hashing), pytest

---

## File Structure

**Modified:**
- `src/prxteinmpnn/model_inputs.py` — add `ARLogitTransformFn` protocol, update `__all__`
- `src/prxteinmpnn/pipeline_fns.py` — add `ar_logit_transform_uid: str | None = None`, add `resolve_ar_logit_transform()` method
- `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py` — add `ar_logit_transform_fn` param, replace `combine_logits_multistate_idx` in `contrib()` (lines 273–281)
- `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py` — same (lines 239–247)
- `src/prxteinmpnn/model/mpnn.py` — thread through `sample_autoregressive_state_vmap_exact` and `sample_autoregressive_state_vmap_exact_from_payload`
- `src/prxteinmpnn/model/ligand_mpnn.py` — same for LigandMPNN methods
- `src/prxteinmpnn/pipeline/autoregressive.py` — resolve and pass `fns.ar_logit_transform_uid`
- `tests/test_pipeline_fns.py` — add `ARLogitTransformFn` importable test

**Created:**
- `tests/pipeline/test_autoregressive_logit_transform.py` — behavioral tests for AR logit transform threading

---

## Task 1: Add `ARLogitTransformFn` protocol to `model_inputs.py`

**Files:**
- Modify: `src/prxteinmpnn/model_inputs.py`
- Test: `tests/test_pipeline_fns.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_fns.py`:

```python
def test_ar_logit_transform_fn_importable():
    from prxteinmpnn.model_inputs import ARLogitTransformFn
    assert ARLogitTransformFn is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_ar_logit_transform_fn_importable -v
```

Expected: FAIL — `ImportError: cannot import name 'ARLogitTransformFn'`

- [ ] **Step 3: Add `ARLogitTransformFn` to `model_inputs.py`**

After the `LogitTransformFn` class definition (around line 97), add:

```python
class ARLogitTransformFn(Protocol):
  """JAX-traceable fn combining per-state logits for ONE decode position into a single vector.

  Called per decode step inside the AR wave-parallel scan, where logits are accumulated
  one position at a time (shape (S, V)), not across the full sequence.
  Contrast with LogitTransformFn which operates on (S, L, V).

  Must use only jnp ops — no Python branching on traced values.
  state_weights is always a concrete array (uniform 1/S if absent).
  """

  def __call__(
    self,
    state_logits: Float[Array, "S V"],
    state_index: Int[Array, "S"],
    state_weights: Float[Array, "S"],
  ) -> Float[Array, "V"]: ...
```

Update `__all__` (around line 99) to include `"ARLogitTransformFn"`.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_ar_logit_transform_fn_importable -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/model_inputs.py tests/test_pipeline_fns.py
git commit -m "feat(sprint-A): add ARLogitTransformFn protocol to model_inputs"
```

---

## Task 2: Add `ar_logit_transform_uid` to `PipelineFns`

**Files:**
- Modify: `src/prxteinmpnn/pipeline_fns.py`
- Test: `tests/test_pipeline_fns.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_fns.py`:

```python
def test_pipeline_fns_has_ar_logit_transform_uid():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert hasattr(fns, "ar_logit_transform_uid")
    assert fns.ar_logit_transform_uid is None


def test_pipeline_fns_resolve_ar_logit_transform_returns_none_by_default():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert fns.resolve_ar_logit_transform() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_pipeline_fns_has_ar_logit_transform_uid tests/test_pipeline_fns.py::test_pipeline_fns_resolve_ar_logit_transform_returns_none_by_default -v
```

Expected: FAIL — `AttributeError: 'PipelineFns' object has no attribute 'ar_logit_transform_uid'`

- [ ] **Step 3: Update `PipelineFns`**

In `src/prxteinmpnn/pipeline_fns.py`, add field and method:

```python
@dataclasses.dataclass(frozen=True)
class PipelineFns:
  logit_transform_uid: str
  encoder_pre_process_uid: str | None = None
  encoder_post_process_uid: str | None = None
  ar_logit_transform_uid: str | None = None          # <-- add this

  # ... existing methods ...

  def resolve_ar_logit_transform(self) -> ARLogitTransformFn | None:
    if self.ar_logit_transform_uid is None:
      return None
    return resolve_hook(self.ar_logit_transform_uid)
```

Add `ARLogitTransformFn` to the `TYPE_CHECKING` import block at the top of `pipeline_fns.py`. The import source is `prxteinmpnn.model_inputs` (where the protocol is defined in Task 1), NOT `prxteinmpnn.protocols`:

```python
if TYPE_CHECKING:
    from prxteinmpnn.protocols import EncoderPostFn, EncoderPreFn, LogitTransformFn
    from prxteinmpnn.model_inputs import ARLogitTransformFn  # <-- add
```

First, add a typed registrar alias to `src/prxteinmpnn/pipeline_registry.py` (following the existing pattern for `register_logit_transform_fn`):

```python
def register_ar_logit_transform_fn(fn: Any, *, name: str | None = None) -> str:
    """Typed alias for register_hook for ARLogitTransformFn callables."""
    return register_hook(fn, name=name)
```

Add `"register_ar_logit_transform_fn"` to `pipeline_registry.__all__`.

Then update `from_callables` in `pipeline_fns.py`. The existing signature uses keyword-only params (`*` after `cls`) — preserve this:

```python
@classmethod
def from_callables(
  cls,
  *,                                                         # keyword-only marker — keep
  logit_transform: LogitTransformFn | None = None,
  encoder_pre_process: EncoderPreFn | None = None,
  encoder_post_process: EncoderPostFn | None = None,
  ar_logit_transform: ARLogitTransformFn | None = None,  # <-- add
) -> PipelineFns:
    # ... existing registration ...
    ar_uid = register_ar_logit_transform_fn(ar_logit_transform) if ar_logit_transform is not None else None
    return cls(
      logit_transform_uid=lt_uid,
      encoder_pre_process_uid=pre_uid,
      encoder_post_process_uid=post_uid,
      ar_logit_transform_uid=ar_uid,
    )
```

Add `register_ar_logit_transform_fn` to the imports from `pipeline_registry` in `pipeline_fns.py`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/pipeline_fns.py tests/test_pipeline_fns.py
git commit -m "feat(sprint-A): add ar_logit_transform_uid to PipelineFns"
```

---

## Task 3: Thread `ar_logit_transform_fn` through `run_sample_autoregressive_state_vmap_exact`

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py`
- Test: `tests/pipeline/test_autoregressive_logit_transform.py`

Background: The fusion site is in `contrib()` at lines 273–281. `combine_logits_multistate_idx` takes `lrows_fin: (S, V)` and `cmask_fin: (S,)` and returns `(1, V)`. The replacement: when `ar_logit_transform_fn` is not None, call it on `lrows_fin * cmask_fin[:, None]` (zero out invalid state rows) with `state_index=row_state_map` and `state_weights=sw_use`, get `(V,)`. When None, use the existing call unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_autoregressive_logit_transform.py`:

```python
"""Verify ARLogitTransformFn is threaded through the AR scan."""
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model_inputs import ARLogitTransformFn
from prxteinmpnn.payloads import MultistateStackPayload
import equinox as eqx


def _make_model():
    return eqx.tree_inference(
        PrxteinMPNN(16, 16, 16, 1, 1, 4, key=jax.random.PRNGKey(0)),
        value=True,
    )


def _make_stack(S=2, L=4):
    return MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.stack([jnp.arange(L, dtype=jnp.int32) + i * L for i in range(S)]),
        flat_row_offsets=jnp.array([0, L], dtype=jnp.int32),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=S * L,
    )


def _build_wave_tables(S, L, k_neighbors=4):
    """Build wave tables for S identical states with L canonical positions."""
    import numpy as np
    from prxteinmpnn.utils.wave_parallel import compute_wave_assignments
    from prxteinmpnn.sampling.state_vmap_prep import remap_wave_positions_flat_to_local

    # One canonical position per tie group; S members per group
    tie_flat = np.tile(np.arange(L, dtype=np.int32), S)  # (S*L,)
    flat_offsets = np.array([s * L for s in range(S + 1)], dtype=np.int32)  # S+1 elements
    git = np.full((L, S), -1, dtype=np.int32)
    gvt = np.zeros((L, S), dtype=bool)
    for g in range(L):
        for s in range(S):
            git[g, s] = int(flat_offsets[s] + g)
            gvt[g, s] = True
    ca0 = np.zeros((L, 4, 3), dtype=np.float32)
    w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=k_neighbors, n_canonical=L)
    w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, flat_offsets)
    return (
        jnp.asarray(w_id, dtype=jnp.int32),
        jnp.asarray(w_loc, dtype=jnp.int32),
        jnp.asarray(w_gv),
        jnp.asarray(w_pv2),
    )


def test_ar_logit_transform_fn_changes_output():
    """ARLogitTransformFn must be on the runtime compute path — output must differ."""
    m = _make_model()
    S, L, V = 2, 4, 21
    stack = _make_stack(S=S, L=L)
    w_id, w_loc, w_gv, w_pv = _build_wave_tables(S, L, k_neighbors=4)
    ar_mask = jnp.zeros((S, L, L), dtype=jnp.float32)
    bias = jnp.zeros((S, L, 21), dtype=jnp.float32)
    key = jax.random.PRNGKey(1)

    # Run 1: default path (no transform)
    seqs_default, _ = m.sample_autoregressive_from_payload(
        key, stack, ar_mask, bias, 1.0, 0, None, w_id, w_loc, w_gv, w_pv,
    )

    # Run 2: transform that forces logits strongly toward token 0
    def always_token_zero(state_logits, state_index, state_weights):
        return jnp.zeros(V).at[0].set(1e9)

    seqs_forced, _ = m.sample_autoregressive_from_payload(
        key, stack, ar_mask, bias, 1.0, 0, None, w_id, w_loc, w_gv, w_pv,
        ar_logit_transform_fn=always_token_zero,
    )
    # With 1e9 logit on token 0, sampling must produce all zeros
    assert jnp.all(seqs_forced == 0), "Transform on runtime path must force all-zero sequences"
    # Default path should differ (model produces non-trivial logits)
    assert not jnp.all(seqs_default == seqs_forced), "Default and transformed outputs must differ"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive_logit_transform.py::test_ar_logit_transform_fn_changes_output -v
```

Expected: FAIL — `TypeError: unexpected keyword argument 'ar_logit_transform_fn'`

- [ ] **Step 3: Add `ar_logit_transform_fn` param to `run_sample_autoregressive_state_vmap_exact`**

In `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py`:

1. Add import at top (TYPE_CHECKING block):
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prxteinmpnn.model_inputs import ARLogitTransformFn
```

2. Add parameter to function signature after `state_weights`:
```python
def run_sample_autoregressive_state_vmap_exact(
  ...
  state_weights: jnp.ndarray | None,
  ...
  ar_logit_transform_fn: "ARLogitTransformFn | None" = None,
) -> tuple[OneHotProteinSequence, Logits]:
```

3. In `contrib()` closure, replace lines 273–281:

**Before:**
```python
combined = combine_logits_multistate_idx(
    lrows_fin,
    cmask_fin.astype(jnp.bool_),
    strat_idx,
    ms_temp,
    sw_use,
    row_state_map,
)
comb_vec = jnp.squeeze(combined, axis=0).astype(log_dtype)
```

**After:**
```python
if ar_logit_transform_fn is not None:
    comb_vec = ar_logit_transform_fn(
        lrows_fin * cmask_fin[:, jnp.newaxis].astype(lrows_fin.dtype),
        row_state_map,
        sw_use,
    ).astype(log_dtype)
else:
    combined = combine_logits_multistate_idx(
        lrows_fin,
        cmask_fin.astype(jnp.bool_),
        strat_idx,
        ms_temp,
        sw_use,
        row_state_map,
    )
    comb_vec = jnp.squeeze(combined, axis=0).astype(log_dtype)
```

Note: `lrows_fin * cmask_fin[:, None]` zeros out rows from states not in this decode group before passing to the transform. `row_state_map` is `jnp.arange(max_gs, dtype=jnp.int32)` (the state indices for this slot).

- [ ] **Step 4: Thread `ar_logit_transform_fn` through `sample_autoregressive_state_vmap_exact` in `mpnn.py`**

In `src/prxteinmpnn/model/mpnn.py`, in `sample_autoregressive_state_vmap_exact` (line 960), add parameter:

```python
def sample_autoregressive_state_vmap_exact(
    self,
    ...
    wave_position_valid_local: jax.Array,
    ar_logit_transform_fn: "ARLogitTransformFn | None" = None,
) -> tuple[OneHotProteinSequence, Logits]:
```

Pass it to `run_sample_autoregressive_state_vmap_exact(... ar_logit_transform_fn=ar_logit_transform_fn)`.

Also update `sample_autoregressive_state_vmap_exact_from_payload` (line 1005) and the `sample_autoregressive_from_payload` alias (line 1046) to accept and forward `ar_logit_transform_fn=None`.

- [ ] **Step 5: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive_logit_transform.py::test_ar_logit_transform_fn_changes_output -v
```

Expected: PASS

- [ ] **Step 6: Run fast regression suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/ tests/sampling/ tests/model/ -q --tb=short
```

Expected: All previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py \
        src/prxteinmpnn/model/mpnn.py \
        tests/pipeline/test_autoregressive_logit_transform.py
git commit -m "feat(sprint-A): thread ARLogitTransformFn through AR wave scan (MPNN)"
```

---

## Task 4: Same threading for LigandMPNN

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py`
- Modify: `src/prxteinmpnn/model/ligand_mpnn.py`
- Test: `tests/pipeline/test_autoregressive_logit_transform.py`

- [ ] **Step 1: Write failing test for LigandMPNN**

Add to `tests/pipeline/test_autoregressive_logit_transform.py`:

```python
def test_ligand_ar_logit_transform_fn_accepted():
    """LigandMPNN explicit-param method must have ar_logit_transform_fn in signature."""
    import inspect
    from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
    m = eqx.tree_inference(
        PrxteinLigandMPNN(16, 16, 16, 1, 1, 4, key=jax.random.PRNGKey(0)),
        value=True,
    )
    # Inspect the explicit-param wrapper, not sample_autoregressive_from_payload which is *args/**kwargs
    sig = inspect.signature(m.sample_autoregressive_state_vmap_exact_from_payload)
    assert "ar_logit_transform_fn" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive_logit_transform.py::test_ligand_ar_logit_transform_fn_accepted -v
```

Expected: FAIL

- [ ] **Step 3: Mirror changes to LigandMPNN files**

Apply the same pattern to:
- `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py` lines 239–247 (identical fusion site)
- `src/prxteinmpnn/model/ligand_mpnn.py` — `sample_autoregressive_state_vmap_exact`, `sample_autoregressive_state_vmap_exact_from_payload`, `sample_autoregressive_from_payload`

Pattern is identical to Task 3 — add `ar_logit_transform_fn: ARLogitTransformFn | None = None` parameter and thread through.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive_logit_transform.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py \
        src/prxteinmpnn/model/ligand_mpnn.py \
        tests/pipeline/test_autoregressive_logit_transform.py
git commit -m "feat(sprint-A): thread ARLogitTransformFn through AR wave scan (LigandMPNN)"
```

---

## Task 5: Wire into `AutoregressivePipeline`

**Files:**
- Modify: `src/prxteinmpnn/pipeline/autoregressive.py`
- Test: `tests/pipeline/test_autoregressive_logit_transform.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pipeline/test_autoregressive_logit_transform.py`:

```python
def test_autoregressive_pipeline_resolves_ar_logit_transform():
    """AutoregressivePipeline must resolve and pass ar_logit_transform_fn — output must change."""
    from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline, AutoregressiveInputs
    from prxteinmpnn.pipeline_fns import PipelineFns

    from prxteinmpnn.payloads import WaveParallelPayload
    m = _make_model()
    S, L, V = 2, 4, 21
    stack = _make_stack(S=S, L=L)
    w_id, w_loc, w_gv, w_pv = _build_wave_tables(S, L, k_neighbors=4)
    wave = WaveParallelPayload(
        wave_group_ids=w_id,
        wave_group_positions=w_loc,
        wave_group_valid=w_gv,
        wave_position_valid=w_pv,
    )
    inputs = AutoregressiveInputs(
        stack=stack,
        wave=wave,
        autoregressive_mask_stack=jnp.zeros((S, L, L)),
        bias_stack=jnp.zeros((S, L, V)),
    )

    def always_token_zero(state_logits, state_index, state_weights):
        return jnp.zeros(V).at[0].set(1e9)

    fns_default = PipelineFns.default()
    fns_forced = PipelineFns.from_callables(ar_logit_transform=always_token_zero)
    pipeline = AutoregressivePipeline(temperature=1.0)
    key = jax.random.PRNGKey(0)

    seqs_default, _ = pipeline(m, key, inputs, fns=fns_default)
    seqs_forced, _ = pipeline(m, key, inputs, fns=fns_forced)

    assert jnp.all(seqs_forced == 0), "Forced transform must produce all-zero sequences"
    assert not jnp.all(seqs_default == seqs_forced)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive_logit_transform.py::test_autoregressive_pipeline_resolves_ar_logit_transform -v
```

Expected: FAIL

- [ ] **Step 3: Update `AutoregressivePipeline.__call__`**

In `src/prxteinmpnn/pipeline/autoregressive.py`, resolve and pass the transform:

```python
def __call__(self, module, key, inputs, *, fns):
    S = inputs.stack.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S
    ar_logit_transform_fn = fns.resolve_ar_logit_transform()  # <-- add this

    sequences, logits = module.sample_autoregressive_from_payload(
        key,
        inputs.stack,
        inputs.autoregressive_mask_stack,
        inputs.bias_stack,
        self.temperature,
        self.multi_state_strategy_idx,
        state_weights,
        inputs.wave.wave_group_ids,
        inputs.wave.wave_group_positions,
        inputs.wave.wave_group_valid,
        inputs.wave.wave_position_valid,
        ar_logit_transform_fn=ar_logit_transform_fn,  # <-- add this
    )
    return sequences, logits
```

- [ ] **Step 4: Run all autoregressive logit transform tests**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive_logit_transform.py -v
```

Expected: All pass.

- [ ] **Step 5: Run full pipeline + sampling suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/ tests/sampling/ -q --tb=short
```

Expected: All previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/pipeline/autoregressive.py tests/pipeline/test_autoregressive_logit_transform.py
git commit -m "feat(sprint-A): AutoregressivePipeline resolves and passes ARLogitTransformFn"
```
