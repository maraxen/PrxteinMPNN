# Sprint B: HLO Export Smoke Tests — Conditional, Autoregressive, Payload Paths

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compile-time HLO export smoke tests for the conditional, autoregressive, and unconditional-payload decode paths, verifying they can be lowered by JAX and stay within the byte limits defined in `hlo_allowlist.toml`. Currently only the unconditional `__call__` path is covered.

**Architecture:** Each test wraps a model method call in a lambda that accepts only traced JAX arrays (PRNG key, coordinates, etc.), calls `export_hlo()` from `hlo_tools.py`, and asserts the byte count is under the allowlist ceiling. No numerical correctness is checked — these are compile-time smoke tests only. Fixtures are extracted to `tests/profiling/conftest.py` for reuse.

**Tech Stack:** JAX, equinox, pytest, jaxtyping

---

## File Structure

**Created:**
- `tests/profiling/conftest.py` — shared `tiny_model` and `make_stack` fixtures

**Modified:**
- `tests/profiling/test_hlo_baseline.py` — three new export tests, import fixtures from conftest
- `tests/profiling/hlo_allowlist.toml` — add entries for new paths (conditional, sample, score_unconditional_payload)

---

## Task 1: Extract shared fixtures to `tests/profiling/conftest.py`

**Files:**
- Create: `tests/profiling/conftest.py`
- Modify: `tests/profiling/test_hlo_baseline.py`

Background: `tiny_model` is currently defined as a local pytest fixture in `test_hlo_baseline.py`. `_make_model()` and `_make_stack()` are defined in `tests/pipeline/test_unconditional.py` and `tests/pipeline/test_conditional.py` but not shared. We extract them to a shared conftest.

- [ ] **Step 1: Create `tests/profiling/conftest.py`**

```python
"""Shared fixtures for profiling/HLO tests."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload


@pytest.fixture
def tiny_model() -> PrxteinMPNN:
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=4,
        key=key,
    )
    return eqx.tree_inference(m, value=True)


@pytest.fixture
def tiny_stack() -> MultistateStackPayload:
    S, L = 2, 6
    return MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.stack([jnp.arange(L, dtype=jnp.int32) + i * L for i in range(S)]),
        flat_row_offsets=jnp.array([0, L, S * L], dtype=jnp.int32),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=S * L,
    )
```

- [ ] **Step 2: Verify existing `tiny_model` fixture in `test_hlo_baseline.py` still works**

The existing local fixture in `test_hlo_baseline.py` should be removed and replaced with the one from conftest (conftest fixtures take precedence when named the same).

Remove the local `tiny_model` fixture definition from `test_hlo_baseline.py` and confirm conftest provides it:

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py --collect-only -q
```

Verify the test IDs are collected (no import errors), then run:

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py -v
```

Expected: All existing tests still pass (fixture resolved from conftest).

- [ ] **Step 3: Commit**

```bash
git add tests/profiling/conftest.py tests/profiling/test_hlo_baseline.py
git commit -m "refactor(sprint-B): extract tiny_model/tiny_stack fixtures to profiling conftest"
```

---

## Task 2: Add HLO export test for unconditional payload path

**Files:**
- Modify: `tests/profiling/test_hlo_baseline.py`
- Modify: `tests/profiling/hlo_allowlist.toml`

Background: The existing test calls `tiny_model(coords, mask, ri, ci, "unconditional", ...)` via `__call__`. The payload path `score_unconditional_from_payload(key, stack, ...)` has a different JIT graph (no mode dispatch, direct vmap encode). Both should be covered.

- [ ] **Step 1: Write the failing test**

Add to `tests/profiling/test_hlo_baseline.py`:

```python
def test_export_hlo_score_unconditional_payload(
    tiny_model: PrxteinMPNN,
    tiny_stack: MultistateStackPayload,
) -> None:
    # Compile-time smoke test: score_unconditional_from_payload must lower to HLO
    # and stay within byte budget. Not a numerical correctness check.
    def f(pk: jax.Array) -> jax.Array:
        return tiny_model.score_unconditional_from_payload(
            pk,
            tiny_stack,
            tie_group_map=None,
            multi_state_strategy_idx=0,
            state_weights=None,
            state_mapping=None,
            inference=True,
        )

    hlo = export_hlo(f, jax.random.PRNGKey(0))
    max_b = int(_allowlist()["score_unconditional_payload"]["max_hlo_bytes"])
    assert len(hlo.encode("utf-8")) <= max_b
```

Add to `tests/profiling/hlo_allowlist.toml`:

```toml
[score_unconditional_payload]
max_hlo_bytes = 12_000_000
rationale = "Unconditional payload path (score_unconditional_from_payload); similar graph to model_call."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py::test_export_hlo_score_unconditional_payload -v
```

Expected: FAIL — KeyError `score_unconditional_payload` in allowlist (or import error if imports not updated).

- [ ] **Step 3: Verify test passes after allowlist update**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py::test_export_hlo_score_unconditional_payload -v
```

Expected: PASS (HLO export succeeds, byte count under 12MB for tiny model).

- [ ] **Step 4: Commit**

```bash
git add tests/profiling/test_hlo_baseline.py tests/profiling/hlo_allowlist.toml
git commit -m "test(sprint-B): HLO smoke test for score_unconditional_from_payload"
```

---

## Task 3: Add HLO export test for conditional (teacher-forced) path

**Files:**
- Modify: `tests/profiling/test_hlo_baseline.py`
- Modify: `tests/profiling/hlo_allowlist.toml`

Background: `score_conditional_from_payload` requires `seq_oh_stack: (S, L, 21)` and `ar_mask_stack: (S, L, L)` in addition to the standard stack. These are concrete JAX arrays — they can be dummy zeros for a compile-time test.

- [ ] **Step 1: Write the failing test**

Add to `tests/profiling/test_hlo_baseline.py`:

```python
def test_export_hlo_score_conditional_payload(
    tiny_model: PrxteinMPNN,
    tiny_stack: MultistateStackPayload,
) -> None:
    S, L, V = tiny_stack.n_states, tiny_stack.n_canonical, 21
    seq_oh = jnp.zeros((S, L, V))
    ar_mask = jnp.eye(L, dtype=jnp.float32)[None].repeat(S, axis=0)

    def f(pk: jax.Array) -> jax.Array:
        return tiny_model.score_conditional_from_payload(
            pk,
            tiny_stack,
            seq_oh_stack=seq_oh,
            ar_mask_stack=ar_mask,
            tie_group_map=None,
            multi_state_strategy_idx=0,
            state_weights=None,
            state_mapping=None,
            inference=True,
        )

    hlo = export_hlo(f, jax.random.PRNGKey(0))
    max_b = int(_allowlist()["score_conditional"]["max_hlo_bytes"])
    assert len(hlo.encode("utf-8")) <= max_b
```

Add to `hlo_allowlist.toml`:

```toml
[score_conditional]
max_hlo_bytes = 15_000_000
rationale = "Conditional payload path (score_conditional_from_payload); larger than unconditional due to seq embedding."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py::test_export_hlo_score_conditional_payload -v
```

Expected: FAIL — KeyError or assertion.

- [ ] **Step 3: Verify test passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py::test_export_hlo_score_conditional_payload -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/profiling/test_hlo_baseline.py tests/profiling/hlo_allowlist.toml
git commit -m "test(sprint-B): HLO smoke test for score_conditional_from_payload"
```

---

## Task 4: Add HLO export test for autoregressive sampling path

**Files:**
- Modify: `tests/profiling/test_hlo_baseline.py`
- Modify: `tests/profiling/hlo_allowlist.toml`

Background: `sample_autoregressive_from_payload` requires wave group tables. These are built by `compute_wave_assignments()` (used in `tests/sampling/test_state_vmap_exact_jit.py`). For a compile-time smoke test, we use identity wave tables for a trivial single-group decode. The AR scan graph is larger so the allowlist ceiling is 20MB.

- [ ] **Step 1: Write the failing test**

`compute_wave_assignments` is in `prxteinmpnn.utils.wave_parallel` (not `state_vmap_prep`). Its full signature is:
```python
compute_wave_assignments(
    ca_coords: np.ndarray,         # (n_canonical, 4, 3) — first-state CA coords for kNN
    tie_group_flat: np.ndarray,    # (S*L,) — per-flat-position tie group IDs
    group_indices_table: np.ndarray,  # (n_groups, max_group_size) — flat indices per group
    group_valid_table: np.ndarray,    # (n_groups, max_group_size) — validity mask
    k_neighbors: int = 48,
    n_canonical: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

The returned positions are flat indices; use `remap_wave_positions_flat_to_local` to convert to local (per-state) indices before passing to the model.

Add to `tests/profiling/test_hlo_baseline.py`:

```python
def test_export_hlo_sample_autoregressive_payload(
    tiny_model: PrxteinMPNN,
    tiny_stack: MultistateStackPayload,
) -> None:
    import numpy as np
    from prxteinmpnn.utils.wave_parallel import compute_wave_assignments
    from prxteinmpnn.sampling.state_vmap_prep import remap_wave_positions_flat_to_local

    S, L = tiny_stack.n_states, tiny_stack.n_canonical
    # One canonical position per tie group (identity grouping, one member per state)
    tie_flat = np.tile(np.arange(L, dtype=np.int32), S)  # (S*L,)
    num_groups, max_group_size = L, S
    flat_offsets = np.array([s * L for s in range(S + 1)], dtype=np.int32)  # S+1 cumulative offsets for remap — NOT tiny_stack.flat_row_offsets (S elements)
    git = np.full((num_groups, max_group_size), -1, dtype=np.int32)
    gvt = np.zeros((num_groups, max_group_size), dtype=bool)
    for g in range(L):
        for s in range(S):
            git[g, s] = int(flat_offsets[s] + g)
            gvt[g, s] = True
    ca0 = np.zeros((L, 4, 3), dtype=np.float32)  # dummy CA coords (compile-time only)
    w_id, w_pos, w_gv, w_pv = compute_wave_assignments(ca0, tie_flat, git, gvt, k_neighbors=4, n_canonical=L)
    w_loc, w_pv2 = remap_wave_positions_flat_to_local(w_pos, w_pv, flat_offsets)
    ar_mask = jnp.zeros((S, L, L), dtype=jnp.float32)
    bias = jnp.zeros((S, L, 21), dtype=jnp.float32)

    def f(pk: jax.Array) -> jax.Array:
        seqs, _ = tiny_model.sample_autoregressive_from_payload(
            pk,
            tiny_stack,
            ar_mask,
            bias,
            1.0,
            0,
            None,
            jnp.asarray(w_id, dtype=jnp.int32),
            jnp.asarray(w_loc, dtype=jnp.int32),
            jnp.asarray(w_gv),
            jnp.asarray(w_pv2),
        )
        return seqs

    hlo = export_hlo(f, jax.random.PRNGKey(0))
    max_b = int(_allowlist()["sample_autoregressive"]["max_hlo_bytes"])
    assert len(hlo.encode("utf-8")) <= max_b
```

Add to `hlo_allowlist.toml`:

```toml
[sample_autoregressive]
max_hlo_bytes = 20_000_000
rationale = "AR wave-parallel scan (sample_autoregressive_from_payload); larger graph than scoring paths."
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py::test_export_hlo_sample_autoregressive_payload -v
```

Expected: FAIL — KeyError or import error.

- [ ] **Step 3: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/test_hlo_baseline.py::test_export_hlo_sample_autoregressive_payload -v
```

Expected: PASS.

- [ ] **Step 4: Run full profiling test suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/profiling/ -v --tb=short
```

Expected: All tests pass including the new three.

- [ ] **Step 5: Commit**

```bash
git add tests/profiling/test_hlo_baseline.py tests/profiling/hlo_allowlist.toml
git commit -m "test(sprint-B): HLO smoke tests for conditional and autoregressive payload paths"
```
