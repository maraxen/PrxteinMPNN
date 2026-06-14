---
title: G1 Training Parity Gate (backlog #1550)
task_id: 260615_potts-gates-runspec
date: 260615
gate: G1
status: complete
---

# G1: Training Parity Gate — Formal Verification

All three criteria pass. Training checkpoint, resumption, and gradient-based optimization verified.

## G1.1 — pytest suite

```
============================= test session starts ==============================
tests/training/test_checkpoint.py::test_get_checkpoint_manager_creates_directory PASSED [ 12%]
tests/training/test_checkpoint.py::test_save_and_load_roundtrip PASSED   [ 25%]
tests/training/test_checkpoint.py::test_save_uses_state_step PASSED      [ 37%]
tests/training/test_checkpoint.py::test_load_restores_model_weights PASSED [ 50%]
tests/training/test_resumable_state.py::test_resumable_state_construction PASSED [ 62%]
tests/training/test_resumable_state.py::test_resumable_state_step_update PASSED [ 75%]
tests/training/test_resumable_state.py::test_resumable_state_is_pytree PASSED [ 87%]
tests/training/test_resumable_state.py::test_resumable_state_jit_compatible PASSED [100%]

======================== 8 passed, 2 warnings in 1.55s ==========================
```

**Status:** PASS — All training tests pass (8/8).

## G1.2 — Checkpoint round-trip smoke

Round-trip checkpoint save/restore with ResumableState PyTree:

```python
import jax, jax.numpy as jnp, equinox as eqx, tempfile, pathlib
from xtrax.training.types import ResumableState
from aminx.training.checkpoint import get_checkpoint_manager, save_checkpoint, load_checkpoint

key = jax.random.PRNGKey(0)
model = eqx.nn.Linear(8, 8, key=key)
opt_state = None
state = ResumableState(step=jnp.int32(42), key=key, model=model, opt_state=opt_state, extras={})

with tempfile.TemporaryDirectory() as d:
    # Save
    mgr = get_checkpoint_manager(pathlib.Path(d), max_to_keep=None)
    save_checkpoint(mgr, state)
    mgr.close()
    
    # Load and verify
    mgr2 = get_checkpoint_manager(pathlib.Path(d), max_to_keep=None)
    restored = load_checkpoint(mgr2, state)
    mgr2.close()
    
    # Step matches
    assert int(restored.step) == 42
    
    # Model leaves match
    orig_leaves = jax.tree.leaves(eqx.filter(state, eqx.is_array))
    rest_leaves = jax.tree.leaves(eqx.filter(restored, eqx.is_array))
    for i, (o, r) in enumerate(zip(orig_leaves, rest_leaves)):
        assert jnp.allclose(o, r, atol=1e-7, rtol=0)
```

**Output:** `G1.2 CHECKPOINT ROUND-TRIP: PASS`

**Status:** PASS — Checkpoint round-trip preserves state and weights with no loss.

## G1.3 — 50-step overfit smoke

Minimal 50-step training loop with 4-residue dummy batch:

```python
# Dummy model: 2-layer Linear, 128→64→21 (amino acids)
model = DummySeqModel(key)
optimizer = optax.adam(learning_rate=0.01)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# Dummy batch: 4 residues, 128-d features
x_batch = jax.random.normal(key, (4, 128))
y_batch = jax.random.randint(key, (4,), 0, 21)
mask = jnp.ones(4)

# 50 training steps
for step in range(50):
    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model, x_batch, y_batch, mask)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
```

**Output:**
```
Step 0: loss=3.139056
Step 49: loss=0.000000
Loss decrease: 100.00%
G1.3 OVERFIT SMOKE: PASS
```

**Status:** PASS — Loss decreased 100% over 50 steps (exceeds 10% threshold).

---

## Summary

| Criterion | Result | Evidence |
|-----------|--------|----------|
| G1.1 pytest suite | PASS | 8 tests, 0 failures |
| G1.2 checkpoint round-trip | PASS | State + model weights preserved |
| G1.3 50-step overfit smoke | PASS | Loss: 3.14 → 0.00 (100% decrease) |

**GATE VERDICT: PASS**

Training parity verified. ResumableState PyTree checkpoint protocol operational. Gradient-based optimization functional. Ready for T1.5 integration testing.
