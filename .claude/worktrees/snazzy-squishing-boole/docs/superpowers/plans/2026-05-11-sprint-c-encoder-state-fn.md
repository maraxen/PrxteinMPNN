# Sprint C: EncoderStateFn — Carry-Based Scan Over Encoder States

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flat `jax.vmap(encode_one)` in both unconditional and conditional multistate scoring with a `jax.lax.scan` that threads an arbitrary JAX-pytree carry through each state, enabling cross-state accumulation (e.g., running statistics, key/value caches). Introduce `EncoderStateFn` protocol as the single hook for this scan body (replaces the unimplemented `EncoderPreFn`/`EncoderPostFn` pair). When `encoder_state_fn=None`, the existing vmap path is used unchanged.

**Architecture:** `EncoderStateFn` owns the full per-state pipeline: it receives the carry, a scalar state index, and single-state `BackboneGeometry`, internally calls the encoder, and returns `(new_carry, EncoderOutput)`. The scan is `jax.lax.scan(scan_body, init_carry, stacked_inputs)` where `stacked_inputs` is `(coords_stack, mask_stack, ri_stack, ci_stack, state_indices)`. Carry is a JAX pytree with fixed structure/shapes at trace time (required for `lax.scan`). When `encoder_state_fn` is present in `PipelineFns`, `Pipeline` implementations resolve and pass it; the model method uses scan instead of vmap.

**Tech Stack:** JAX (lax.scan, vmap), equinox, jaxtyping, pytest

---

## File Structure

**Modified:**
- `src/prxteinmpnn/protocols.py` — add `EncoderStateFn` protocol; deprecate `EncoderPreFn`/`EncoderPostFn` with TODO comment
- `src/prxteinmpnn/pipeline_fns.py` — add `encoder_state_fn_uid: str | None = None`; add `resolve_encoder_state_fn()` method; add `register_encoder_state_fn` typed alias; deprecate `encoder_pre/post_process_uid` fields (keep for compat, mark with TODO)
- `src/prxteinmpnn/pipeline_registry.py` — add `register_encoder_state_fn` typed alias (mirrors existing `register_encoder_pre_fn` / `register_encoder_post_fn`)
- `src/prxteinmpnn/model/mpnn.py` — replace `jax.vmap(encode_one)` with conditional scan path in `score_unconditional_state_vmap_exact` and `score_conditional_state_vmap_exact`; add `encoder_state_fn` parameter
- `src/prxteinmpnn/pipeline/unconditional.py` — resolve and pass `encoder_state_fn` from fns
- `src/prxteinmpnn/pipeline/conditional.py` — same
- `tests/test_pipeline_fns.py` — add `EncoderStateFn` importable test

**Created:**
- `tests/pipeline/test_encoder_state_fn.py` — behavioral tests for carry-based scan

---

## Task 1: Add `EncoderStateFn` protocol to `protocols.py`

**Files:**
- Modify: `src/prxteinmpnn/protocols.py`
- Test: `tests/test_pipeline_fns.py`

Background: `EncoderStateFn` replaces the `EncoderPreFn` + `EncoderPostFn` pair with a single scan-body callable. It must include `init_carry()` so scan has a valid initial pytree. The carry type is arbitrary but must have fixed structure and shapes at JAX trace time (JAX-traced pytree, not Python-level objects).

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline_fns.py`:

```python
def test_encoder_state_fn_importable():
    from prxteinmpnn.protocols import EncoderStateFn
    assert EncoderStateFn is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_encoder_state_fn_importable -v
```

Expected: FAIL — `ImportError`

- [ ] **Step 3: Add `EncoderStateFn` to `protocols.py`**

After the `EncoderPostFn` definition (around line 259), add:

```python
class EncoderStateFn(Protocol):
  """Carry-based scan body over encoder states.

  Replaces jax.vmap(encode_one) with jax.lax.scan, enabling cross-state
  accumulation via an arbitrary JAX-pytree carry.

  Carry must have fixed structure and shapes at JAX trace time (required for
  jax.lax.scan). The carry is initialized once via init_carry() and threaded
  through all S states sequentially.

  # TODO: extend backbone with group_indices, hierarchy, additional features
  # for more composability (e.g., ligand features, membrane labels, hierarchy
  # embeddings). Current (carry, state_idx, backbone) is the right level of
  # abstraction for single-chain multistate use cases.
  """

  def init_carry(self) -> Any:
    """Return the initial carry pytree (called once before the scan).

    Must return a valid JAX pytree with fixed structure/shapes at trace time.
    Use () or jnp.zeros(()) for stateless hooks.
    """
    ...

  def __call__(
    self,
    carry: Any,
    state_idx: Int[Array, ""],
    backbone: BackboneGeometry,
  ) -> tuple[Any, EncoderOutput]:
    """Process one state in the encoder scan.

    Args:
      carry: Current carry pytree (output of previous state, or init_carry()).
      state_idx: Scalar traced int32 — index of current state in [0, S).
      backbone: Single-state BackboneGeometry (unvmapped coords, mask, ri, ci).

    Returns:
      (new_carry, encoded): updated carry and encoded state output.
      encoded.node_features: (L, D)
      encoded.edge_features: (L, K, D)
      encoded.neighbor_indices: (L, K) int32
      encoded.mask: (L,)
    """
    ...
```

Add `EncoderStateFn` to `__all__` at the bottom of `protocols.py`.

Mark the old `EncoderPreFn` and `EncoderPostFn` with a deprecation note:

```python
class EncoderPreFn(Protocol):
  # DEPRECATED: Use EncoderStateFn instead. Will be removed in a future sprint.
  # EncoderStateFn unifies pre/post with a carry-based scan.
  ...
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_encoder_state_fn_importable -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/protocols.py tests/test_pipeline_fns.py
git commit -m "feat(sprint-C): add EncoderStateFn carry-scan protocol; deprecate EncoderPreFn/PostFn"
```

---

## Task 2: Add `encoder_state_fn_uid` to `PipelineFns`

**Files:**
- Modify: `src/prxteinmpnn/pipeline_fns.py`
- Test: `tests/test_pipeline_fns.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_pipeline_fns.py`:

```python
def test_pipeline_fns_has_encoder_state_fn_uid():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert hasattr(fns, "encoder_state_fn_uid")
    assert fns.encoder_state_fn_uid is None


def test_pipeline_fns_resolve_encoder_state_fn_returns_none_by_default():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert fns.resolve_encoder_state_fn() is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_pipeline_fns_has_encoder_state_fn_uid tests/test_pipeline_fns.py::test_pipeline_fns_resolve_encoder_state_fn_returns_none_by_default -v
```

Expected: FAIL

- [ ] **Step 3: Update `PipelineFns`**

In `src/prxteinmpnn/pipeline_fns.py`, add:

```python
@dataclasses.dataclass(frozen=True)
class PipelineFns:
  logit_transform_uid: str
  encoder_pre_process_uid: str | None = None   # DEPRECATED: use encoder_state_fn_uid
  encoder_post_process_uid: str | None = None  # DEPRECATED: use encoder_state_fn_uid
  encoder_state_fn_uid: str | None = None      # <-- add this

  def resolve_encoder_state_fn(self) -> EncoderStateFn | None:
    if self.encoder_state_fn_uid is None:
      return None
    return resolve_hook(self.encoder_state_fn_uid)
```

Note: Do NOT add `ar_logit_transform_uid` — that field does not exist in the codebase at this sprint's starting point and has no registry backing here. Add only `encoder_state_fn_uid`.

Update `from_callables` to accept `encoder_state_fn: EncoderStateFn | None = None`. When `encoder_state_fn is not None`, call `register_encoder_state_fn(encoder_state_fn)` (from `pipeline_registry`) and store the returned uid in `encoder_state_fn_uid`.

Add `EncoderStateFn` to the `TYPE_CHECKING` import block.

Also add the typed alias to `src/prxteinmpnn/pipeline_registry.py` (after the existing `register_encoder_post_fn` definition):

```python
def register_encoder_state_fn(fn: Any, *, name: str | None = None) -> str:
    """Typed alias for register_hook for EncoderStateFn callables."""
    return register_hook(fn, name=name)
```

Add `"register_encoder_state_fn"` to `pipeline_registry.__all__`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/pipeline_fns.py tests/test_pipeline_fns.py
git commit -m "feat(sprint-C): add encoder_state_fn_uid to PipelineFns"
```

---

## Task 3: Wire `encoder_state_fn` into `score_unconditional_state_vmap_exact` in `mpnn.py`

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Test: `tests/pipeline/test_encoder_state_fn.py`

Background: Currently, `score_unconditional_state_vmap_exact` (line 1061) calls `jax.vmap(encode_one)` over the S-state stack (lines 1114–1120). The scan replacement uses `jax.lax.scan(scan_body, init_carry, stacked_inputs)` where each iteration calls `encoder_state_fn(carry, state_idx, backbone_s)` → `(carry, EncoderOutput)`. The decoder vmap is unchanged. When `encoder_state_fn=None`, the existing vmap is used.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_encoder_state_fn.py`:

```python
"""Verify EncoderStateFn carry-based scan is called during unconditional scoring."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload, EncoderOutput
from prxteinmpnn.model_inputs import BackboneGeometry
from prxteinmpnn.protocols import EncoderStateFn


def _make_model():
    return eqx.tree_inference(
        PrxteinMPNN(16, 16, 16, 1, 1, 4, key=jax.random.PRNGKey(0)),
        value=True,
    )


def _make_stack(S=2, L=6):
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


def test_encoder_state_fn_carry_accumulates():
    """Carry must accumulate across S states at runtime (not just trace time).

    jax.lax.scan traces the body ONCE but executes it S times at runtime.
    We verify this by running the scan directly and checking the concrete carry value.
    """
    m = _make_model()
    stack = _make_stack(S=3, L=4)
    S = stack.n_states

    class CountingEncoder:
        """Carry is a scalar int32 count; increments once per state."""
        def __init__(self, model):
            self.m = model

        def init_carry(self):
            return jnp.zeros((), dtype=jnp.int32)

        def __call__(self, carry, state_idx, backbone):
            ef, nei, nf, _ = self.m.features(
                jax.random.PRNGKey(0),
                backbone.coords, backbone.mask,
                backbone.residue_index, backbone.chain_index,
                jnp.asarray(0.0, jnp.float32),
                structure_mapping=None, initial_node_features=None,
                rbf_features=None, neighbor_indices=None,
            )
            from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors
            nf2, ef2, nei2 = encoder_forward_with_int_neighbors(
                self.m.encoder, ef, nei, backbone.mask, nf,
                inference=True, key=None,
            )
            new_carry = carry + jnp.ones((), dtype=jnp.int32)
            return new_carry, EncoderOutput(nf2, ef2, nei2, backbone.mask)

    encoder_fn = CountingEncoder(m)

    # Run the scan directly on the stacked inputs to verify carry accumulation
    def scan_body(carry, per_state):
        coords_s, mask_s, ri_s, ci_s, idx_s = per_state
        backbone_s = BackboneGeometry(
            coords=coords_s, mask=mask_s,
            residue_index=ri_s, chain_index=ci_s,
        )
        return encoder_fn(carry, idx_s, backbone_s)

    init_carry = encoder_fn.init_carry()
    final_carry, enc_stacked = jax.lax.scan(
        scan_body,
        init_carry,
        (stack.coords_stack, stack.mask_stack,
         stack.residue_index_stack, stack.chain_index_stack,
         stack.state_index),
    )
    # final_carry = 0 + 1*S = S (accumulated once per state)
    assert int(final_carry) == S, f"Expected carry={S}, got {int(final_carry)}"
    # enc_stacked.node_features has shape (S, L, D) — scan ran over all S states
    assert enc_stacked.node_features.shape[0] == S


def test_encoder_state_fn_passthrough_matches_vmap():
    """PassthroughEncoder (scan path) must produce logits identical to vmap path."""
    m = _make_model()
    stack = _make_stack(S=2, L=4)
    key = jax.random.PRNGKey(42)

    class PassthroughEncoder:
        """Replicates encode_one exactly; carry is unused (stateless scan)."""
        def __init__(self, model):
            self.m = model

        def init_carry(self):
            return ()

        def __call__(self, carry, state_idx, backbone):
            ef, nei, nf, _ = self.m.features(
                jax.random.PRNGKey(0),
                backbone.coords, backbone.mask,
                backbone.residue_index, backbone.chain_index,
                jnp.asarray(0.0, jnp.float32),
                structure_mapping=None, initial_node_features=None,
                rbf_features=None, neighbor_indices=None,
            )
            from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors
            nf2, ef2, nei2 = encoder_forward_with_int_neighbors(
                self.m.encoder, ef, nei, backbone.mask, nf,
                inference=True, key=None,
            )
            return (), EncoderOutput(nf2, ef2, nei2, backbone.mask)

    logits_vmap = m.score_unconditional_from_payload(
        key, stack,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=None,  # vmap path
    )
    logits_scan = m.score_unconditional_from_payload(
        key, stack,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=PassthroughEncoder(m),  # scan path, identical encode
    )
    assert jnp.allclose(logits_vmap, logits_scan, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_encoder_state_fn.py -v
```

Expected: FAIL — `TypeError: unexpected keyword argument 'encoder_state_fn'`

- [ ] **Step 3: Add `encoder_state_fn` parameter and scan path to `score_unconditional_state_vmap_exact`**

In `src/prxteinmpnn/model/mpnn.py`, in `score_unconditional_state_vmap_exact` (line 1061):

1. Add parameter to signature:
```python
def score_unconditional_state_vmap_exact(
    self,
    ...
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
) -> Logits:
```

2. Replace the `jax.vmap(encode_one)` block with conditional dispatch:

```python
if encoder_state_fn is not None:
    # Carry-based scan over S states.
    # Each call: encoder_state_fn(carry, state_idx, backbone) → (carry, EncoderOutput)
    # Carry is a JAX pytree with fixed structure/shapes at trace time.
    init_carry = encoder_state_fn.init_carry()
    state_indices = jnp.arange(coords_stack.shape[0], dtype=jnp.int32)

    def scan_body(carry, per_state):
        coords_s, mask_s, ri_s, ci_s, idx_s = per_state
        backbone_s = BackboneGeometry(
            coords=coords_s, mask=mask_s,
            residue_index=ri_s, chain_index=ci_s,
        )
        new_carry, enc_out = encoder_state_fn(carry, idx_s, backbone_s)
        return new_carry, enc_out

    _, enc_stacked = jax.lax.scan(
        scan_body,
        init_carry,
        (coords_stack, mask_stack, residue_index_stack, chain_index_stack, state_indices),
    )
    node_b = enc_stacked.node_features
    edge_b = enc_stacked.edge_features
    nei_b = enc_stacked.neighbor_indices
else:
    # Default: vmap path (existing behavior, unchanged)
    def encode_one(coords, ma, ri, ci):
        ef, nei, nf, _ = self.features(...)
        return encoder_forward_with_int_neighbors(...)

    node_b, edge_b, nei_b = jax.vmap(encode_one)(
        coords_stack, mask_stack, residue_index_stack, chain_index_stack,
    )
```

Note: keep the existing `encode_one` body exactly as-is inside the else branch. The decode vmap (`decode_one` and `w_out`) is unchanged after either path.

3. Update `score_unconditional_state_vmap_exact_from_payload` and `score_unconditional_from_payload` to accept and forward `encoder_state_fn=None`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_encoder_state_fn.py::test_encoder_state_fn_carry_accumulates tests/pipeline/test_encoder_state_fn.py::test_encoder_state_fn_passthrough_matches_vmap -v
```

Expected: PASS

- [ ] **Step 5: Run fast suite to confirm no regressions**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/ tests/model/ tests/sampling/ -q --tb=short
```

Expected: All previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/pipeline/test_encoder_state_fn.py
git commit -m "feat(sprint-C): carry-based scan path in score_unconditional_state_vmap_exact"
```

---

## Task 4: Wire `encoder_state_fn` into `score_conditional_state_vmap_exact`

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Test: `tests/pipeline/test_encoder_state_fn.py`

Background: `score_conditional_state_vmap_exact` (line 1203) has the identical `jax.vmap(encode_one)` structure before the conditional decoder vmap. Same dispatch pattern applies.

- [ ] **Step 1: Write failing test**

Add to `tests/pipeline/test_encoder_state_fn.py`:

```python
def test_encoder_state_fn_in_conditional_path():
    """PassthroughEncoder (scan) must match vmap path for score_conditional_from_payload."""
    m = _make_model()
    stack = _make_stack(S=2, L=4)
    S, L, V = stack.n_states, stack.n_canonical, 21
    seq_oh = jnp.zeros((S, L, V))
    ar_mask = jnp.eye(L, dtype=jnp.float32)[None].repeat(S, axis=0)
    key = jax.random.PRNGKey(7)

    class PassthroughEncoder:
        def __init__(self, model):
            self.m = model

        def init_carry(self):
            return ()

        def __call__(self, carry, state_idx, backbone):
            ef, nei, nf, _ = self.m.features(
                jax.random.PRNGKey(0),
                backbone.coords, backbone.mask,
                backbone.residue_index, backbone.chain_index,
                jnp.asarray(0.0, jnp.float32),
                structure_mapping=None, initial_node_features=None,
                rbf_features=None, neighbor_indices=None,
            )
            from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors
            nf2, ef2, nei2 = encoder_forward_with_int_neighbors(
                self.m.encoder, ef, nei, backbone.mask, nf, inference=True, key=None,
            )
            return (), EncoderOutput(nf2, ef2, nei2, backbone.mask)

    logits_vmap = m.score_conditional_from_payload(
        key, stack,
        seq_oh_stack=seq_oh, ar_mask_stack=ar_mask,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=None,  # vmap path
    )
    logits_scan = m.score_conditional_from_payload(
        key, stack,
        seq_oh_stack=seq_oh, ar_mask_stack=ar_mask,
        tie_group_map=None, multi_state_strategy_idx=0,
        state_weights=None, state_mapping=None, inference=True,
        encoder_state_fn=PassthroughEncoder(m),  # scan path
    )
    assert logits_vmap.shape == (stack.n_canonical, V)
    assert jnp.allclose(logits_vmap, logits_scan, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_encoder_state_fn.py::test_encoder_state_fn_in_conditional_path -v
```

Expected: FAIL — `TypeError`

- [ ] **Step 3: Apply same scan dispatch to `score_conditional_state_vmap_exact`**

Mirror the Task 3 changes to `score_conditional_state_vmap_exact` in `mpnn.py` (line 1203). The encode block is identical; the conditional decode block that follows is unchanged. Update `score_conditional_state_vmap_exact_from_payload` and `score_conditional_from_payload` to forward `encoder_state_fn=None`.

- [ ] **Step 4: Run all encoder state fn tests**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_encoder_state_fn.py -v
```

Expected: All pass.

- [ ] **Step 5: Run full test suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ --ignore=tests/parity --ignore=tests/training -q --tb=short
```

Expected: Same pass/xfail counts as before this sprint.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/pipeline/test_encoder_state_fn.py
git commit -m "feat(sprint-C): carry-based scan path in score_conditional_state_vmap_exact"
```

---

## Task 5: Wire `encoder_state_fn` through `UnconditionalPipeline` and `ConditionalPipeline`

**Files:**
- Modify: `src/prxteinmpnn/pipeline/unconditional.py`
- Modify: `src/prxteinmpnn/pipeline/conditional.py`
- Test: `tests/pipeline/test_encoder_state_fn.py`

- [ ] **Step 1: Write failing test**

Add to `tests/pipeline/test_encoder_state_fn.py`:

```python
def test_unconditional_pipeline_resolves_encoder_state_fn():
    """UnconditionalPipeline must resolve encoder_state_fn from fns and pass it through."""
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    from prxteinmpnn.pipeline_fns import PipelineFns

    class IdentityEncoder:
        def __init__(self, model):
            self.m = model

        def init_carry(self):
            return ()

        def __call__(self, carry, state_idx, backbone):
            ef, nei, nf, _ = self.m.features(
                jax.random.PRNGKey(0),
                backbone.coords, backbone.mask,
                backbone.residue_index, backbone.chain_index,
                jnp.asarray(0.0, jnp.float32),
                structure_mapping=None, initial_node_features=None,
                rbf_features=None, neighbor_indices=None,
            )
            from prxteinmpnn.model.encoder import encoder_forward_with_int_neighbors
            nf2, ef2, nei2 = encoder_forward_with_int_neighbors(
                self.m.encoder, ef, nei, backbone.mask, nf, inference=True, key=None,
            )
            return (), EncoderOutput(nf2, ef2, nei2, backbone.mask)

    fns = PipelineFns.from_callables(encoder_state_fn=IdentityEncoder(_make_model()))
    pipeline = UnconditionalPipeline()
    m = _make_model()
    stack = _make_stack()
    logits, state_logits = pipeline(m, jax.random.PRNGKey(0), stack, fns=fns)
    assert logits.shape[-1] == 21
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_encoder_state_fn.py::test_unconditional_pipeline_resolves_encoder_state_fn -v
```

Expected: FAIL

- [ ] **Step 3: Update `UnconditionalPipeline.__call__`**

In `src/prxteinmpnn/pipeline/unconditional.py`, resolve and pass:

```python
def __call__(self, module, key, inputs, *, fns):
    logit_transform_fn = fns.resolve_logit_transform()
    encoder_state_fn = fns.resolve_encoder_state_fn()  # <-- add

    # ... capturing_transform wrapper unchanged ...

    logits = module.score_unconditional_from_payload(
        key,
        inputs,
        tie_group_map=None,
        multi_state_strategy_idx=self.multi_state_strategy_idx,
        state_weights=state_weights,
        state_mapping=None,
        inference=self.inference,
        logit_transform_fn=capturing_transform,
        encoder_state_fn=encoder_state_fn,  # <-- add
    )
    ...
```

Apply same pattern to `ConditionalPipeline.__call__` in `conditional.py`.

- [ ] **Step 4: Run all encoder state fn tests**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_encoder_state_fn.py tests/pipeline/test_unconditional.py tests/pipeline/test_conditional.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/pipeline/unconditional.py src/prxteinmpnn/pipeline/conditional.py tests/pipeline/test_encoder_state_fn.py
git commit -m "feat(sprint-C): UnconditionalPipeline and ConditionalPipeline resolve EncoderStateFn"
```
