# Pipeline Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `Pipeline` Protocol abstraction layer over prxteinmpnn model calls, replacing ad-hoc `state_vmap_exact` naming and hardcoded `apply_multistate_to_all_logits` with composable `PipelineFns` hooks (`LogitTransformFn`, `EncoderPreFn`, `EncoderPostFn`) and four concrete pipeline types (`UnconditionalPipeline`, `ConditionalPipeline`, `AutoregressivePipeline`, `STEPipeline`).

**Architecture:** Each `Pipeline` implementation is a host-only frozen dataclass that resolves callable hooks from a `PipelineFns` registry, calls the appropriate model method with those hooks plumbed through, and returns typed output. `PipelineFns` stores UID strings (not callables) so it is safe to use as `static_argnames` in JIT. The concrete `*Pipeline` classes eliminate `lax.switch` (each is its own JIT boundary). The hook UID pattern mirrors the existing `decode_registry.py` and is extended to all three hook types.

**Tech Stack:** JAX/Equinox, jaxtyping, Python 3.11+, cloudpickle (for UID hashing), pytest

---

## File Structure

**Modified:**
- `src/prxteinmpnn/model_inputs.py` — rename `BatchLogitsFn` → `LogitTransformFn`, update `__all__`
- `src/prxteinmpnn/protocols.py` — add `Pipeline`, `ModelProtocol`, `EncoderPreFn`, `EncoderPostFn`
- `src/prxteinmpnn/payloads.py` — add `EncoderOutput` multi-state pytree
- `src/prxteinmpnn/run/decode_registry.py` — update docstrings / TYPE_CHECKING import to `LogitTransformFn`
- `src/prxteinmpnn/model/mpnn.py` — add `logit_transform_fn` param to `score_unconditional_state_vmap_exact` + `score_conditional_state_vmap_exact`
- `src/prxteinmpnn/model/ligand_mpnn.py` — same for LigandMPNN scoring methods

**Created:**
- `src/prxteinmpnn/pipeline_registry.py` — UID-based hook registry (shared by all three hook types)
- `src/prxteinmpnn/pipeline_fns.py` — `PipelineFns` frozen dataclass
- `src/prxteinmpnn/pipeline/__init__.py` — public exports
- `src/prxteinmpnn/pipeline/unconditional.py` — `UnconditionalPipeline`
- `src/prxteinmpnn/pipeline/conditional.py` — `ConditionalPipeline`
- `src/prxteinmpnn/pipeline/autoregressive.py` — `AutoregressivePipeline`
- `src/prxteinmpnn/pipeline/ste.py` — `STEPipeline`
- `tests/test_pipeline_fns.py` — `PipelineFns` unit tests
- `tests/pipeline/__init__.py`
- `tests/pipeline/test_unconditional.py`
- `tests/pipeline/test_conditional.py`
- `tests/pipeline/test_autoregressive.py`
- `tests/pipeline/test_ste.py`

---

## Task 1: Rename `BatchLogitsFn` → `LogitTransformFn`

**Files:**
- Modify: `src/prxteinmpnn/model_inputs.py`
- Modify: `src/prxteinmpnn/run/decode_registry.py`

Background: `BatchLogitsFn` is a Protocol for `(state_logits: S L V, state_index: S, state_weights: S) -> L V`. The name "Batch" is misleading — it's a per-step logit combination function, not a batching primitive. Renamed to `LogitTransformFn` to align with established naming convention. Current locations: `model_inputs.py:80` (definition) and `decode_registry.py:17` (TYPE_CHECKING import). No test files reference `BatchLogitsFn` directly yet.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline_fns.py` with a rename guard:

```python
"""Guards that BatchLogitsFn is gone and LogitTransformFn is importable."""
import pytest


def test_logit_transform_fn_importable():
    from prxteinmpnn.model_inputs import LogitTransformFn
    assert LogitTransformFn is not None


def test_batch_logits_fn_removed():
    import prxteinmpnn.model_inputs as mi
    assert not hasattr(mi, "BatchLogitsFn"), (
        "BatchLogitsFn must be removed; use LogitTransformFn"
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_logit_transform_fn_importable tests/test_pipeline_fns.py::test_batch_logits_fn_removed -v
```

Expected: FAIL — `ImportError: cannot import name 'LogitTransformFn'` and `test_batch_logits_fn_removed` PASSES (BatchLogitsFn still exists).

- [ ] **Step 3: Rename in `model_inputs.py`**

In `src/prxteinmpnn/model_inputs.py`, change:

```python
class BatchLogitsFn(Protocol):
  """JAX-traceable fn combining per-state logits into a single flat distribution.

  Passed as static_argnames to the outer JIT and inlined at jax.export time.
  Must use only jnp ops — no Python branching on traced values.

  state_weights is always a concrete array (uniform 1/S resolved on host if absent).
  """

  def __call__(
    self,
    logits_stack: Float[Array, "S L V"],
    state_index: Int[Array, "S"],
    state_weights: Float[Array, "S"],
  ) -> Float[Array, "L V"]: ...
```

to:

```python
class LogitTransformFn(Protocol):
  """JAX-traceable fn combining per-state logits into a single flat distribution.

  Passed as static_argnames to the outer JIT and inlined at jax.export time.
  Must use only jnp ops — no Python branching on traced values.

  state_weights is always a concrete array (uniform 1/S resolved on host if absent).
  state_index is an Int[Array, "S"] identifying each state (future: may carry richer metadata).
  """

  def __call__(
    self,
    state_logits: Float[Array, "S L V"],
    state_index: Int[Array, "S"],
    state_weights: Float[Array, "S"],
  ) -> Float[Array, "L V"]: ...
```

Also update `__all__` at line 99:

```python
__all__ = [
  "BackboneGeometry",
  "LogitTransformFn",
  "ConditioningFeatures",
  "SamplingInputs",
  "SamplingStaticConfig",
  "ScoringInputs",
  "ScoringStaticConfig",
]
```

Also update the module docstring line 5:
```python
LogitTransformFn (Protocol) defines the JAX-traceable post-processing contract.
```

- [ ] **Step 4: Update `decode_registry.py`**

In `src/prxteinmpnn/run/decode_registry.py`, change:
- Line 1 docstring: `"""Decode function registry for tracking LogitTransformFn provenance.`
- Line 17: `from prxteinmpnn.model_inputs import LogitTransformFn`
- Line 22 `DecodeFnEntry` docstring: `"""Registry entry for a LogitTransformFn with provenance metadata."""`
- Line 35 `register_decode_fn` docstring: `"""Register a LogitTransformFn and return its UID.`
- Line 56 `resolve_decode_fn` docstring: `"""Return the registered LogitTransformFn for a given UID."""`

- [ ] **Step 5: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_logit_transform_fn_importable tests/test_pipeline_fns.py::test_batch_logits_fn_removed -v
```

Expected: PASS — both tests green.

- [ ] **Step 6: Run full fast suite to check no regressions**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all tests green (BatchLogitsFn had no external test consumers).

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/model_inputs.py src/prxteinmpnn/run/decode_registry.py tests/test_pipeline_fns.py
git commit -m "refactor: rename BatchLogitsFn → LogitTransformFn; rename logits_stack → state_logits in signature"
```

---

## Task 2: Add `EncoderOutput` multi-state pytree

**Files:**
- Modify: `src/prxteinmpnn/payloads.py`
- Test: `tests/test_pipeline_fns.py`

Background: `EncoderPostFn` needs a structured input type for the encoder's output over S states. Currently `EncodedFeatures` in `payloads.py` is single-state. We add `EncoderOutput` as the S-state batched version. Shape: `node_features: (S, L, D)`, `edge_features: (S, L, K, E)`, `neighbor_indices: (S, L, K)`, `mask: (S, L)`. This is what `jax.vmap(encode_one)` produces in `score_unconditional_state_vmap_exact`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_fns.py`:

```python
def test_encoder_output_importable():
    from prxteinmpnn.payloads import EncoderOutput
    assert EncoderOutput is not None


def test_encoder_output_is_pytree():
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.payloads import EncoderOutput

    S, L, K, D, E = 2, 6, 16, 32, 32
    enc = EncoderOutput(
        node_features=jnp.zeros((S, L, D)),
        edge_features=jnp.zeros((S, L, K, E)),
        neighbor_indices=jnp.zeros((S, L, K), dtype=jnp.int32),
        mask=jnp.ones((S, L)),
    )
    leaves, treedef = jax.tree_util.tree_flatten(enc)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.node_features.shape == (S, L, D)
    assert restored.mask.shape == (S, L)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_encoder_output_importable tests/test_pipeline_fns.py::test_encoder_output_is_pytree -v
```

Expected: FAIL — `ImportError: cannot import name 'EncoderOutput'`

- [ ] **Step 3: Add `EncoderOutput` to `payloads.py`**

Add after the `EncodedFeatures` class (after line 172) in `src/prxteinmpnn/payloads.py`:

```python
class EncoderOutput(eqx.Module):
  """Multi-state encoder output (S states) for EncoderPostFn hook injection.

  Produced by jax.vmap(encode_one) over a state stack; passed to EncoderPostFn
  before decoding. Use this type in EncoderPostFn signatures (not EncodedFeatures).
  """

  node_features: Float[Array, ...]
  edge_features: Float[Array, ...]
  neighbor_indices: Int[Array, ...]
  mask: Float[Array, ...]

  def replace(self, **kw: Any) -> EncoderOutput:
    fields = ("node_features", "edge_features", "neighbor_indices", "mask")
    return _replace_payload(self, EncoderOutput, fields, frozenset(), **kw)
```

Also add `"EncoderOutput"` to `__all__` in `payloads.py`.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_encoder_output_importable tests/test_pipeline_fns.py::test_encoder_output_is_pytree -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/payloads.py tests/test_pipeline_fns.py
git commit -m "feat: add EncoderOutput multi-state pytree for EncoderPostFn hook signature"
```

---

## Task 3: Add `EncoderPreFn`, `EncoderPostFn`, `Pipeline`, `ModelProtocol` to protocols

**Files:**
- Modify: `src/prxteinmpnn/protocols.py`
- Test: `tests/test_pipeline_fns.py`

Background: Four new Protocol definitions. `EncoderPreFn` and `EncoderPostFn` are hook Protocols (not yet wired inside the model encoder — that requires a separate wiring task). `Pipeline` is a generic Protocol over `(module, key, inputs, *, fns) -> OutputT`. `ModelProtocol` declares the sub-modules accessed by Pipeline implementations so that they can be type-checked structurally.

**`EncoderPreFn`** signature: `(backbone: BackboneGeometry, state_index: Int[Array, "S"]) -> dict[str, Any] | None`. Returns a dict of precomputed features that supplements `self.features(...)` inputs (e.g. `{"initial_node_features": ..., "rbf_features": ...}`). Returns `None` means "no precomputation; use default feature extraction."

**`EncoderPostFn`** signature: `(encoded: EncoderOutput, state_index: Int[Array, "S"]) -> EncoderOutput`. The returned `EncoderOutput` replaces the encoder output before decoding. Used for cosine similarity re-weighting across states.

**`Pipeline`** signature: `(module: ModelProtocol, key: PRNGKeyArray, inputs: Any, *, fns: PipelineFns) -> Any`. Uses `Any` for inputs/outputs since each concrete pipeline defines its own typed version.

**`ModelProtocol`** declares: `features`, `encoder`, `decoder`, `w_out`, `w_s_embed`, `capabilities`. Structural — any module with these attributes satisfies it.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline_fns.py`:

```python
def test_encoder_pre_fn_protocol():
    from prxteinmpnn.protocols import EncoderPreFn
    assert EncoderPreFn is not None


def test_encoder_post_fn_protocol():
    from prxteinmpnn.protocols import EncoderPostFn
    assert EncoderPostFn is not None


def test_pipeline_protocol():
    from prxteinmpnn.protocols import Pipeline
    assert Pipeline is not None


def test_model_protocol():
    from prxteinmpnn.protocols import ModelProtocol
    assert ModelProtocol is not None


def test_model_protocol_runtime_checkable_vs_prxtein_mpnn():
    """PrxteinMPNN satisfies ModelProtocol structurally."""
    import jax
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.protocols import ModelProtocol

    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)
    assert isinstance(m, ModelProtocol)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_encoder_pre_fn_protocol tests/test_pipeline_fns.py::test_encoder_post_fn_protocol tests/test_pipeline_fns.py::test_pipeline_protocol tests/test_pipeline_fns.py::test_model_protocol tests/test_pipeline_fns.py::test_model_protocol_runtime_checkable_vs_prxtein_mpnn -v
```

Expected: FAIL — `ImportError` for all four names.

- [ ] **Step 3: Add Protocols to `protocols.py`**

At the top of `src/prxteinmpnn/protocols.py`, add to imports:
```python
from typing import Any
```
(already present via `TYPE_CHECKING` — move `Any` to the always-imported block if needed)

Add to the `TYPE_CHECKING` block:
```python
  from prxteinmpnn.payloads import EncoderOutput
  from prxteinmpnn.pipeline_fns import PipelineFns
```

Then add these four Protocol classes at the bottom of `src/prxteinmpnn/protocols.py` (before the final `__all__` if it exists):

```python
@runtime_checkable
class EncoderPreFn(Protocol):
  """Hook called before the encoder on each state batch.

  Returns a dict of supplemental feature arrays keyed by feature name
  (e.g. ``"initial_node_features"``, ``"rbf_features"``), or ``None``
  to use default feature extraction. Must use only jnp ops.

  NOT yet wired inside the model encoder — see pipeline wiring task.
  """

  def __call__(
    self,
    backbone: BackboneGeometry,
    state_index: Int[Array, "S"],
  ) -> dict[str, Any] | None: ...


@runtime_checkable
class EncoderPostFn(Protocol):
  """Hook called after jax.vmap(encode_one) on the full state batch.

  Receives the stacked encoder output (S states) and may return a modified
  EncoderOutput (e.g. cosine-similarity re-weighting across states). Must
  use only jnp ops — no Python branching on traced values.

  NOT yet wired inside the model encoder — see pipeline wiring task.
  """

  def __call__(
    self,
    encoded: EncoderOutput,
    state_index: Int[Array, "S"],
  ) -> EncoderOutput: ...


@runtime_checkable
class ModelProtocol(Protocol):
  """Structural protocol over prxteinmpnn model modules.

  Declares the sub-modules and static fields accessed by Pipeline implementations.
  ``PrxteinMPNN`` and ``PrxteinLigandMPNN`` must satisfy this structurally.
  """

  features: Any
  encoder: Any
  decoder: Any
  w_out: Any
  w_s_embed: Any
  capabilities: Any


@runtime_checkable
class Pipeline(Protocol):
  """Callable protocol for model pipeline implementations.

  Each concrete Pipeline (Unconditional, Conditional, Autoregressive, STE) implements
  this by calling the appropriate model method with PipelineFns hooks resolved.
  Inputs and outputs are typed by each concrete subclass.
  """

  def __call__(
    self,
    module: ModelProtocol,
    key: PRNGKeyArray,
    inputs: Any,
    *,
    fns: PipelineFns,
  ) -> Any: ...
```

Also add `BackboneGeometry` to the `TYPE_CHECKING` import block:
```python
  from prxteinmpnn.model_inputs import BackboneGeometry
```

- [ ] **Step 4: Check that `PrxteinMPNN` has `w_s_embed`**

```bash
grep -n "w_s_embed" /home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/model/mpnn.py | head -5
```

If `w_s_embed` is not an attribute name used by `PrxteinMPNN`, find the actual attribute name:

```bash
grep -n "s_embed\|state_embed\|w_s" /home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/model/mpnn.py | head -10
```

Update `ModelProtocol` to use the actual attribute name (e.g. if it's `W_s`, use `W_s`).

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_encoder_pre_fn_protocol tests/test_pipeline_fns.py::test_encoder_post_fn_protocol tests/test_pipeline_fns.py::test_pipeline_protocol tests/test_pipeline_fns.py::test_model_protocol tests/test_pipeline_fns.py::test_model_protocol_runtime_checkable_vs_prxtein_mpnn -v
```

Expected: PASS. If `test_model_protocol_runtime_checkable_vs_prxtein_mpnn` fails due to a missing attribute, update `ModelProtocol` to match what `PrxteinMPNN` actually declares.

- [ ] **Step 6: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/protocols.py tests/test_pipeline_fns.py
git commit -m "feat: add EncoderPreFn, EncoderPostFn, ModelProtocol, Pipeline protocols"
```

---

## Task 4: Create `pipeline_registry.py` (UID-based hook registry)

**Files:**
- Create: `src/prxteinmpnn/pipeline_registry.py`
- Test: `tests/test_pipeline_fns.py`

Background: `decode_registry.py` already implements cloudpickle-hash UID registration for `LogitTransformFn`. We need the same for `EncoderPreFn` and `EncoderPostFn`. Rather than extending `decode_registry.py` (which is scoped to logit transforms), we create `pipeline_registry.py` as a unified hook registry for all three hook types. The `decode_registry.py` stays for backward compatibility (its `DEFAULT_DECODE_FN_UID` is used in `SamplingStaticConfig`).

`pipeline_registry.py` implements:
- `register_hook(fn, *, name=None) -> str` — cloudpickle-hash UID, idempotent
- `resolve_hook(uid: str) -> Any` — looks up the registered callable
- `DEFAULT_LOGIT_TRANSFORM_UID: str` — same arithmetic mean default, registered here
- `register_encoder_pre_fn(fn, *, name=None) -> str` — typed alias
- `register_encoder_post_fn(fn, *, name=None) -> str` — typed alias
- `register_logit_transform_fn(fn, *, name=None) -> str` — typed alias

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline_fns.py`:

```python
def test_pipeline_registry_register_resolve_roundtrip():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_registry import register_hook, resolve_hook

    def my_transform(state_logits, state_index, state_weights):
        return jnp.mean(state_logits, axis=0)

    uid = register_hook(my_transform, name="test_mean")
    resolved = resolve_hook(uid)
    assert resolved is my_transform
    assert len(uid) == 16


def test_pipeline_registry_idempotent():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_registry import register_hook

    def fn(state_logits, state_index, state_weights):
        return jnp.sum(state_logits, axis=0)

    uid1 = register_hook(fn, name="idem_test")
    uid2 = register_hook(fn, name="idem_test")
    assert uid1 == uid2


def test_default_logit_transform_uid_exists():
    from prxteinmpnn.pipeline_registry import DEFAULT_LOGIT_TRANSFORM_UID, resolve_hook
    assert isinstance(DEFAULT_LOGIT_TRANSFORM_UID, str)
    assert len(DEFAULT_LOGIT_TRANSFORM_UID) == 16
    fn = resolve_hook(DEFAULT_LOGIT_TRANSFORM_UID)
    assert callable(fn)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_pipeline_registry_register_resolve_roundtrip tests/test_pipeline_fns.py::test_pipeline_registry_idempotent tests/test_pipeline_fns.py::test_default_logit_transform_uid_exists -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'prxteinmpnn.pipeline_registry'`

- [ ] **Step 3: Create `src/prxteinmpnn/pipeline_registry.py`**

```python
"""Unified UID-based hook registry for PipelineFns callables.

Mirrors decode_registry.py but covers all three hook types:
LogitTransformFn, EncoderPreFn, EncoderPostFn.

Host-only: never imported from JAX-traced code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from typing import Any


@dataclasses.dataclass
class HookEntry:
  """Registry entry for a pipeline hook with provenance metadata."""

  uid: str
  name: str
  fn: Any
  cloudpickle_bytes: bytes
  env_trace: dict[str, str]


_REGISTRY: dict[str, HookEntry] = {}


def register_hook(fn: Any, *, name: str | None = None) -> str:
  """Register a pipeline hook callable and return its UID.

  UID is a 16-char hex prefix of SHA-256(cloudpickle(fn)).
  Idempotent: re-registering the same fn returns the same UID.
  Works for LogitTransformFn, EncoderPreFn, EncoderPostFn.
  """
  import cloudpickle  # noqa: PLC0415

  pkl = cloudpickle.dumps(fn)
  uid = hashlib.sha256(pkl).hexdigest()[:16]
  if uid not in _REGISTRY:
    _REGISTRY[uid] = HookEntry(
      uid=uid,
      name=name or getattr(fn, "__name__", repr(fn)),
      fn=fn,
      cloudpickle_bytes=pkl,
      env_trace=_capture_env(),
    )
  return uid


def resolve_hook(uid: str) -> Any:
  """Return the registered callable for a given UID."""
  if uid not in _REGISTRY:
    msg = f"No hook registered for uid={uid!r}. Call register_hook first."
    raise KeyError(msg)
  return _REGISTRY[uid].fn


def register_logit_transform_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for LogitTransformFn callables."""
  return register_hook(fn, name=name)


def register_encoder_pre_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for EncoderPreFn callables."""
  return register_hook(fn, name=name)


def register_encoder_post_fn(fn: Any, *, name: str | None = None) -> str:
  """Typed alias for register_hook for EncoderPostFn callables."""
  return register_hook(fn, name=name)


def _capture_env() -> dict[str, str]:
  import jax  # noqa: PLC0415

  return {"python": sys.version, "jax": jax.__version__}


def _default_arithmetic_mean(
  state_logits: Any,
  state_index: Any,
  state_weights: Any,
) -> Any:
  """Default LogitTransformFn: uniform arithmetic mean across states."""
  import jax.numpy as jnp  # noqa: PLC0415

  return jnp.mean(state_logits, axis=0)


DEFAULT_LOGIT_TRANSFORM_UID: str = register_hook(
  _default_arithmetic_mean,
  name="arithmetic_mean_default",
)


__all__ = [
  "DEFAULT_LOGIT_TRANSFORM_UID",
  "HookEntry",
  "register_encoder_post_fn",
  "register_encoder_pre_fn",
  "register_hook",
  "register_logit_transform_fn",
  "resolve_hook",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_pipeline_registry_register_resolve_roundtrip tests/test_pipeline_fns.py::test_pipeline_registry_idempotent tests/test_pipeline_fns.py::test_default_logit_transform_uid_exists -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/pipeline_registry.py tests/test_pipeline_fns.py
git commit -m "feat: add pipeline_registry — unified UID-based hook registry for PipelineFns"
```

---

## Task 5: Create `PipelineFns` frozen dataclass

**Files:**
- Create: `src/prxteinmpnn/pipeline_fns.py`
- Test: `tests/test_pipeline_fns.py`

Background: `PipelineFns` is a host-only frozen dataclass (NOT `eqx.Module`) that stores UID strings for all three hook types. UID strings are static at JIT trace time. `PipelineFns.default()` returns a `PipelineFns` with the arithmetic mean `LogitTransformFn` and no encoder hooks. `PipelineFns.from_callables(...)` registers callables and returns UIDs. A `TrainingFns` stub is also included here (empty for now — loss fn wiring is future work) to establish the separation between sampling/scoring hooks and training hooks.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline_fns.py`:

```python
def test_pipeline_fns_default_constructible():
    from prxteinmpnn.pipeline_fns import PipelineFns
    fns = PipelineFns.default()
    assert isinstance(fns.logit_transform_uid, str)
    assert len(fns.logit_transform_uid) == 16
    assert fns.encoder_pre_process_uid is None
    assert fns.encoder_post_process_uid is None


def test_pipeline_fns_from_callables():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_fns import PipelineFns

    def my_transform(state_logits, state_index, state_weights):
        return jnp.mean(state_logits, axis=0)

    fns = PipelineFns.from_callables(logit_transform=my_transform)
    assert isinstance(fns.logit_transform_uid, str)
    assert len(fns.logit_transform_uid) == 16
    assert fns.encoder_pre_process_uid is None


def test_pipeline_fns_resolve_logit_transform():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_fns import PipelineFns

    fns = PipelineFns.default()
    fn = fns.resolve_logit_transform()
    assert callable(fn)
    import jax
    state_logits = jnp.ones((2, 4, 21))
    result = fn(state_logits, jnp.arange(2), jnp.ones(2))
    assert result.shape == (4, 21)


def test_pipeline_fns_is_frozen():
    from prxteinmpnn.pipeline_fns import PipelineFns
    import dataclasses
    fns = PipelineFns.default()
    assert dataclasses.is_dataclass(fns)
    with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
        fns.logit_transform_uid = "something_else"


def test_pipeline_fns_with_encoder_post():
    import jax.numpy as jnp
    from prxteinmpnn.payloads import EncoderOutput
    from prxteinmpnn.pipeline_fns import PipelineFns

    def my_post(encoded, state_index):
        return encoded

    fns = PipelineFns.from_callables(encoder_post_process=my_post)
    assert fns.encoder_post_process_uid is not None
    resolved = fns.resolve_encoder_post_process()
    assert resolved is my_post
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py -k "pipeline_fns" -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'prxteinmpnn.pipeline_fns'`

- [ ] **Step 3: Create `src/prxteinmpnn/pipeline_fns.py`**

```python
"""PipelineFns: host-only frozen dataclass for pipeline hook UIDs.

PipelineFns stores UID strings (not callables) for LogitTransformFn,
EncoderPreFn, and EncoderPostFn. UIDs are safe to pass as static_argnames
to JIT — the callable is resolved at dispatch time via pipeline_registry.

Usage:
    fns = PipelineFns.default()                          # arithmetic mean
    fns = PipelineFns.from_callables(logit_transform=fn) # custom transform
    logit_fn = fns.resolve_logit_transform()             # get callable
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from prxteinmpnn.pipeline_registry import (
  DEFAULT_LOGIT_TRANSFORM_UID,
  register_encoder_post_fn,
  register_encoder_pre_fn,
  register_logit_transform_fn,
  resolve_hook,
)

if TYPE_CHECKING:
  from prxteinmpnn.protocols import EncoderPostFn, EncoderPreFn, LogitTransformFn


@dataclasses.dataclass(frozen=True)
class PipelineFns:
  """Host-only container for pipeline hook UIDs.

  All three hook types are optional except logit_transform (defaults to
  arithmetic mean). Encoder hooks are None by default — set them when
  custom encoder pre/post processing is needed (e.g. cosine similarity
  across states for multistate design).
  """

  logit_transform_uid: str
  encoder_pre_process_uid: str | None = None
  encoder_post_process_uid: str | None = None

  @classmethod
  def default(cls) -> PipelineFns:
    """Default PipelineFns: arithmetic mean logit transform, no encoder hooks."""
    return cls(logit_transform_uid=DEFAULT_LOGIT_TRANSFORM_UID)

  @classmethod
  def from_callables(
    cls,
    *,
    logit_transform: Any | None = None,
    encoder_pre_process: Any | None = None,
    encoder_post_process: Any | None = None,
  ) -> PipelineFns:
    """Register callables and return a PipelineFns with their UIDs.

    Any hook not provided defaults to None (or arithmetic mean for logit_transform).
    Re-registering the same callable is idempotent.
    """
    if logit_transform is not None:
      lt_uid = register_logit_transform_fn(logit_transform)
    else:
      lt_uid = DEFAULT_LOGIT_TRANSFORM_UID

    pre_uid = (
      register_encoder_pre_fn(encoder_pre_process)
      if encoder_pre_process is not None
      else None
    )
    post_uid = (
      register_encoder_post_fn(encoder_post_process)
      if encoder_post_process is not None
      else None
    )
    return cls(
      logit_transform_uid=lt_uid,
      encoder_pre_process_uid=pre_uid,
      encoder_post_process_uid=post_uid,
    )

  def resolve_logit_transform(self) -> LogitTransformFn:
    """Return the registered LogitTransformFn callable."""
    return resolve_hook(self.logit_transform_uid)  # type: ignore[return-value]

  def resolve_encoder_pre_process(self) -> EncoderPreFn | None:
    """Return the registered EncoderPreFn callable, or None if not set."""
    if self.encoder_pre_process_uid is None:
      return None
    return resolve_hook(self.encoder_pre_process_uid)  # type: ignore[return-value]

  def resolve_encoder_post_process(self) -> EncoderPostFn | None:
    """Return the registered EncoderPostFn callable, or None if not set."""
    if self.encoder_post_process_uid is None:
      return None
    return resolve_hook(self.encoder_post_process_uid)  # type: ignore[return-value]


@dataclasses.dataclass(frozen=True)
class TrainingFns:
  """Host-only container for training-specific hook UIDs.

  Separate from PipelineFns to keep sampling/scoring hooks isolated from
  training hooks (loss fns, etc.). Currently empty — extend when training
  hook wiring is needed.
  """

  pass


__all__ = ["PipelineFns", "TrainingFns"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py -k "pipeline_fns" -v
```

Expected: PASS for all `test_pipeline_fns_*` tests.

- [ ] **Step 5: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/pipeline_fns.py tests/test_pipeline_fns.py
git commit -m "feat: add PipelineFns frozen dataclass with UID-based hook registry"
```

---

## Task 6: Wire `LogitTransformFn` into `score_unconditional_state_vmap_exact`

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Test: `tests/pipeline/test_unconditional.py`

Background: `score_unconditional_state_vmap_exact` currently hardcodes `apply_multistate_to_all_logits` when `tie_group_map is not None`. The `LogitTransformFn` hook should replace this. We add an optional `logit_transform_fn: LogitTransformFn | None = None` parameter. When `None`, it falls back to the existing `apply_multistate_to_all_logits` behavior for backward compatibility. When provided, the fn is called with `(state_logits, state_index, state_weights)` — but note the current `score_unconditional_state_vmap_exact` applies the transform AFTER scatter to flat (i.e. on `logits_flat`). The `LogitTransformFn` signature takes `state_logits: (S, L, V)` BEFORE scatter, which is the right place to apply multistate fusion. We'll thread `logit_transform_fn` to replace the flat post-scatter application.

**Key shape note:** In `score_unconditional_state_vmap_exact`, `logits_s` at line 1116 has shape `(S, L_pad, V)` — this is the pre-scatter per-state logits tensor. This is where `LogitTransformFn` is applied. The current `scatter_stack_to_flat + apply_multistate_to_all_logits` pattern maps `(S, L_pad, V)` → `(n_flat, V)`. `LogitTransformFn(logits_s, state_index, state_weights)` → `(L_canonical, V)`. These are different shapes. For backward compatibility, when `logit_transform_fn is None`, keep the existing scatter path. When `logit_transform_fn is not None`, call it and skip scatter (the pipeline is responsible for the scatter or the output shape is different).

**Note:** In the first wiring pass, keep the existing scatter path as the default and add `logit_transform_fn` as an ADDITIONAL hook called on `logits_s` before scatter. This avoids changing the output contract for existing callers. The `UnconditionalPipeline` in Task 7 will use this.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/__init__.py` (empty):
```python
```

Create `tests/pipeline/test_unconditional.py`:

```python
"""Tests for UnconditionalPipeline and logit_transform_fn wiring."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.pipeline_fns import PipelineFns
from prxteinmpnn.payloads import MultistateStackPayload


def _make_model(key=None):
    key = key or jax.random.PRNGKey(0)
    return PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)


def _make_stack(S: int = 2, L: int = 6, n_canonical: int = 6):
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
        n_canonical=n_canonical,
        n_flat=S * L,
    )


def test_score_unconditional_accepts_logit_transform_fn():
    """score_unconditional_state_vmap_exact_from_payload accepts logit_transform_fn kwarg."""
    key = jax.random.PRNGKey(42)
    m = _make_model()
    stack = _make_stack(S=2, L=6)

    call_count = []
    def counting_transform(state_logits, state_index, state_weights):
        call_count.append(1)
        return jnp.mean(state_logits, axis=0)

    logits = m.score_unconditional_state_vmap_exact_from_payload(
        key,
        stack,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        multi_state_temperature=1.0,
        state_weights=None,
        state_mapping=None,
        logit_transform_fn=counting_transform,
    )
    assert len(call_count) > 0, "logit_transform_fn must be called"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_unconditional.py::test_score_unconditional_accepts_logit_transform_fn -v
```

Expected: FAIL — `TypeError: score_unconditional_state_vmap_exact_from_payload() got an unexpected keyword argument 'logit_transform_fn'`

- [ ] **Step 3: Add `logit_transform_fn` param to `score_unconditional_state_vmap_exact_from_payload`**

In `src/prxteinmpnn/model/mpnn.py`, find `score_unconditional_state_vmap_exact_from_payload` (line ~1130). Change its signature to accept and forward `logit_transform_fn`:

```python
  def score_unconditional_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float | float,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    inference: bool = True,
    logit_transform_fn: Any | None = None,
  ) -> Logits:
    """Same as score_unconditional_state_vmap_exact with geometry from stack."""
    return self.score_unconditional_state_vmap_exact(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      multi_state_temperature=multi_state_temperature,
      state_weights=state_weights,
      state_mapping=state_mapping,
      inference=inference,
      logit_transform_fn=logit_transform_fn,
    )
```

Also update `score_unconditional_state_vmap_exact` signature to accept `logit_transform_fn: Any | None = None`:

```python
  def score_unconditional_state_vmap_exact(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    multi_state_temperature: Float | float,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    inference: bool = True,
    logit_transform_fn: Any | None = None,
  ) -> Logits:
```

And in the method body, after `logits_s = jax.vmap(jax.vmap(self.w_out))(decoded)` (line ~1116), add the hook call before the existing scatter logic:

```python
    # Hook: apply logit_transform_fn on (S, L, V) state logits if provided
    if logit_transform_fn is not None:
      _state_weights = (
        jnp.ones(logits_s.shape[0], dtype=jnp.float32) / logits_s.shape[0]
        if state_weights is None
        else state_weights
      )
      _state_index = jnp.arange(logits_s.shape[0], dtype=jnp.int32)
      logits_s = logit_transform_fn(logits_s, _state_index, _state_weights)

    logits_flat = scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)
```

Note: when `logit_transform_fn` is provided, `logits_s` becomes `(L, V)` after the transform. This changes the shape expected by `scatter_stack_to_flat`. For the initial implementation, only call `logit_transform_fn` and return early before `scatter_stack_to_flat` when `logit_transform_fn is not None`:

```python
    logits_s = jax.vmap(jax.vmap(self.w_out))(decoded)  # (S, L_pad, V)

    if logit_transform_fn is not None:
      _state_weights = (
        jnp.ones(logits_s.shape[0], dtype=jnp.float32) / logits_s.shape[0]
        if state_weights is None
        else state_weights
      )
      _state_index = jnp.arange(logits_s.shape[0], dtype=jnp.int32)
      return logit_transform_fn(logits_s, _state_index, _state_weights)

    logits_flat = scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)
    if tie_group_map is not None:
      logits_flat = apply_multistate_to_all_logits(...)
    return logits_flat
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_unconditional.py::test_score_unconditional_accepts_logit_transform_fn -v
```

Expected: PASS.

- [ ] **Step 5: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all green. The `logit_transform_fn=None` default preserves the existing behavior for all current callers.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/pipeline/__init__.py tests/pipeline/test_unconditional.py
git commit -m "feat: add logit_transform_fn hook to score_unconditional_state_vmap_exact"
```

---

## Task 7: Wire `LogitTransformFn` into `score_conditional_state_vmap_exact`

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Modify: `src/prxteinmpnn/model/ligand_mpnn.py`
- Test: `tests/pipeline/test_conditional.py`

Background: Same pattern as Task 6, but for `score_conditional_state_vmap_exact` and both protein and ligand variants. The conditional path includes `seq_oh_stack` (one-hot sequence) and `ar_mask_stack`. The logit transform hook is applied to per-state logits `(S, L, V)` before the existing scatter.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_conditional.py`:

```python
"""Tests for ConditionalPipeline and logit_transform_fn wiring on conditional path."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload


def _make_model():
    return PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))


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


def test_score_conditional_accepts_logit_transform_fn():
    S, L = 2, 6
    key = jax.random.PRNGKey(1)
    m = _make_model()
    stack = _make_stack(S=S, L=L)
    seq_oh = jnp.zeros((S, L, 21))
    ar_mask = jnp.eye(L)[None].repeat(S, axis=0)

    call_count = []
    def counting_transform(state_logits, state_index, state_weights):
        call_count.append(1)
        return jnp.mean(state_logits, axis=0)

    logits = m.score_conditional_state_vmap_exact_from_payload(
        key,
        stack,
        seq_oh_stack=seq_oh,
        ar_mask_stack=ar_mask,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        multi_state_temperature=1.0,
        state_weights=None,
        state_mapping=None,
        logit_transform_fn=counting_transform,
    )
    assert len(call_count) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_conditional.py::test_score_conditional_accepts_logit_transform_fn -v
```

Expected: FAIL — unexpected keyword argument.

- [ ] **Step 3: Add `logit_transform_fn` to `score_conditional_state_vmap_exact_from_payload` in `mpnn.py`**

Find `score_conditional_state_vmap_exact_from_payload` (line ~1250 in `mpnn.py`). Add `logit_transform_fn: Any | None = None` to both the `_from_payload` method and the underlying `score_conditional_state_vmap_exact` method. Apply the same pattern as Task 6: call `logit_transform_fn(logits_s, _state_index, _state_weights)` → return early before scatter, when `logit_transform_fn is not None`.

Also apply the same change to `ligand_mpnn.py:score_conditional_state_vmap_exact` and `ligand_mpnn.py:score_conditional_state_vmap_exact_from_payload`.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_conditional.py -v
```

Expected: PASS.

- [ ] **Step 5: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py src/prxteinmpnn/model/ligand_mpnn.py tests/pipeline/test_conditional.py
git commit -m "feat: add logit_transform_fn hook to score_conditional_state_vmap_exact (protein + ligand)"
```

---

## Task 8: Create `UnconditionalPipeline`

**Files:**
- Create: `src/prxteinmpnn/pipeline/__init__.py` (empty for now)
- Create: `src/prxteinmpnn/pipeline/unconditional.py`
- Test: `tests/pipeline/test_unconditional.py`

Background: `UnconditionalPipeline` wraps `score_unconditional_state_vmap_exact_from_payload`. It resolves `LogitTransformFn` from `PipelineFns` and passes it through. Inputs: `ScoringInputs`. Outputs: `(logits, state_logits)` where `logits` is the combined `(L, V)` result and `state_logits` is the raw `(S, L, V)` per-state output (needed for downstream uses like computing per-state perplexity).

Actually, looking at the existing `score_unconditional_state_vmap_exact_from_payload`, it returns `Logits` (flat). For `UnconditionalPipeline`, we want to preserve `state_logits` separately. This means the pipeline needs to capture the pre-transform state logits. We'll do this by wrapping the `logit_transform_fn` with a capture.

- [ ] **Step 1: Write the failing tests**

Add to `tests/pipeline/test_unconditional.py`:

```python
def test_unconditional_pipeline_importable():
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    assert UnconditionalPipeline is not None


def test_unconditional_pipeline_smoke():
    """UnconditionalPipeline runs and returns logits of correct shape."""
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.payloads import MultistateStackPayload

    S, L, V = 2, 6, 21
    key = jax.random.PRNGKey(7)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    stack = MultistateStackPayload(
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

    fns = PipelineFns.default()
    pipeline = UnconditionalPipeline()
    result = pipeline(m, key, stack, fns=fns)
    # result is (logits: (L, V), state_logits: (S, L, V))
    logits, state_logits = result
    assert logits.shape == (L, V)
    assert state_logits.shape == (S, L, V)


def test_unconditional_pipeline_matches_direct_call():
    """UnconditionalPipeline output matches direct score_unconditional call."""
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.payloads import MultistateStackPayload

    S, L = 2, 6
    key = jax.random.PRNGKey(11)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    stack = MultistateStackPayload(
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

    fns = PipelineFns.default()
    pipeline = UnconditionalPipeline()
    pipeline_logits, pipeline_state_logits = pipeline(m, key, stack, fns=fns)

    # Direct call with arithmetic mean
    direct_logits = m.score_unconditional_state_vmap_exact_from_payload(
        key,
        stack,
        tie_group_map=None,
        multi_state_strategy_idx=0,
        multi_state_temperature=1.0,
        state_weights=None,
        state_mapping=None,
        logit_transform_fn=lambda sl, si, sw: jnp.mean(sl, axis=0),
    )
    assert jnp.allclose(pipeline_logits, direct_logits, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_unconditional.py -k "pipeline" -v
```

Expected: FAIL — `ModuleNotFoundError` for `prxteinmpnn.pipeline.unconditional`.

- [ ] **Step 3: Create `src/prxteinmpnn/pipeline/__init__.py`**

```python
"""Pipeline Protocol implementations for prxteinmpnn.

Each Pipeline is a host-only frozen dataclass implementing the Pipeline protocol:
    pipeline(module, key, inputs, *, fns: PipelineFns) -> OutputT

Available pipelines:
    UnconditionalPipeline  — unconditional sequence scoring
    ConditionalPipeline    — conditional (teacher-forced) sequence scoring
    AutoregressivePipeline — temperature-sampled autoregressive sequence design
    STEPipeline            — straight-through estimator for differentiable design
"""
```

- [ ] **Step 4: Create `src/prxteinmpnn/pipeline/unconditional.py`**

```python
"""UnconditionalPipeline: unconditional sequence scoring over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

from prxteinmpnn.payloads import MultistateStackPayload


@dataclasses.dataclass(frozen=True)
class UnconditionalPipeline:
  """Wraps score_unconditional_state_vmap_exact_from_payload with PipelineFns hooks.

  Inputs:  MultistateStackPayload (stacked backbone geometry)
  Outputs: (logits: (L, V), state_logits: (S, L, V))
           where logits = logit_transform_fn(state_logits, state_index, state_weights)
           and state_logits is the raw per-state encoder/decoder output.

  multi_state_strategy_idx and state_weights are captured in the closure here;
  pass them at construction time rather than threading through the model method.
  """

  multi_state_strategy_idx: int = 0
  inference: bool = True

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: MultistateStackPayload,
    *,
    fns: Any,
  ) -> tuple[Any, Any]:
    """Run unconditional scoring and return (combined_logits, state_logits).

    Args:
      module: ModelProtocol (PrxteinMPNN or PrxteinLigandMPNN).
      key: JAX PRNGKey.
      inputs: MultistateStackPayload with state geometry.
      fns: PipelineFns with logit_transform_uid (and optional encoder hooks).

    Returns:
      (logits, state_logits) where logits is (L, V) and state_logits is (S, L, V).
    """
    logit_transform_fn = fns.resolve_logit_transform()

    captured_state_logits: list[Any] = []

    def capturing_transform(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
      captured_state_logits.append(state_logits)
      return logit_transform_fn(state_logits, state_index, state_weights)

    state_weights = (
      jnp.ones(inputs.n_states, dtype=jnp.float32) / inputs.n_states
    )
    logits = module.score_unconditional_state_vmap_exact_from_payload(
      key,
      inputs,
      tie_group_map=None,
      multi_state_strategy_idx=self.multi_state_strategy_idx,
      multi_state_temperature=1.0,
      state_weights=state_weights,
      state_mapping=None,
      inference=self.inference,
      logit_transform_fn=capturing_transform,
    )
    state_logits = captured_state_logits[0] if captured_state_logits else None
    return logits, state_logits
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_unconditional.py -k "pipeline" -v
```

Expected: PASS — all `test_unconditional_pipeline_*` tests green.

- [ ] **Step 6: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/pipeline/__init__.py src/prxteinmpnn/pipeline/unconditional.py tests/pipeline/test_unconditional.py
git commit -m "feat: add UnconditionalPipeline wrapping score_unconditional_state_vmap_exact"
```

---

## Task 9: Create `ConditionalPipeline`

**Files:**
- Create: `src/prxteinmpnn/pipeline/conditional.py`
- Test: `tests/pipeline/test_conditional.py`

Background: `ConditionalPipeline` wraps `score_conditional_state_vmap_exact_from_payload`. Requires `seq_oh_stack` (one-hot sequences, stacked per state) and `ar_mask_stack` in addition to backbone geometry. These are passed as part of inputs — we define `ConditionalInputs` as a thin struct.

- [ ] **Step 1: Write the failing tests**

Add to `tests/pipeline/test_conditional.py`:

```python
def test_conditional_pipeline_importable():
    from prxteinmpnn.pipeline.conditional import ConditionalPipeline, ConditionalInputs
    assert ConditionalPipeline is not None
    assert ConditionalInputs is not None


def test_conditional_pipeline_smoke():
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.pipeline.conditional import ConditionalPipeline, ConditionalInputs
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.payloads import MultistateStackPayload

    S, L, V = 2, 6, 21
    key = jax.random.PRNGKey(3)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))

    stack = MultistateStackPayload(
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
    inputs = ConditionalInputs(
        stack=stack,
        seq_oh_stack=jnp.zeros((S, L, V)),
        ar_mask_stack=jnp.eye(L)[None].repeat(S, axis=0),
    )

    fns = PipelineFns.default()
    pipeline = ConditionalPipeline()
    logits, state_logits = pipeline(m, key, inputs, fns=fns)
    assert logits.shape == (L, V)
    assert state_logits.shape == (S, L, V)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_conditional.py -k "pipeline_importable or conditional_pipeline_smoke" -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/prxteinmpnn/pipeline/conditional.py`**

```python
"""ConditionalPipeline: teacher-forced conditional scoring over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class ConditionalInputs(eqx.Module):
  """Inputs for ConditionalPipeline.

  stack: MultistateStackPayload with backbone geometry.
  seq_oh_stack: one-hot sequences per state, shape (S, L, 21).
  ar_mask_stack: autoregressive mask per state, shape (S, L, L).
  """

  stack: Any
  seq_oh_stack: Float[Array, ...]
  ar_mask_stack: Float[Array, ...]


@dataclasses.dataclass(frozen=True)
class ConditionalPipeline:
  """Wraps score_conditional_state_vmap_exact_from_payload with PipelineFns hooks.

  Inputs:  ConditionalInputs
  Outputs: (logits: (L, V), state_logits: (S, L, V))
  """

  multi_state_strategy_idx: int = 0
  inference: bool = True

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: ConditionalInputs,
    *,
    fns: Any,
  ) -> tuple[Any, Any]:
    """Run conditional scoring and return (combined_logits, state_logits)."""
    logit_transform_fn = fns.resolve_logit_transform()
    captured: list[Any] = []

    def capturing_transform(state_logits: Any, state_index: Any, state_weights: Any) -> Any:
      captured.append(state_logits)
      return logit_transform_fn(state_logits, state_index, state_weights)

    S = inputs.stack.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S

    logits = module.score_conditional_state_vmap_exact_from_payload(
      key,
      inputs.stack,
      seq_oh_stack=inputs.seq_oh_stack,
      ar_mask_stack=inputs.ar_mask_stack,
      tie_group_map=None,
      multi_state_strategy_idx=self.multi_state_strategy_idx,
      multi_state_temperature=1.0,
      state_weights=state_weights,
      state_mapping=None,
      inference=self.inference,
      logit_transform_fn=capturing_transform,
    )
    state_logits = captured[0] if captured else None
    return logits, state_logits
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_conditional.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/pipeline/conditional.py tests/pipeline/test_conditional.py
git commit -m "feat: add ConditionalPipeline + ConditionalInputs"
```

---

## Task 10: Create `AutoregressivePipeline`

**Files:**
- Create: `src/prxteinmpnn/pipeline/autoregressive.py`
- Test: `tests/pipeline/test_autoregressive.py`

Background: `AutoregressivePipeline` wraps `sample_autoregressive_state_vmap_exact_from_payload`. The `batch_fn` parameter already exists in `_sample_sequences_jitted` (from PR-3). The pipeline resolves `logit_transform_fn` from `PipelineFns` and passes it as `batch_fn`. The autoregressive method applies the transform per decode step inside the wave-parallel scan. The `AutoregressivePipeline` holds `temperature`, `multi_state_strategy_idx`, and provides a `__call__` that accepts `AutoregressiveInputs`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_autoregressive.py`:

```python
"""Tests for AutoregressivePipeline."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload


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


def test_autoregressive_pipeline_importable():
    from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline, AutoregressiveInputs
    assert AutoregressivePipeline is not None


def test_autoregressive_pipeline_smoke():
    """AutoregressivePipeline samples sequences without error."""
    from prxteinmpnn.pipeline.autoregressive import AutoregressivePipeline, AutoregressiveInputs
    from prxteinmpnn.pipeline_fns import PipelineFns

    S, L = 2, 6
    key = jax.random.PRNGKey(5)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    stack = _make_stack(S=S, L=L)

    # Wave parallel payload (trivial single-group ordering)
    wave = WaveParallelPayload(
        wave_group_ids=jnp.zeros((1, L), dtype=jnp.int32),
        wave_group_positions=jnp.arange(L, dtype=jnp.int32)[None],
        wave_group_valid=jnp.ones((1,), dtype=jnp.bool_),
        wave_position_valid=jnp.ones((1, L), dtype=jnp.bool_),
    )
    inputs = AutoregressiveInputs(
        stack=stack,
        wave=wave,
        autoregressive_mask_stack=jnp.zeros((S, L, L)),
        bias_stack=jnp.zeros((S, L, 21)),
    )

    fns = PipelineFns.default()
    pipeline = AutoregressivePipeline(temperature=0.1)
    sequences, logits = pipeline(m, key, inputs, fns=fns)
    assert sequences.shape[:-1] == (S, L)  # (S, L, 21) one-hot
    assert logits.shape == (S, L, 21)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/prxteinmpnn/pipeline/autoregressive.py`**

```python
"""AutoregressivePipeline: temperature-sampled AR sequence design over a state stack."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class AutoregressiveInputs(eqx.Module):
  """Inputs for AutoregressivePipeline.

  stack: MultistateStackPayload with backbone geometry.
  wave: WaveParallelPayload with wave-parallel decode schedule.
  autoregressive_mask_stack: (S, L, L) AR mask per state.
  bias_stack: (S, L, 21) logit bias per state.
  """

  stack: Any
  wave: Any
  autoregressive_mask_stack: Float[Array, ...]
  bias_stack: Float[Array, ...]


@dataclasses.dataclass(frozen=True)
class AutoregressivePipeline:
  """Wraps sample_autoregressive_state_vmap_exact_from_payload with PipelineFns hooks.

  Inputs:  AutoregressiveInputs
  Outputs: (sequences: (S, L, 21) one-hot, logits: (S, L, 21))

  The logit_transform_fn from PipelineFns is threaded into the model as batch_fn,
  which is applied per decode step inside the wave-parallel scan.
  """

  temperature: float = 1.0
  multi_state_strategy_idx: int = 0

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: AutoregressiveInputs,
    *,
    fns: Any,
  ) -> tuple[Any, Any]:
    """Sample sequences autoregressively and return (sequences, state_logits).

    Args:
      module: ModelProtocol — must have sample_autoregressive_state_vmap_exact_from_payload.
      key: JAX PRNGKey.
      inputs: AutoregressiveInputs with stack, wave, ar_mask, bias.
      fns: PipelineFns — logit_transform_uid is resolved to batch_fn.

    Returns:
      (sequences, logits) where sequences is (S, L, 21) one-hot and logits is (S, L, 21).
    """
    logit_transform_fn = fns.resolve_logit_transform()
    S = inputs.stack.n_states
    state_weights = jnp.ones(S, dtype=jnp.float32) / S

    sequences, logits = module.sample_autoregressive_state_vmap_exact_from_payload(
      key,
      inputs.stack,
      inputs.autoregressive_mask_stack,
      inputs.bias_stack,
      self.temperature,
      self.multi_state_strategy_idx,
      1.0,  # multi_state_temperature captured in logit_transform_fn closure
      state_weights,
      inputs.wave.wave_group_ids,
      inputs.wave.wave_group_positions,
      inputs.wave.wave_group_valid,
      inputs.wave.wave_position_valid,
    )
    return sequences, logits
```

Note: `batch_fn` wiring for AR pipeline requires threading through `sample_autoregressive_state_vmap_exact_from_payload`. If `batch_fn` is not yet accepted by the model method, the pipeline still works — it just uses the model's built-in multistate fusion. Add a `# TODO: thread logit_transform_fn as batch_fn once model method accepts it` comment.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_autoregressive.py -v
```

Expected: PASS for smoke test. If the model method signature differs from what's expected, adjust the positional arguments to match.

- [ ] **Step 5: Commit**

```bash
git add src/prxteinmpnn/pipeline/autoregressive.py tests/pipeline/test_autoregressive.py
git commit -m "feat: add AutoregressivePipeline + AutoregressiveInputs"
```

---

## Task 11: Create `STEPipeline`

**Files:**
- Create: `src/prxteinmpnn/pipeline/ste.py`
- Test: `tests/pipeline/test_ste.py`

Background: `STEPipeline` wraps the Straight-Through Estimator optimize path (`make_optimize_sequence_fn` in `sampling/ste_optimize.py`). The STE path does differentiable soft-token optimization then discretizes. The `LogitTransformFn` hook is applied to per-state logits before the STE temperature annealing step. Unlike the other pipelines, `STEPipeline` output includes the gradient-supporting soft sequence tensor.

- [ ] **Step 1: Write the failing test**

Create `tests/pipeline/test_ste.py`:

```python
"""Tests for STEPipeline."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN


def test_ste_pipeline_importable():
    from prxteinmpnn.pipeline.ste import STEPipeline
    assert STEPipeline is not None


def test_ste_pipeline_smoke():
    """STEPipeline constructs and calls make_optimize_sequence_fn without error."""
    from prxteinmpnn.pipeline.ste import STEPipeline, STEInputs
    from prxteinmpnn.pipeline_fns import PipelineFns

    L = 6
    key = jax.random.PRNGKey(9)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))

    inputs = STEInputs(
        coords=jnp.zeros((L, 4, 3)),
        mask=jnp.ones((L,)),
        residue_index=jnp.arange(L, dtype=jnp.int32),
        chain_index=jnp.zeros((L,), dtype=jnp.int32),
        initial_sequence=jnp.zeros((L,), dtype=jnp.int32),
        n_steps=3,
        learning_rate=0.1,
        temperature=1.0,
    )

    fns = PipelineFns.default()
    pipeline = STEPipeline()
    result = pipeline(m, key, inputs, fns=fns)
    assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_ste.py -v
```

Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Inspect `make_optimize_sequence_fn` signature**

```bash
grep -n "def make_optimize_sequence_fn\|def optimize_sequence" /home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/sampling/ste_optimize.py | head -10
grep -n "class\|def " /home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/sampling/ste_optimize.py | head -30
```

Use the actual signature to write `STEPipeline`.

- [ ] **Step 4: Create `src/prxteinmpnn/pipeline/ste.py`**

```python
"""STEPipeline: straight-through estimator differentiable sequence design."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class STEInputs(eqx.Module):
  """Inputs for STEPipeline.

  Single-state STE optimization (not multistate); wraps make_optimize_sequence_fn.
  """

  coords: Float[Array, ...]
  mask: Float[Array, ...]
  residue_index: Int[Array, ...]
  chain_index: Int[Array, ...]
  initial_sequence: Int[Array, ...]
  n_steps: int = eqx.field(static=True)
  learning_rate: float = eqx.field(static=True)
  temperature: float = eqx.field(static=True)


@dataclasses.dataclass(frozen=True)
class STEPipeline:
  """Wraps make_optimize_sequence_fn with PipelineFns hooks.

  Inputs:  STEInputs (single-state backbone + STE hyperparams)
  Outputs: whatever make_optimize_sequence_fn returns (sequence, logits tuple)

  Note: LogitTransformFn is not currently threaded into the STE inner loop.
  It will be wired in a follow-up PR when multistate STE is needed.
  """

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: STEInputs,
    *,
    fns: Any,
  ) -> Any:
    """Run STE optimization and return optimized (sequence, logits)."""
    from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn  # noqa: PLC0415

    optimize_fn = make_optimize_sequence_fn(
      module,
      n_steps=inputs.n_steps,
      learning_rate=inputs.learning_rate,
      temperature=inputs.temperature,
    )
    return optimize_fn(
      key,
      inputs.coords,
      inputs.mask,
      inputs.residue_index,
      inputs.chain_index,
      inputs.initial_sequence,
    )
```

Note: the exact `make_optimize_sequence_fn` call may differ from the actual API. After running step 3, adjust the arguments to match. If the function signature is different, update accordingly.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/pipeline/test_ste.py -v
```

If they fail due to API mismatch with `make_optimize_sequence_fn`, read the actual function signature:

```bash
grep -n "def make_optimize_sequence_fn" /home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/sampling/ste_optimize.py
head -60 /home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/sampling/ste_optimize.py
```

Then update `src/prxteinmpnn/pipeline/ste.py` to match the actual API.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/pipeline/ste.py tests/pipeline/test_ste.py
git commit -m "feat: add STEPipeline wrapping make_optimize_sequence_fn"
```

---

## Task 12: Wire exports from `pipeline/__init__.py` and top-level `__init__.py`

**Files:**
- Modify: `src/prxteinmpnn/pipeline/__init__.py`
- Modify: `src/prxteinmpnn/__init__.py` (add `pipeline` to public surface if needed)
- Test: `tests/pipeline/test_autoregressive.py`, `tests/pipeline/test_unconditional.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline_fns.py`:

```python
def test_pipeline_top_level_imports():
    """All four pipeline types importable from prxteinmpnn.pipeline."""
    from prxteinmpnn.pipeline import (
        AutoregressivePipeline,
        ConditionalPipeline,
        STEPipeline,
        UnconditionalPipeline,
    )
    assert all(x is not None for x in [
        AutoregressivePipeline, ConditionalPipeline, STEPipeline, UnconditionalPipeline
    ])


def test_pipeline_fns_top_level_import():
    """PipelineFns importable from prxteinmpnn.pipeline_fns."""
    from prxteinmpnn.pipeline_fns import PipelineFns
    assert PipelineFns is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_pipeline_top_level_imports -v
```

Expected: FAIL — `ImportError: cannot import name 'AutoregressivePipeline' from 'prxteinmpnn.pipeline'`

- [ ] **Step 3: Update `src/prxteinmpnn/pipeline/__init__.py`**

```python
"""Pipeline Protocol implementations for prxteinmpnn.

Each Pipeline is a host-only frozen dataclass implementing the Pipeline protocol:
    pipeline(module, key, inputs, *, fns: PipelineFns) -> OutputT

Available pipelines:
    UnconditionalPipeline  — unconditional sequence scoring
    ConditionalPipeline    — conditional (teacher-forced) sequence scoring
    AutoregressivePipeline — temperature-sampled autoregressive sequence design
    STEPipeline            — straight-through estimator for differentiable design
"""

from prxteinmpnn.pipeline.autoregressive import AutoregressiveInputs, AutoregressivePipeline
from prxteinmpnn.pipeline.conditional import ConditionalInputs, ConditionalPipeline
from prxteinmpnn.pipeline.ste import STEInputs, STEPipeline
from prxteinmpnn.pipeline.unconditional import UnconditionalPipeline

__all__ = [
  "AutoregressiveInputs",
  "AutoregressivePipeline",
  "ConditionalInputs",
  "ConditionalPipeline",
  "STEInputs",
  "STEPipeline",
  "UnconditionalPipeline",
]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_pipeline_top_level_imports tests/test_pipeline_fns.py::test_pipeline_fns_top_level_import -v
```

Expected: PASS.

- [ ] **Step 5: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/prxteinmpnn/pipeline/__init__.py tests/test_pipeline_fns.py
git commit -m "feat: wire pipeline exports — all four Pipeline types importable from prxteinmpnn.pipeline"
```

---

## Task 13: Deprecate `state_vmap_exact` naming with backward-compat shims

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Modify: `src/prxteinmpnn/model/ligand_mpnn.py`
- Test: `tests/test_pipeline_fns.py`

Background: The `state_vmap_exact` suffix on public scoring/sampling method names is confusing (it conflates the transport mechanism with the operation). Replace with cleaner names. Add `DeprecationWarning` shims that call the renamed method. The underlying implementation is unchanged.

Renamed methods:
- `score_unconditional_state_vmap_exact` → `score_unconditional`
- `score_unconditional_state_vmap_exact_from_payload` → `score_unconditional_from_payload`
- `score_conditional_state_vmap_exact` → `score_conditional`
- `score_conditional_state_vmap_exact_from_payload` → `score_conditional_from_payload`
- `sample_autoregressive_state_vmap_exact` → `sample_autoregressive`
- `sample_autoregressive_state_vmap_exact_from_payload` → `sample_autoregressive_from_payload`

The OLD names become shims:
```python
def score_unconditional_state_vmap_exact_from_payload(self, *args, **kwargs):
    import warnings
    warnings.warn(
        "score_unconditional_state_vmap_exact_from_payload is deprecated; "
        "use score_unconditional_from_payload",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.score_unconditional_from_payload(*args, **kwargs)
```

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_pipeline_fns.py`:

```python
def test_score_unconditional_from_payload_importable():
    """New clean method name exists on PrxteinMPNN."""
    import jax
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    assert hasattr(m, "score_unconditional_from_payload")


def test_deprecated_name_warns():
    """Old _state_vmap_exact name emits DeprecationWarning."""
    import warnings
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.payloads import MultistateStackPayload

    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    S, L = 1, 4
    stack = MultistateStackPayload(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.arange(L, dtype=jnp.int32)[None],
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        tie_group_map_stack=jnp.zeros((S, L), dtype=jnp.int32),
        fixed_mask_stack=jnp.zeros((S, L)),
        fixed_tokens_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=jnp.arange(L, dtype=jnp.int32)[None],
        flat_row_offsets=jnp.array([0], dtype=jnp.int32),
        state_index=jnp.arange(S, dtype=jnp.int32),
        state_embedding=jnp.zeros((S, 1)),
        n_states=S,
        n_canonical=L,
        n_flat=L,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.score_unconditional_state_vmap_exact_from_payload(
            jax.random.PRNGKey(0),
            stack,
            tie_group_map=None,
            multi_state_strategy_idx=0,
            multi_state_temperature=1.0,
            state_weights=None,
            state_mapping=None,
        )
    assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_score_unconditional_from_payload_importable tests/test_pipeline_fns.py::test_deprecated_name_warns -v
```

Expected: FAIL — `AttributeError: 'PrxteinMPNN' object has no attribute 'score_unconditional_from_payload'` and no warning.

- [ ] **Step 3: Rename methods in `mpnn.py` and add shims**

In `src/prxteinmpnn/model/mpnn.py`:

1. Rename `score_unconditional_state_vmap_exact` → `score_unconditional` (keep implementation identical)
2. Rename `score_unconditional_state_vmap_exact_from_payload` → `score_unconditional_from_payload`
3. Rename `score_conditional_state_vmap_exact` → `score_conditional`
4. Rename `score_conditional_state_vmap_exact_from_payload` → `score_conditional_from_payload`
5. Rename `sample_autoregressive_state_vmap_exact` → `sample_autoregressive`
6. Rename `sample_autoregressive_state_vmap_exact_from_payload` → `sample_autoregressive_from_payload`

Then add deprecation shims after each new method:

```python
  # Deprecated aliases — will be removed in a future version
  def score_unconditional_state_vmap_exact_from_payload(self, *args: Any, **kwargs: Any) -> Any:
    import warnings
    warnings.warn(
      "score_unconditional_state_vmap_exact_from_payload is deprecated; "
      "use score_unconditional_from_payload",
      DeprecationWarning,
      stacklevel=2,
    )
    return self.score_unconditional_from_payload(*args, **kwargs)

  def score_conditional_state_vmap_exact_from_payload(self, *args: Any, **kwargs: Any) -> Any:
    import warnings
    warnings.warn(
      "score_conditional_state_vmap_exact_from_payload is deprecated; "
      "use score_conditional_from_payload",
      DeprecationWarning,
      stacklevel=2,
    )
    return self.score_conditional_from_payload(*args, **kwargs)

  def sample_autoregressive_state_vmap_exact_from_payload(self, *args: Any, **kwargs: Any) -> Any:
    import warnings
    warnings.warn(
      "sample_autoregressive_state_vmap_exact_from_payload is deprecated; "
      "use sample_autoregressive_from_payload",
      DeprecationWarning,
      stacklevel=2,
    )
    return self.sample_autoregressive_from_payload(*args, **kwargs)
```

Apply the same renaming to `ligand_mpnn.py`.

**Important:** Also update the pipeline files created in Tasks 8–11 to use the new method names (`score_unconditional_from_payload`, etc.) instead of the deprecated ones.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/test_pipeline_fns.py::test_score_unconditional_from_payload_importable tests/test_pipeline_fns.py::test_deprecated_name_warns -v
```

Expected: PASS.

- [ ] **Step 5: Run full fast suite**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/ -q --ignore=tests/parity -x
```

Expected: PASS with DeprecationWarnings for any tests that use old names. Update those tests to use new names.

- [ ] **Step 6: Run sampling fast tests**

```bash
cd prxteinmpnn && PYTHONPATH=src uv run pytest tests/sampling/ tests/model/ tests/pipeline/ -v
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py src/prxteinmpnn/model/ligand_mpnn.py src/prxteinmpnn/pipeline/ tests/test_pipeline_fns.py
git commit -m "refactor: rename state_vmap_exact methods to clean names; add DeprecationWarning shims"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| Rename `BatchLogitsFn` → `LogitTransformFn` | Task 1 |
| `EncoderOutput` multi-state pytree | Task 2 |
| `EncoderPreFn`, `EncoderPostFn`, `Pipeline`, `ModelProtocol` protocols | Task 3 |
| UID-based hook registry | Task 4 |
| `PipelineFns` frozen dataclass | Task 5 |
| `TrainingFns` stub | Task 5 |
| Wire `LogitTransformFn` into unconditional scoring | Task 6 |
| Wire `LogitTransformFn` into conditional scoring | Task 7 |
| `UnconditionalPipeline` | Task 8 |
| `ConditionalPipeline` | Task 9 |
| `AutoregressivePipeline` | Task 10 |
| `STEPipeline` | Task 11 |
| Public exports from `prxteinmpnn.pipeline` | Task 12 |
| Deprecate `state_vmap_exact` naming | Task 13 |
| `state_logits` naming (not `logits_stack`) | Tasks 1, 6, 7 |

**Deferred (out of scope for this plan):**
- `EncoderPreFn` wiring inside encoder (protocol defined in Task 3, `PipelineFns` field added in Task 5 — actual wiring into `self.features(...)` is a follow-up requiring deeper model surgery)
- `EncoderPostFn` wiring (same — protocol + UID field ready, call site wiring is follow-up)
- `multi_state_temperature` removal from method signatures (deferred — it's a scan-path param; wait until multistate scan path is also Pipeline-ized)
- Generalized `StateIndex` type (currently `Int[Array, "S"]` — generalizing to a richer descriptor is a Phase 6 concern)
- Removing `*args/**kwargs` from `SamplerFn` Protocol (Phase 7 typing cleanup)

**Type consistency check:** All pipelines return `(logits, state_logits)` tuples. `STEPipeline` returns whatever `make_optimize_sequence_fn` returns — this is acceptable since STE has a different contract (differentiable soft tokens, not hard logits). `ConditionalInputs` and `AutoregressiveInputs` are `eqx.Module` pytrees; `STEInputs` is also `eqx.Module`. All pipeline classes are `@dataclasses.dataclass(frozen=True)` — NOT `eqx.Module`.
