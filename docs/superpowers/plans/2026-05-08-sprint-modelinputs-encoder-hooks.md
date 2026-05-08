# Sprint: MODELINPUTS PR-4 + EncoderHooks + multi_state_temperature Removal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement ROADMAP items D → C → A → B in that order — remove `multi_state_temperature` from all model method signatures (capturing it in `LogitTransformFn` closures instead), wire `EncoderPreFn` / `EncoderPostFn` hooks, push the `model.__call__` boundary to accept `SamplingInputs`, and smoke-test `jax.export`.

**Architecture:** D shrinks the model method surface before C adds the hook channel; A builds the `SamplingInputs` pytree that C's clean channel will use; B validates the resulting shape for static export. The order ensures each PR can be reviewed independently without leaving the codebase in a broken state.

**Tech Stack:** JAX, Equinox, jaxtyping, prxteinmpnn (src/prxteinmpnn/)

**Key files:**
- `src/prxteinmpnn/model/mpnn.py` — 1388 LoC, `__call__` at line 649
- `src/prxteinmpnn/model/ligand_mpnn.py` — 1385 LoC
- `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py`
- `src/prxteinmpnn/model/mpnn_autoregressive_scan.py`
- `src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py`
- `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py`
- `src/prxteinmpnn/model/multi_state_sampling.py` — `geometric_mean_logits` lines 91–128
- `src/prxteinmpnn/payloads.py` — `SamplingControls` at lines 123–141, `LigandStack` at lines 99–108
- `src/prxteinmpnn/protocols.py` — `EncoderPreFn` at line 224, `EncoderPostFn` at line 241
- `src/prxteinmpnn/pipeline_registry.py` — hook registry, `DEFAULT_LOGIT_TRANSFORM_UID`
- `src/prxteinmpnn/pipeline_fns.py` — `PipelineFns`
- `src/prxteinmpnn/model_inputs.py` — `SamplingInputs`, `BackboneGeometry`
- `src/prxteinmpnn/scoring/score.py` — passes `multi_state_temperature=` at lines 146, 166
- `src/prxteinmpnn/sampling/sample.py` — passes `multi_state_temperature` at lines 607, 622
- `tests/test_sprint_modelinputs.py` — new test file for this sprint

---

## Sprint D — Remove `multi_state_temperature` from model signatures

**Scope:** Remove the `multi_state_temperature` positional/keyword argument from all model method
signatures and the `lax.switch` operand list in `mpnn.py` and `ligand_mpnn.py`, migrate the four
downstream model-file consumers, add a `make_geometric_mean_transform` factory so callers can
express temperature via `LogitTransformFn`, migrate the two deferred callers (`scoring/score.py`,
`sampling/sample.py`), and then remove the field from `SamplingControls`.

**Execution order within Sprint D:** D.1 → D.2a → D.2b → D.3a → D.3b → D.4 → **D.5 (callers)** → **D.6 (payload field)**. D.6 explicitly depends on D.5 being complete.

**Note on inner scan path:** The `multi_state_sampling.geometric_mean_logits` function at lines 91–128
still accepts `temperature` as a parameter — do **not** change that. It is called from the inner scan
path which is not yet Pipeline-ized. Only remove `multi_state_temperature` from the *outer* model
method boundaries listed in D.2a, D.2b, D.3a, D.3b, and the callers in D.5.

---

### Task D.1 — Add `make_geometric_mean_transform` factory to pipeline_registry.py

**Files:**
- Modify: `src/prxteinmpnn/pipeline_registry.py`
- Test: `tests/test_sprint_modelinputs.py`

**Note on registry idempotency:** Each distinct closure created by `make_geometric_mean_transform(T)`
produces a different `cloudpickle` hash and therefore a new UID in the registry. For temperature
sweeps this leaks registry entries. To avoid this, add a module-level `_geom_mean_cache: dict[float,
Any] = {}` in pipeline_registry.py so that repeated calls with the same `T` return the same closure
object (and thus the same UID).

- [ ] **Step D.1.1: Write failing test**

```python
# tests/test_sprint_modelinputs.py
import jax.numpy as jnp
from prxteinmpnn.pipeline_registry import make_geometric_mean_transform, register_hook

def test_make_geometric_mean_transform_factory():
    T = 0.5
    fn = make_geometric_mean_transform(T)
    state_logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # (S=2, 3)
    result = fn(state_logits, state_index=None, state_weights=None)
    # Closed-form: mean(state_logits, axis=0) / T
    expected = jnp.mean(state_logits, axis=0) / T
    assert jnp.allclose(result, expected, atol=1e-5)

def test_make_geometric_mean_transform_registerable():
    fn = make_geometric_mean_transform(0.1)
    uid = register_hook(fn, name="test_geom_mean")
    assert isinstance(uid, str) and len(uid) == 16

def test_make_geometric_mean_transform_cache_idempotent():
    """Same temperature must return the same closure object (no registry leak)."""
    fn_a = make_geometric_mean_transform(0.5)
    fn_b = make_geometric_mean_transform(0.5)
    assert fn_a is fn_b, "Same temperature must return the same cached closure"
```

- [ ] **Step D.1.2: Run test to verify it fails**

```bash
cd /home/marielle/projects/tev_design/prxteinmpnn
uv run pytest tests/test_sprint_modelinputs.py::test_make_geometric_mean_transform_factory -xvs
```

Expected: `ImportError: cannot import name 'make_geometric_mean_transform'`

- [ ] **Step D.1.3: Implement the factory with cache**

In `src/prxteinmpnn/pipeline_registry.py`, add before `__all__`:

```python
_geom_mean_cache: dict[float, Any] = {}


def make_geometric_mean_transform(temperature: float) -> Any:
    """Return a memoised LogitTransformFn that scales arithmetic mean by 1/temperature.

    Suitable for geometric-mean multistate sampling: log P_geom ∝ mean(logits) / T.
    Memoised so that the same T always returns the same closure object,
    preventing registry UID proliferation during temperature sweeps.

    Note: different temperatures produce different compiled JAX artifacts
    (recompile-on-change), because temperature is captured at closure creation
    time as a Python float, not a traced array.
    """
    import jax.numpy as jnp  # noqa: PLC0415

    if temperature in _geom_mean_cache:
        return _geom_mean_cache[temperature]

    def _geom_mean(state_logits: Any, _state_index: Any, _state_weights: Any) -> Any:
        return jnp.mean(state_logits, axis=0) / temperature

    _geom_mean_cache[temperature] = _geom_mean
    return _geom_mean
```

Also add `"make_geometric_mean_transform"` to `__all__`.

- [ ] **Step D.1.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_sprint_modelinputs.py -xvs -k "make_geometric_mean"
```

Expected: 3 passed.

- [ ] **Step D.1.5: Commit**

```bash
git add src/prxteinmpnn/pipeline_registry.py tests/test_sprint_modelinputs.py
git commit -m "feat(D.1): add memoised make_geometric_mean_transform factory to pipeline_registry"
```

---

### Task D.2a — Remove `multi_state_temperature` from mpnn.py model boundaries

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`

The `lax.switch` operand list at ~lines 951–972 currently has 19 operands; `multi_state_temperature`
is operand #11 (0-indexed). Three branch functions reference it: `_call_unconditional` (~190),
`_call_conditional` (~274), `_call_autoregressive` (~360). The public `__call__` at line 649 and all
`*_from_payload` / `*_state_vmap_exact` method aliases that accept it must also be updated.

- [ ] **Step D.2a.1: Grep to enumerate all occurrences before editing**

```bash
grep -n "multi_state_temperature" src/prxteinmpnn/model/mpnn.py
```

Record the line numbers. You will update every one of them.

- [ ] **Step D.2a.2: Remove from `_call_unconditional`, `_call_conditional`, `_call_autoregressive`**

For each of the three `_call_*` branch functions, remove the `multi_state_temperature` parameter from
the function signature and remove any usage of it inside the function body (it is only passed through
to an inner scoring/sampling helper — that helper receives temperature via `PipelineFns` in D.2b).

Example pattern (apply to all three branches):

```python
# Before
def _call_unconditional(
    module,
    prng_key,
    ...,
    multi_state_temperature,   # <-- remove this line
    ...
):

# After
def _call_unconditional(
    module,
    prng_key,
    ...,
    # multi_state_temperature removed: temperature captured in LogitTransformFn closure
    ...
):
```

- [ ] **Step D.2a.3: Remove from `__call__` signature (line 649) and lax.switch operand list**

In `__call__`, remove `multi_state_temperature` from the parameter list. Then in the `lax.switch`
call (~lines 951–972), remove it from the operands list. The 3 branch functions must match
(otherwise JAX will raise a shape/count mismatch at trace time). After removal the operand count
goes from 19 to 18.

- [ ] **Step D.2a.4: Remove from all `*_from_payload` and `*_state_vmap_exact` public aliases**

Search for all methods in `mpnn.py` that include `multi_state_temperature` as a parameter:

```bash
grep -n "def.*multi_state_temperature\|multi_state_temperature" src/prxteinmpnn/model/mpnn.py
```

Remove the parameter from each method signature and its forwarding call.

- [ ] **Step D.2a.5: Write regression test**

```python
# in tests/test_sprint_modelinputs.py
import jax
from prxteinmpnn.model.mpnn import PrxteinMPNN

def test_mpnn_score_unconditional_no_temperature_param():
    import inspect
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    sig = inspect.signature(m.score_unconditional_state_vmap_exact)
    assert "multi_state_temperature" not in sig.parameters, (
        "multi_state_temperature must not appear in score_unconditional_state_vmap_exact"
    )
```

- [ ] **Step D.2a.6: Run test**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_mpnn_score_unconditional_no_temperature_param -xvs
```

Expected: PASS.

- [ ] **Step D.2a.7: Run existing sampling/parity tests to catch regressions**

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  -q --tb=short
```

Expected: No new failures.

- [ ] **Step D.2a.8: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/test_sprint_modelinputs.py
git commit -m "refactor(D.2a): remove multi_state_temperature from mpnn.py model boundaries"
```

---

### Task D.2b — Update four downstream model files that receive multi_state_temperature

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py`
- Modify: `src/prxteinmpnn/model/mpnn_autoregressive_scan.py`
- Modify: `src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py`
- Note: `multi_state_sampling.geometric_mean_logits` itself is NOT touched here.

These files receive `multi_state_temperature` as a positional/keyword arg from the model methods
updated in D.2a. After D.2a those callers no longer pass it. The correct mechanism:

**For inner-scan path callers that call `geometric_mean_logits(logits, group_mask, temperature)`:**
The temperature value is no longer a traced pytree arg — it is resolved at JIT-compile time from the
`LogitTransformFn` registered in `PipelineFns`. The concrete pattern is:
1. The outer model method receives `batch_fn` (which is a `PipelineFns` object) in its
   `static_argnames` at the `@eqx.filter_jit` boundary.
2. At compile time (not trace time), call `make_geometric_mean_transform(T)` to build the closure,
   where `T` is extracted as a Python float from `batch_fn` metadata. Alternatively, replace the
   direct `geometric_mean_logits(logits, mask, temperature)` call with a call to the registered
   `LogitTransformFn`: `logit_transform_fn(logits_stacked, state_index, state_weights)`.
3. Because `LogitTransformFn` is resolved from `PipelineFns` at compile time (it is a UID string,
   which is static), this does not add a new traced operand. Different temperatures produce
   different compiled artifacts — this is expected and acceptable for this codebase.

**Important:** Do NOT use a default temperature value (e.g. `temperature=1.0`) as a fallback when
the `LogitTransformFn` approach is not yet available. That would silently change scoring behaviour
across multistate calls. If threading PipelineFns to a particular call site requires more refactoring
than fits in this sprint, leave it as a compiler error (missing arg) and document it as a D.2b-blocker
rather than silently changing behavior.

- [ ] **Step D.2b.1: Audit current usage in each file**

```bash
grep -n "multi_state_temperature" \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py \
  src/prxteinmpnn/model/mpnn_autoregressive_scan.py \
  src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py
```

Record each occurrence. Distinguish: (a) function signature, (b) passed to `geometric_mean_logits`,
(c) used in another way.

- [ ] **Step D.2b.2: Confirm PipelineFns / batch_fn threading in each file**

For each of the three files, search for `batch_fn` or `fns` to confirm it is already in scope:

```bash
grep -n "batch_fn\|pipeline_fns\|PipelineFns" \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py \
  src/prxteinmpnn/model/mpnn_autoregressive_scan.py \
  src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py
```

If `batch_fn` / `fns` is present: use `resolve_hook(fns.logit_transform_fn_uid)` to get the
transform, then call it in place of `geometric_mean_logits(logits, mask, temperature)`.

If `batch_fn` / `fns` is NOT present in a file: do not add a silent default. Instead, add it to the
function signature with a type of `PipelineFns` and thread it from the D.2a call site.

- [ ] **Step D.2b.3: Remove from signatures and replace geometric_mean_logits calls**

For each file:
- (a) Remove `multi_state_temperature` from the function signature.
- (b) Replace `geometric_mean_logits(logits, group_mask, multi_state_temperature, ...)` with the
  `LogitTransformFn` call pattern described in D.2b.2.

- [ ] **Step D.2b.4: Verify no multi_state_temperature references remain in these three files**

```bash
grep -c "multi_state_temperature" \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py \
  src/prxteinmpnn/model/mpnn_autoregressive_scan.py \
  src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py
```

Expected: all three lines show `0`.

- [ ] **Step D.2b.5: Run regression tests**

```bash
PYTHONPATH=src uv run pytest tests/sampling/ tests/model/ -q --tb=short
```

Expected: No new failures.

- [ ] **Step D.2b.6: Commit**

```bash
git add \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py \
  src/prxteinmpnn/model/mpnn_autoregressive_scan.py \
  src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py
git commit -m "refactor(D.2b): remove multi_state_temperature from downstream mpnn model files"
```

---

### Task D.3a — Remove `multi_state_temperature` from ligand_mpnn.py boundaries

**Files:**
- Modify: `src/prxteinmpnn/model/ligand_mpnn.py`

Parallel to D.2a. Apply the same pattern: remove from `__call__` and all `*_from_payload` /
`*_state_vmap_exact` public method signatures. Remove from the `lax.switch` operand list.

- [ ] **Step D.3a.1: Grep to enumerate all occurrences**

```bash
grep -n "multi_state_temperature" src/prxteinmpnn/model/ligand_mpnn.py
```

- [ ] **Step D.3a.2: Remove from all method signatures and operand list**

Follow the same pattern as D.2a. Do not touch `ligand_encode_stack_row` (the encoder helper) —
that is wired in C.3, not here.

- [ ] **Step D.3a.3: Verify grep == 0 for ligand_mpnn.py**

```bash
grep -c "multi_state_temperature" src/prxteinmpnn/model/ligand_mpnn.py
```

Expected: `0`.

- [ ] **Step D.3a.4: Commit**

```bash
git add src/prxteinmpnn/model/ligand_mpnn.py
git commit -m "refactor(D.3a): remove multi_state_temperature from ligand_mpnn.py boundaries"
```

---

### Task D.3b — Update ligand downstream file

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py`

- [ ] **Step D.3b.1: Grep and remove**

```bash
grep -n "multi_state_temperature" \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py
```

Remove from signature; follow the same `PipelineFns`/`LogitTransformFn` pattern as D.2b.

- [ ] **Step D.3b.2: Verify grep == 0**

```bash
grep -c "multi_state_temperature" \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py
```

Expected: `0`.

- [ ] **Step D.3b.3: Commit**

```bash
git add src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py
git commit -m "refactor(D.3b): remove multi_state_temperature from ligand autoregressive file"
```

---

### Task D.4 — Verify all sprint-edited model files are clean; behavioral temperature test

**Files:**
- Test: `tests/test_sprint_modelinputs.py`

- [ ] **Step D.4.1: Grep on sprint-edited files to confirm zero references**

```bash
grep -c "multi_state_temperature" \
  src/prxteinmpnn/model/mpnn.py \
  src/prxteinmpnn/model/ligand_mpnn.py \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact.py \
  src/prxteinmpnn/model/mpnn_autoregressive_scan.py \
  src/prxteinmpnn/model/mpnn_scoring_state_vmap_exact_ligand.py \
  src/prxteinmpnn/model/mpnn_autoregressive_state_vmap_exact_ligand.py
```

Expected: every file shows `0`. Note: `scoring/score.py` and `sampling/sample.py` will still show
references — those are migrated in D.5 (the next task).

- [ ] **Step D.4.2: Write behavioral temperature test to confirm D.1 factory affects output**

```python
# in tests/test_sprint_modelinputs.py
def test_geometric_mean_transform_temperature_effect():
    import jax.numpy as jnp
    from prxteinmpnn.pipeline_registry import make_geometric_mean_transform
    state_logits = jnp.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]])
    fn_hot = make_geometric_mean_transform(2.0)
    fn_cold = make_geometric_mean_transform(0.5)
    out_hot = fn_hot(state_logits, None, None)
    out_cold = fn_cold(state_logits, None, None)
    # cold (T=0.5) should produce 4x the magnitude of hot (T=2.0)
    assert jnp.allclose(out_cold, 4.0 * out_hot, atol=1e-5)
```

- [ ] **Step D.4.3: Run behavioral test**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_geometric_mean_transform_temperature_effect -xvs
```

Expected: PASS.

- [ ] **Step D.4.4: Commit**

```bash
git add tests/test_sprint_modelinputs.py
git commit -m "test(D.4): verify multi_state_temperature removed from all sprint model files"
```

---

### Task D.5 — Migrate scoring/score.py and sampling/sample.py to remove temperature kwarg

**Purpose:** Closes the cross-sprint API break. After D.2a/D.3a remove `multi_state_temperature`
from model method signatures, these two files still pass it as a keyword/positional arg,
causing `TypeError` at runtime. This task migrates them before D.6 removes the field from
`SamplingControls`. **D.5 must be complete before D.6.**

**Files:**
- Modify: `src/prxteinmpnn/scoring/score.py` — lines 146, 166
- Modify: `src/prxteinmpnn/sampling/sample.py` — lines 607, 622

**Migration pattern (mandatory):** Replace the `multi_state_temperature=jnp.asarray(...)` kwarg
with a `LogitTransformFn` closure registered in `PipelineFns`. The correct pattern:

```python
from prxteinmpnn.pipeline_registry import make_geometric_mean_transform
from prxteinmpnn.pipeline_fns import PipelineFns

# Before:
logits = model.score_conditional_state_vmap_exact(
    ...,
    multi_state_temperature=jnp.asarray(multi_state_temperature, jnp.float32),
    ...
)

# After: thread PipelineFns; temperature is captured in the LogitTransformFn closure.
# PipelineFns is a frozen dataclass — use from_callables(), not .replace()
_transform_fn = make_geometric_mean_transform(float(multi_state_temperature))
fns = PipelineFns.from_callables(logit_transform=_transform_fn)
logits = model.score_conditional_state_vmap_exact(
    ...,
    fns=fns,
    ...
)
```

**Do NOT use `**_ignored_legacy_kwargs` or any kwarg-absorbing shim** — that silently discards
the temperature value, causing geometric-mean fusion to run at T=1.0 regardless of caller intent
with no error or warning. Temperature semantics must be preserved.

If `PipelineFns` is not already in scope at these call sites, thread it down from the outer function
parameter. Both `score.py` and `sampling/sample.py` are host-side (not inside JIT traces), so
threading `fns` as a new function parameter is safe.

- [ ] **Step D.5.1: Read scoring/score.py context around lines 100–175**

Read `src/prxteinmpnn/scoring/score.py` lines 100–175 to confirm:
- how `multi_state_temperature` is received (as a function parameter)
- whether `PipelineFns` is already imported or in scope
- what outer function to add `fns: PipelineFns` parameter to if not already present

- [ ] **Step D.5.2: Read sampling/sample.py context around lines 575–640**

Read `src/prxteinmpnn/sampling/sample.py` lines 575–640 for the same purpose.

- [ ] **Step D.5.3: Update scoring/score.py**

At lines 146 and 166, replace `multi_state_temperature=jnp.asarray(...)` with the `PipelineFns`
threading pattern shown above. Add `fns: PipelineFns | None = None` to the outer function signature
if not already present (use `PipelineFns.default()` as fallback).

- [ ] **Step D.5.4: Update sampling/sample.py**

Apply the same pattern at lines 607 and 622.

- [ ] **Step D.5.5: Write behavioral test through the actual score.py call path**

This test calls `_make_score_fn_state_vmap_exact` (the function modified in D.5.3/D.5.4)
with two different `PipelineFns` objects (T=0.1 vs T=2.0) and asserts the returned logits differ.
This exercises the real code path — not just the factory in isolation. Written for the
post-D.5 signature of `score_sequence_core_inner` (no `multi_state_temperature`, `fns` added).

```python
# in tests/test_sprint_modelinputs.py
def test_score_py_temperature_semantics_preserved():
    """After D.5 migration, scoring via _make_score_fn_state_vmap_exact with two
    different LogitTransformFn temperatures must yield different logits.

    If score.py silently discards temperature (e.g., **kwargs or missing threading),
    logits will be identical regardless of fns — and this assertion fails.
    """
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.scoring.score import _make_score_fn_state_vmap_exact
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.pipeline_registry import make_geometric_mean_transform
    from tests.pipeline.test_autoregressive import _make_stack_and_wave

    S, L = 2, 6
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)
    stack, _ = _make_stack_and_wave(S=S, L=L)

    fns_T01 = PipelineFns.from_callables(logit_transform=make_geometric_mean_transform(0.1))
    fns_T20 = PipelineFns.from_callables(logit_transform=make_geometric_mean_transform(2.0))

    # Post-D.5 calling convention: multi_state_temperature removed, fns added
    score_fn = _make_score_fn_state_vmap_exact(m, inference=False)

    coords = jnp.zeros((S, L, 4, 3))
    seq_flat = jnp.zeros(S * L, dtype=jnp.int32)
    ri = jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0)
    ci = jnp.zeros((S, L), dtype=jnp.int32)

    common_kw = dict(
        coords_stack=coords,
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=ri,
        chain_index_stack=ci,
        state_flat_rows=stack.state_flat_rows,
        n_flat_int=int(S * L),
        structure_mapping=None,
        tie_group_map=None,
        multi_state_strategy="geometric_mean",
        state_weights=jnp.ones(S),
        y_stack=None,
        y_t_stack=None,
        y_m_stack=None,
        ar_mask_stack=None,
        bias_flat=None,
    )
    _, logits_T01, _ = score_fn(key, seq_flat, fns=fns_T01, **common_kw)
    _, logits_T20, _ = score_fn(key, seq_flat, fns=fns_T20, **common_kw)

    assert not jnp.allclose(logits_T01, logits_T20, atol=1e-5), (
        "T=0.1 and T=2.0 must produce different logits via _make_score_fn_state_vmap_exact. "
        "If temperature is silently discarded, this assertion fails."
    )
```

- [ ] **Step D.5.6: Run test**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_score_py_temperature_semantics_preserved -xvs
```

Expected: PASS.

- [ ] **Step D.5.7: Commit**

```bash
git add src/prxteinmpnn/scoring/score.py src/prxteinmpnn/sampling/sample.py \
  tests/test_sprint_modelinputs.py
git commit -m "refactor(D.5): migrate score.py + sample.py to LogitTransformFn, preserve temperature"
```

---

### Task D.6 — Remove `SamplingControls.multi_state_temperature` from payloads.py

**Depends on:** D.5 complete (verified in step D.6.1).

**Files:**
- Modify: `src/prxteinmpnn/payloads.py`

`SamplingControls` at lines 123–141 has `multi_state_temperature: Float[Array, ...]` as a field.
After D.2a/D.2b/D.3a/D.3b remove it from model methods, and D.5 removes it from the two deferred
callers, this field has no remaining readers.

- [ ] **Step D.6.1: Verify D.5 is complete first (dependency check)**

```bash
git log --oneline | grep "D.5"
```

Do not proceed if D.5 is not committed.

- [ ] **Step D.6.2: Remove `multi_state_temperature` field from `SamplingControls`**

In `src/prxteinmpnn/payloads.py`, remove:
```python
  multi_state_temperature: Float[Array, ...]
```
from `SamplingControls` and remove `"multi_state_temperature"` from the `fields` tuple in its
`replace()` method.

- [ ] **Step D.6.3: Verify no readers remain in scoring/ and sampling/**

```bash
grep -c "multi_state_temperature" \
  src/prxteinmpnn/scoring/score.py \
  src/prxteinmpnn/sampling/sample.py
```

Expected: both show `0`.

- [ ] **Step D.6.4: Run full test suite**

```bash
PYTHONPATH=src uv run pytest tests/ -q --tb=short -x
```

Expected: No failures.

- [ ] **Step D.6.5: Commit**

```bash
git add src/prxteinmpnn/payloads.py
git commit -m "refactor(D.6): remove SamplingControls.multi_state_temperature field"
```

---

## Sprint C — Wire EncoderPreFn / EncoderPostFn

---

### Task C.0 — Define `EncoderPreOutput` and fix Protocol return type

**Files:**
- Modify: `src/prxteinmpnn/payloads.py`
- Modify: `src/prxteinmpnn/protocols.py`
- Test: `tests/test_sprint_modelinputs.py`

The current `EncoderPreFn.return` type is `dict[str, Any] | None` (protocols.py:238). This is
**not JAX-pytree-stable**: `dict` and `None` have different tree structures, which causes retrace
on every call. Replace it with `EncoderPreOutput(eqx.Module)` — a concrete module with fixed fields.

- [ ] **Step C.0.1: Write failing JIT trace-count test**

```python
# in tests/test_sprint_modelinputs.py
def test_encoder_pre_output_jit_stable():
    """EncoderPreOutput must not cause retrace on same-shape second call."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from prxteinmpnn.payloads import EncoderPreOutput

    trace_count = [0]

    @eqx.filter_jit
    def fn(pre: EncoderPreOutput) -> jax.Array:
        trace_count[0] += 1
        return pre.initial_node_features.sum()

    S, L, D, k = 2, 6, 16, 4
    out1 = EncoderPreOutput(
        initial_node_features=jnp.zeros((S, L, D)),
        rbf_features=jnp.zeros((S, L, D)),
        neighbor_indices=jnp.zeros((S, L, k), dtype=jnp.int32),
    )
    out2 = EncoderPreOutput(
        initial_node_features=jnp.ones((S, L, D)),
        rbf_features=jnp.ones((S, L, D)),
        neighbor_indices=jnp.ones((S, L, k), dtype=jnp.int32),
    )
    fn(out1)
    fn(out2)
    assert trace_count[0] == 1, f"Expected 1 trace, got {trace_count[0]}"
```

- [ ] **Step C.0.2: Run test to verify it fails**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_encoder_pre_output_jit_stable -xvs
```

Expected: `ImportError: cannot import name 'EncoderPreOutput' from 'prxteinmpnn.payloads'`

- [ ] **Step C.0.3: Add `EncoderPreOutput` to payloads.py**

In `src/prxteinmpnn/payloads.py`, after `EncoderOutput` (line 175), add:

```python
class EncoderPreOutput(eqx.Module):
    """Pre-encoder feature override returned by EncoderPreFn hooks.

    All fields are concrete arrays — no Optional[Array] fields allowed,
    as those would break JAX pytree stability under JIT.
    Return the encoder's own defaults for fields you don't want to override.
    Hook absence is expressed via PipelineFns.encoder_pre_process_uid = None,
    not by returning None from the hook.
    """

    initial_node_features: Float[Array, "S L D"]
    rbf_features: Float[Array, "S L D"]
    neighbor_indices: Int[Array, "S L k"]
```

- [ ] **Step C.0.4: Update `EncoderPreFn` Protocol in protocols.py**

Change the return annotation on `EncoderPreFn.__call__` from:

```python
  ) -> dict[str, Any] | None: ...
```

to:

```python
  ) -> EncoderPreOutput: ...
```

Also update the docstring to remove mention of `None` return and `dict`.

- [ ] **Step C.0.5: Update `__all__` in payloads.py to include `EncoderPreOutput`**

- [ ] **Step C.0.6: Run JIT stability test**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_encoder_pre_output_jit_stable -xvs
```

Expected: PASS (trace_count == 1).

- [ ] **Step C.0.7: Commit**

```bash
git add src/prxteinmpnn/payloads.py src/prxteinmpnn/protocols.py tests/test_sprint_modelinputs.py
git commit -m "feat(C.0): add EncoderPreOutput pytree-stable type, fix EncoderPreFn return type"
```

---

### Task C.1 — Wire EncoderPreFn in mpnn.py (outside vmap)

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Test: `tests/test_sprint_modelinputs.py`

`EncoderPreFn` must be called **outside** `jax.vmap(encode_one)`, on the full stacked geometry
(shape `[S, L, ...]`), before vmap is applied. Calling inside vmap would re-trace per state.
The hook result (`EncoderPreOutput`) is sliced per-state inside the `encode_one` closure.

The encoder wiring points in `mpnn.py` are:
- `encode_one` closure at ~line 1110 (for `score_unconditional_state_vmap_exact`)
- `encode_one` closure at ~line 1252 (for `score_conditional_state_vmap_exact`)

- [ ] **Step C.1.1: Write shape-based falsifiability test**

```python
# in tests/test_sprint_modelinputs.py
def test_encoder_pre_fn_called_outside_vmap():
    """EncoderPreFn must receive the full state stack (S states), not a single state."""
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.payloads import EncoderPreOutput
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.pipeline_registry import register_encoder_pre_fn

    S, L, D, k = 2, 6, 16, 4
    observed_shape = []

    def _pre_hook(backbone, state_index):
        # Record the leading dimension of coords — must be S, not 1
        observed_shape.append(backbone.coords.shape[0])
        return EncoderPreOutput(
            initial_node_features=jnp.zeros((S, L, D)),
            rbf_features=jnp.zeros((S, L, D)),
            neighbor_indices=jnp.zeros((S, L, k), dtype=jnp.int32),
        )

    # PipelineFns is a frozen dataclass; use from_callables to register and create in one call
    fns = PipelineFns.from_callables(encoder_pre_process=_pre_hook)
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(D, D, D, 1, 1, k, key=key)

    from prxteinmpnn.model_inputs import BackboneGeometry
    coords = jnp.zeros((S, L, 4, 3))
    mask = jnp.ones((S, L))
    backbone = BackboneGeometry(
        coords=coords,
        mask=mask,
        residue_index=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
    )
    m.score_unconditional_from_payload(key, backbone, fns=fns)

    assert observed_shape, "EncoderPreFn was never called"
    assert observed_shape[0] == S, (
        f"Expected backbone.coords.shape[0] == S={S}, got {observed_shape[0]}. "
        "Hook is being called inside vmap (wrong)."
    )
```

- [ ] **Step C.1.2: Run test to verify it fails (hook not yet wired)**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_encoder_pre_fn_called_outside_vmap -xvs
```

Expected: either the hook is never called (observed_shape empty) or fails shape assertion.

- [ ] **Step C.1.3: Wire EncoderPreFn at both encode_one sites in mpnn.py**

For each `encode_one` closure (~1110, ~1252):

1. Before `jax.vmap(encode_one)(...)`, check if `fns.encoder_pre_process_uid` is not None.
2. If set, resolve the hook via `resolve_hook(fns.encoder_pre_process_uid)`.
3. Call `pre_output = hook(full_backbone_stack, state_index_array)` on the full stacked geometry.
4. Inside `encode_one`, slice `pre_output` to get per-state features for the current state.

The `PipelineFns` object is already threaded to the call sites via `batch_fn` / `fns` — find
how it arrives at the encode_one closure and use the same channel.

- [ ] **Step C.1.4: Run shape-based test**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_encoder_pre_fn_called_outside_vmap -xvs
```

Expected: PASS (observed_shape[0] == S == 2).

- [ ] **Step C.1.5: Run regression tests**

```bash
PYTHONPATH=src uv run pytest tests/sampling/ tests/model/ tests/pipeline/ -q --tb=short
```

Expected: No new failures.

- [ ] **Step C.1.6: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/test_sprint_modelinputs.py
git commit -m "feat(C.1): wire EncoderPreFn in mpnn.py outside jax.vmap(encode_one)"
```

---

### Task C.2 — Wire EncoderPostFn in mpnn.py

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`

`EncoderPostFn` is called after `jax.vmap(encode_one)` returns the full stacked
`(node_features, edge_features, neighbor_indices, mask)`. It receives the stacked `EncoderOutput`
and returns a (possibly modified) `EncoderOutput`. If `fns.encoder_post_process_uid` is None,
pass through unchanged.

- [ ] **Step C.2.1: Write wiring test**

```python
# in tests/test_sprint_modelinputs.py
def test_encoder_post_fn_receives_encoder_output():
    """EncoderPostFn must receive EncoderOutput with correct batch dimensions."""
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.protocols import EncoderOutput
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.pipeline_registry import register_encoder_post_fn

    S, L, D, k = 2, 6, 16, 4
    received = []

    def _post_hook(encoded: EncoderOutput, state_index):
        received.append((encoded.node_features.shape, encoded.edge_features.shape))
        return encoded

    # PipelineFns is a frozen dataclass; use from_callables to register and create in one call
    fns = PipelineFns.from_callables(encoder_post_process=_post_hook)
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(D, D, D, 1, 1, k, key=key)

    from prxteinmpnn.model_inputs import BackboneGeometry
    backbone = BackboneGeometry(
        coords=jnp.zeros((S, L, 4, 3)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
    )
    m.score_unconditional_from_payload(key, backbone, fns=fns)

    assert received, "EncoderPostFn was never called"
    node_shape, edge_shape = received[0]
    assert node_shape[0] == S, f"Expected S={S} in node_features leading dim, got {node_shape[0]}"
```

- [ ] **Step C.2.2: Implement and verify**

Wire `EncoderPostFn` at the same two encode-call sites in mpnn.py, immediately after
`jax.vmap(encode_one)(...)` returns. Pass the result through the hook if set.

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_encoder_post_fn_receives_encoder_output -xvs
```

Expected: PASS.

- [ ] **Step C.2.3: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/test_sprint_modelinputs.py
git commit -m "feat(C.2): wire EncoderPostFn in mpnn.py after jax.vmap(encode_one)"
```

---

### Task C.3 — Wire EncoderPreFn and EncoderPostFn in ligand_mpnn.py

**Files:**
- Modify: `src/prxteinmpnn/model/ligand_mpnn.py`

The ligand encoder wiring point is `ligand_encode_stack_row` at ~line 1300. Apply the same
outside-vmap pattern as C.1/C.2. The backbone geometry includes `yy/yt/ym` ligand arrays —
confirm `EncoderPreOutput` fields cover what ligand_mpnn uses, or extend them.

- [ ] **Step C.3.1: Read ligand_encode_stack_row (lines 1280–1340)**

Confirm which features come from `model.features(...)` at line 1300 and which of those
`EncoderPreOutput` fields correspond to (V, E, E_idx etc.).

- [ ] **Step C.3.2: Wire hooks**

Apply the same pattern: call `EncoderPreFn` on the full ligand stack before vmap, slice inside
closure; call `EncoderPostFn` after vmap returns.

- [ ] **Step C.3.3: Write ligand encoder hook test mirroring C.1.1**

```python
# in tests/test_sprint_modelinputs.py
def test_ligand_encoder_pre_fn_called_outside_vmap():
    """EncoderPreFn must see S-state backbone stack in ligand_mpnn path."""
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN  # class is PrxteinLigandMPNN
    from prxteinmpnn.payloads import EncoderPreOutput, LigandStack, MultistateStackPayload
    from prxteinmpnn.pipeline_fns import PipelineFns
    from tests.pipeline.test_autoregressive import _make_stack_and_wave

    S, L, D, k = 2, 6, 16, 4
    observed_shape = []

    def _pre_hook(backbone, state_index):
        observed_shape.append(backbone.coords.shape[0])
        return EncoderPreOutput(
            initial_node_features=jnp.zeros((S, L, D)),
            rbf_features=jnp.zeros((S, L, D)),
            neighbor_indices=jnp.zeros((S, L, k), dtype=jnp.int32),
        )

    # PipelineFns is a frozen dataclass; use from_callables to register and create
    fns = PipelineFns.from_callables(encoder_pre_process=_pre_hook)
    key = jax.random.PRNGKey(0)
    # PrxteinLigandMPNN constructor: (node_features, edge_features, hidden_features,
    # num_encoder_layers, num_decoder_layers, k_neighbors, *, key)
    m = PrxteinLigandMPNN(D, D, D, 1, 1, k, key=key)

    stack, _ = _make_stack_and_wave(S=S, L=L)
    ligand_n = 4  # number of ligand atoms
    ligand_stack = LigandStack(
        y_stack=jnp.zeros((S, ligand_n, 3)),
        # dtype per A.0 audit — adjust to Int if A.0 finds y_t_stack is an integer token index
        y_t_stack=jnp.zeros((S, ligand_n)),
        y_m_stack=jnp.ones((S, ligand_n)),
    )
    # score_unconditional_state_vmap_exact_from_payload is the equivalent of
    # PrxteinMPNN.score_unconditional_from_payload for ligand paths (D.3a removes
    # multi_state_temperature from this method's signature before C runs)
    m.score_unconditional_state_vmap_exact_from_payload(
        key,
        stack,
        ligand_stack,
        tie_group_map=None,
        multi_state_strategy_idx=jnp.int32(0),
        state_weights=None,
        state_mapping=None,
        fns=fns,
    )

    assert observed_shape, "EncoderPreFn was never called in PrxteinLigandMPNN path"
    assert observed_shape[0] == S, (
        f"Expected backbone.coords.shape[0] == S={S}, got {observed_shape[0]}. "
        "Hook is inside vmap (wrong)."
    )
```

- [ ] **Step C.3.4: Run regression tests**

```bash
PYTHONPATH=src uv run pytest tests/ -q --tb=short -x
```

Expected: No failures.

- [ ] **Step C.3.5: Commit**

```bash
git add src/prxteinmpnn/model/ligand_mpnn.py tests/test_sprint_modelinputs.py
git commit -m "feat(C.3): wire EncoderPreFn/EncoderPostFn in ligand_mpnn.py"
```

---

## Sprint A — MODELINPUTS PR-4: Push model.__call__ boundary

---

### Task A.0 — Pre-audit: confirm LigandStack.y_t_stack dtype

**Files:**
- Read-only: `src/prxteinmpnn/model/ligand_mpnn.py`, `src/prxteinmpnn/payloads.py`
- Record finding: add a comment in `payloads.py` above `LigandStack.y_t_stack`

`payloads.py:103` types `y_t_stack` as `Float[Array, ...]`. But ligand integer token indices
are typically `Int`. Confirm by tracing `yt` in `ligand_encode_stack_row` (line 1300) through
`model.features(fe_k, coords, ma, ri, ci, yy, yt, ym, ...)` to see if it is used in integer
index operations (e.g. `jnp.take`) or float arithmetic.

- [ ] **Step A.0.1: Read ligand_mpnn.py features call and downstream usage**

```bash
grep -n "yt\b\|y_t\b\|y_t_stack" src/prxteinmpnn/model/ligand_mpnn.py | head -40
```

Trace the type expectation: integer embedding index or float continuous feature.

- [ ] **Step A.0.2: Record finding**

In `src/prxteinmpnn/payloads.py`, update the `LigandStack` docstring to add one line:

```python
# dtype audit (2026-05-08): y_t_stack is [Float|Int] because <reason>
```

Replace `[Float|Int]` and `<reason>` with the actual finding. If it should be `Int`, update
the type annotation from `Float` to `Int`.

- [ ] **Step A.0.3: Commit**

```bash
git add src/prxteinmpnn/payloads.py
git commit -m "docs(A.0): record y_t_stack dtype audit finding in LigandStack"
```

---

### Task A.1 — Define `LigandSamplingInputs` in model_inputs.py

**Files:**
- Modify: `src/prxteinmpnn/model_inputs.py`
- Test: `tests/test_sprint_modelinputs.py`

`LigandSamplingInputs` embeds the full `LigandStack` type from `payloads.py` — do **not** inline
`y_stack / y_t_stack / y_m_stack` as separate fields. This avoids dtype mismatch duplication.

- [ ] **Step A.1.1: Write failing test**

```python
# in tests/test_sprint_modelinputs.py
def test_ligand_sampling_inputs_embeds_ligand_stack():
    from prxteinmpnn.model_inputs import LigandSamplingInputs
    import inspect
    sig = inspect.signature(LigandSamplingInputs)
    assert "ligand_stack" in sig.parameters, "LigandSamplingInputs must have ligand_stack field"
    assert "y_stack" not in sig.parameters, "y_stack must not be inlined (use LigandStack)"
    assert "y_t_stack" not in sig.parameters, "y_t_stack must not be inlined (use LigandStack)"
```

- [ ] **Step A.1.2: Run test to verify it fails**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_ligand_sampling_inputs_embeds_ligand_stack -xvs
```

Expected: `ImportError` or `AssertionError`.

- [ ] **Step A.1.3: Define `LigandSamplingInputs` in model_inputs.py**

```python
class LigandSamplingInputs(eqx.Module):
    """Pytree-safe inputs for LigandMPNN model.__call__ (MODELINPUTS PR-4)."""

    backbone: BackboneGeometry
    state_stack: MultistateStackPayload
    wave_parallel: WaveParallelPayload
    conditioning: ConditioningFeatures
    ligand_stack: LigandStack  # embeds existing payloads.py type; no field duplication
```

- [ ] **Step A.1.4: Run test**

```bash
uv run pytest tests/test_sprint_modelinputs.py::test_ligand_sampling_inputs_embeds_ligand_stack -xvs
```

Expected: PASS.

- [ ] **Step A.1.5: Commit**

```bash
git add src/prxteinmpnn/model_inputs.py tests/test_sprint_modelinputs.py
git commit -m "feat(A.1): define LigandSamplingInputs with embedded LigandStack in model_inputs"
```

---

### Task A.2 — Add `PrxteinMPNN.forward_score` and `forward_sample` entry points (additive)

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`
- Test: `tests/test_sprint_modelinputs.py`

This is **additive**: add two entry points alongside `__call__`. Each has a stable, single return type
(not a union of Array and None). Keep `__call__` unchanged — parity tests still exercise it.

**Return type contract:**
- `forward_score(inputs: SamplingInputs, *, fns: PipelineFns) -> jax.Array` — returns logits only.
- `forward_sample(inputs: SamplingInputs, *, fns: PipelineFns) -> tuple[jax.Array, jax.Array]` —
  returns `(sequences, logits)`.

Do NOT use `None` as a return value or a union type like `Array | None`. These are not JAX-pytree-stable and cause retrace and `jax.export` failures.

- [ ] **Step A.2.1: Write failing tests**

```python
# in tests/test_sprint_modelinputs.py
def test_prxtein_mpnn_has_forward_score_and_sample():
    import jax
    from prxteinmpnn.model.mpnn import PrxteinMPNN

    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)
    assert hasattr(m, "forward_score"), "PrxteinMPNN must have forward_score()"
    assert hasattr(m, "forward_sample"), "PrxteinMPNN must have forward_sample()"
```

- [ ] **Step A.2.2: Add `forward_score()` and `forward_sample()` to PrxteinMPNN**

```python
def forward_score(
    self,
    inputs: SamplingInputs,
    *,
    fns: PipelineFns,
) -> jax.Array:
    """Pytree-based scoring entry point. Returns logits only.

    Accepts SamplingInputs as a single pytree operand — suitable for
    jax.lax.switch on a single operand and jax.export.
    """
    # Unpack SamplingInputs, call the appropriate scoring branch.
    ...

def forward_sample(
    self,
    inputs: SamplingInputs,
    *,
    fns: PipelineFns,
) -> tuple[jax.Array, jax.Array]:
    """Pytree-based sampling entry point. Returns (sequences, logits).

    Accepts SamplingInputs as a single pytree operand.
    """
    # Unpack SamplingInputs, call the autoregressive sampling branch.
    ...
```

- [ ] **Step A.2.3: Verify parity — forward_score() matches __call__ scoring path**

```python
# in tests/test_sprint_modelinputs.py
def test_forward_score_parity_with_call():
    """forward_score() and __call__ must produce identical logits on same input."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.model_inputs import SamplingInputs, BackboneGeometry, ConditioningFeatures
    from prxteinmpnn.payloads import MultistateStackPayload, WaveParallelPayload
    from prxteinmpnn.pipeline_fns import PipelineFns
    from tests.pipeline.test_autoregressive import _make_stack_and_wave  # reuse fixture

    S, L, D, k = 2, 6, 16, 4
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(D, D, D, 1, 1, k, key=key)
    stack, wave = _make_stack_and_wave(S=S, L=L)
    backbone = BackboneGeometry(
        coords=jnp.zeros((S, L, 4, 3)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
    )
    # ConditioningFeatures fields: fixed_tokens (Int), bias (Float), ar_mask (Float)
    # (see src/prxteinmpnn/model_inputs.py:33-38)
    cond = ConditioningFeatures(
        fixed_tokens=jnp.zeros((S, L), dtype=jnp.int32),
        bias=jnp.zeros((S, L, 21)),
        ar_mask=jnp.zeros((S, L, L)),
    )
    inputs = SamplingInputs(backbone=backbone, state_stack=stack, wave_parallel=wave, conditioning=cond)
    fns = PipelineFns.default()

    logits_new = m.forward_score(inputs, fns=fns)
    # __call__ equivalent — pass same data via the positional interface
    logits_old = m.score_unconditional_from_payload(key, backbone, fns=fns)

    assert jnp.allclose(logits_new, logits_old, atol=1e-5), (
        "forward_score and __call__ scoring path must produce identical logits"
    )
```

- [ ] **Step A.2.4: Run tests**

```bash
uv run pytest tests/test_sprint_modelinputs.py -k "forward" -xvs
```

Expected: PASS.

- [ ] **Step A.2.5: Run full regression suite**

```bash
PYTHONPATH=src uv run pytest tests/ -q --tb=short
```

Expected: No new failures.

- [ ] **Step A.2.6: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py tests/test_sprint_modelinputs.py
git commit -m "feat(A.2): add PrxteinMPNN.forward_score and forward_sample additive entry points"
```

---

### Task A.3 — Add `LigandMPNN.forward_score` and `forward_sample`

**Files:**
- Modify: `src/prxteinmpnn/model/ligand_mpnn.py`

Apply the same additive pattern as A.2 for `LigandMPNN`. Embed `LigandSamplingInputs` and
extract the `ligand_stack` field to pass to the ligand encoder.

- [ ] **Step A.3.1: Write test and implement similarly to A.2**

Follow the same pattern: test for `hasattr(m, "forward_score")`, test parity with `__call__`,
commit.

- [ ] **Step A.3.2: Commit**

```bash
git add src/prxteinmpnn/model/ligand_mpnn.py tests/test_sprint_modelinputs.py
git commit -m "feat(A.3): add LigandMPNN.forward_score and forward_sample entry points"
```

---

## Sprint B — StableHLO export smoke test

---

### Task B.1 — Smoke-test jax.export on forward_score()

**Files:**
- Test: `tests/test_sprint_modelinputs.py`

- [ ] **Step B.1.1: Write jax.export smoke test**

```python
# in tests/test_sprint_modelinputs.py
import pytest

@pytest.mark.slow
def test_jax_export_sampling_inputs_smoke():
    """jax.export must accept forward_score() with SamplingInputs without raising."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.model_inputs import SamplingInputs, BackboneGeometry, ConditioningFeatures
    from prxteinmpnn.pipeline_fns import PipelineFns
    from tests.pipeline.test_autoregressive import _make_stack_and_wave

    S, L, D, k = 2, 6, 16, 4
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(D, D, D, 1, 1, k, key=key)
    fns = PipelineFns.default()

    stack, wave = _make_stack_and_wave(S=S, L=L)
    backbone = BackboneGeometry(
        coords=jnp.zeros((S, L, 4, 3)),
        mask=jnp.ones((S, L)),
        residue_index=jnp.arange(L, dtype=jnp.int32)[None].repeat(S, axis=0),
        chain_index=jnp.zeros((S, L), dtype=jnp.int32),
    )
    # ConditioningFeatures fields: fixed_tokens (Int), bias (Float), ar_mask (Float)
    cond = ConditioningFeatures(
        fixed_tokens=jnp.zeros((S, L), dtype=jnp.int32),
        bias=jnp.zeros((S, L, 21)),
        ar_mask=jnp.zeros((S, L, L)),
    )
    inputs = SamplingInputs(backbone=backbone, state_stack=stack, wave_parallel=wave, conditioning=cond)

    jitted = eqx.filter_jit(m.forward_score)
    exported = jax.export.export(jitted)(inputs, fns=fns)
    assert exported is not None
```

- [ ] **Step B.1.2: Run smoke test**

```bash
PYTHONPATH=src uv run pytest tests/test_sprint_modelinputs.py::test_jax_export_sampling_inputs_smoke -xvs -m slow
```

Expected: PASS (no XLA compilation error).

- [ ] **Step B.1.3: Commit**

```bash
git add tests/test_sprint_modelinputs.py
git commit -m "test(B.1): jax.export smoke test for PrxteinMPNN.forward_score(SamplingInputs)"
```

---

### Task B.2 — Remove remaining `Optional[Array]` from JIT-boundary methods

**Files:**
- Modify: `src/prxteinmpnn/model/mpnn.py`, `src/prxteinmpnn/model/ligand_mpnn.py`

- [ ] **Step B.2.1: Grep for Optional[Array] at JIT boundaries**

```bash
grep -n "Optional\[" \
  src/prxteinmpnn/model/mpnn.py \
  src/prxteinmpnn/model/ligand_mpnn.py
```

- [ ] **Step B.2.2: Remove or replace with concrete defaults**

For each `Optional[Array]` parameter in a JIT-boundary method (those decorated with
`@eqx.filter_jit` or called via `lax.switch`): replace with a concrete zero-array default
or a boolean sentinel, never `None`. Non-JIT-boundary helper methods may keep `Optional`.

- [ ] **Step B.2.3: Verify export still passes**

```bash
PYTHONPATH=src uv run pytest tests/test_sprint_modelinputs.py::test_jax_export_sampling_inputs_smoke -xvs -m slow
```

Expected: PASS.

- [ ] **Step B.2.4: Commit**

```bash
git add src/prxteinmpnn/model/mpnn.py src/prxteinmpnn/model/ligand_mpnn.py
git commit -m "refactor(B.2): remove Optional[Array] from JIT-boundary model methods"
```

---

## Final verification

- [ ] Run full test suite:

```bash
PYTHONPATH=src uv run pytest tests/ -q --tb=short
```

Expected: No failures (slow tests may be skipped with `-m "not slow"`).

- [ ] Verify ROADMAP.md items A–D are marked done; update `.agents/ROADMAP.md`.
