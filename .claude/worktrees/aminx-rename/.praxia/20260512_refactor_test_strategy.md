# Test Strategy: Tier 1/Tier 2 + StageSet Refactoring (7 Fixers)

## Cross-Fixer Fixture Architecture

Create **`tests/pipeline/conftest.py`** (new file) with these shared fixtures:

### registry_snapshot (function-scope, autouse)
```python
@pytest.fixture(autouse=True)
def registry_snapshot():
    """Snapshot pipeline_registry._REGISTRY before each test, restore after.
    Prevents cloudpickle-UID pollution from test ordering."""
    import aminx.pipeline_registry as _reg
    snap = dict(_reg._REGISTRY)
    yield
    _reg._REGISTRY.clear()
    _reg._REGISTRY.update(snap)
```

**⚠️ FIX #5:** This fixture must enumerate and snapshot:
- `pipeline_registry._REGISTRY` (main hook registry)
- `DEFAULT_FEATURIZE_UID`, `DEFAULT_ENCODE_UID`, `DEFAULT_DECODE_UID` (sentinel constants)
- All cloudpickle-hashed UID keys registered during test

### mini_mpnn_model and mini_ligand_model (session-scope)
Thin wrappers over existing pattern in `tests/model/test_ligand_wave_parallel.py:31-42`:
- Build `node_features=32, edge_features=32, hidden_features=32, num_encoder_layers=1, num_decoder_layers=1` models
- Fixed `PRNGKey(42)` for determinism
- Reuse `rng_key` and `model_inputs` from `tests/conftest.py`

### trace_counter (function-scope)
Returns `[0]` mutable list and wraps `jax.make_jaxpr`/`jax.jit` call count via side-effecting transform closure. Used by compile-counter tests.

---

## Fixer 1: Tier 1/Tier 2 Protocols + Types

**New file:** `tests/pipeline/test_tier_protocols.py`

| Test name | Kind | Input | Expected | Gate |
|---|---|---|---|---|
| `test_transformfn_protocol_runtime_checkable` | unit | `lambda x: x` | `isinstance(fn, TransformFn)` is `True` | `pytest tests/pipeline/test_tier_protocols.py::test_transformfn_protocol_runtime_checkable -q` |
| `test_rollingfn_protocol_runtime_checkable` | unit | stateful closure | `isinstance(fn, RollingFn)` is `True` | same file, `-k test_rollingfn` |
| `test_fusefn_protocol_runtime_checkable` | unit | combining closure | `isinstance(fn, FuseFn)` is `True` | same |
| `test_tier2_featurizefn_resolves_to_transformfn_shape` | unit | `FeaturizeFn` TypeAlias | `inspect.signature(FeaturizeFn.__call__)` matches `TransformFn.__call__` | same file |
| `test_tier2_proteinencodefn_resolves_correctly` | unit | `ProteinEncodeFn` | signature matches bound Tier 1 | same file |
| `test_ar_logit_transform_fn_distinguished_from_logit_transform_fn` | unit | wrong-shape fn | `isinstance` fails or `stage_schema()` raises `StageSchemaError` | same file |
| `test_logit_transform_fn_distinguished_from_ar_logit_transform_fn` | unit | `(S, V)`-returning passed as LogitTransformFn | same error gate | same file |
| `test_no_circular_import` | smoke | `import aminx.pipeline.tier1` | no ImportError | same file |
| `test_protocols_are_not_runtime_checkable_with_unbound_typevar` | edge case | Generic protocol with unbound TypeVar | `isinstance(x, GenericProto)` raises `TypeError` | same file |

**Note:** Mark all tests `@pytest.mark.xfail(strict=True, reason="Fixer 1 not yet landed")` until Fixer 1 PR merges. Remove marker in Fixer 6.

---

## Fixer 2: StageSet + Registry

**New file:** `tests/pipeline/test_stageset.py`

| Test name | Kind | Input | Expected | Gate |
|---|---|---|---|---|
| `test_stageset_default_instantiates` | smoke | `StageSet.default()` | no error; `isinstance(s, StageSet)` | `pytest tests/pipeline/test_stageset.py::test_stageset_default_instantiates -q` |
| `test_stageset_from_callables_idempotent` | unit | same fn registered twice | same UID both times; `_REGISTRY` unchanged | same file |
| `test_stageset_resolve_all_returns_correct_dict` | unit | `StageSet.default()` | `resolve_all()` returns dict with expected keys | same file |
| `test_stageset_sentinel_uids_resolve_to_none` | unit | sentinel UID constants | `resolve_all()['featurize_fn']` is `None` for unset stages | same file |
| `test_stageset_validate_for_raises_on_schema_mismatch` | unit | StageSet with wrong-typed fn | `StageSchemaError` raised | same file |
| `test_stageset_validate_for_passes_on_correct_schema` | unit | `StageSet.default()` | no error | same file |
| `test_pipelinefns_default_emits_deprecation_warning` | unit | `PipelineFns.default()` | `pytest.warns(DeprecationWarning)` | same file |
| `test_stageset_default_does_not_warn` | unit | `StageSet.default()` | no `DeprecationWarning` | same file |
| `test_stageset_default_does_not_re_trace_across_calls` | unit | jitted fn called twice with same `StageSet` | trace count = 1 (not 2) | same file |
| `test_registering_none_callable_raises` | edge case | `StageSet.from_callables(logit_transform=None)` | `TypeError` or `ValueError` with message naming slot | same file |

**HARD BLOCKER:** `test_stageset_default_does_not_re_trace_across_calls` — If StageSet stores callables in pytree instead of UIDs, JAX re-traces on every call. Do not merge until fixed.

---

## Fixer 3: Model.stage_schema()

**New file:** `tests/pipeline/test_stage_schema.py`

| Test name | Kind | Input | Expected | Gate |
|---|---|---|---|---|
| `test_prxtein_mpnn_has_stage_schema_method` | unit | `Aminx(...)` | `hasattr(model, 'stage_schema')` | `pytest tests/pipeline/test_stage_schema.py::test_prxtein_mpnn_has_stage_schema_method -q` |
| `test_ligand_mpnn_has_stage_schema_method` | unit | `PrxteinLigandMPNN(...)` | `hasattr(model, 'stage_schema')` | same file |
| `test_stage_schema_returns_mapping` | unit | `model.stage_schema()` | returns `dict` or `StageSchema` with named slots | same file |
| `test_stage_schema_mpnn_vs_ligandmpnn_differ` | unit | both model types | ligand schema has `encoder_state_fn` slot absent from base | same file |
| `test_stage_schema_slot_types_are_protocol_types` | unit | `model.stage_schema()` | each value is a `type` or `Protocol` subclass | same file |
| `test_stage_schema_is_frozen` | unit | `schema = model.stage_schema(); schema['logit_transform'] = 42` | raises `TypeError` (frozen) | same file |

---

## Fixer 4: Executor Validation

**New file:** `tests/pipeline/test_executor_validation.py`

| Test name | Kind | Input | Expected | Gate |
|---|---|---|---|---|
| `test_executor_with_default_stageset_no_error` | smoke | executor + `StageSet.default()` | runs to completion | `pytest tests/pipeline/test_executor_validation.py::test_executor_with_default_stageset_no_error -q` |
| `test_executor_with_custom_stage_fn` | integration | executor + custom logit transform | output numerically differs but completes | same file |
| `test_stage_schema_validation_error_on_wrong_type` | unit | pass `ARLogitTransformFn` into `logit_transform` slot | `StageSchemaError` raised | same file |
| `test_stage_schema_error_message_helpful` | unit | wrong type in any slot | error message names the slot and expected type | same file |
| `test_executor_handles_negative_one_sentinel_flat_rows` | unit | `state_flat_rows` with `-1` sentinel | logits at sentinel positions not aggregated | same file |
| `test_executor_rejects_none_model` | edge case | `executor(None, ...)` | `TypeError` with message naming `module` | same file |
| `test_executor_validation_does_not_retrace_jit` | unit | executor called twice with identical StageSet | trace count = 1 | same file |

**HARD BLOCKER:** `test_stage_schema_error_message_helpful` — If Fixer 4 validation rejects a custom stage without clear error, document in error message.

---

## Fixer 5: LigandMPNN encoder_state_fn Threading

**New file:** `tests/pipeline/test_ligandmpnn_encoder_state_fn.py`

| Test name | Kind | Input | Expected | Gate |
|---|---|---|---|---|
| `test_ligandmpnn_encoder_state_fn_identity_carry_matches_vmap` | unit | identity-carry `EncoderStateFn`, small ligand-free input | logits from scan path = vmap path (atol=1e-5) | `pytest tests/pipeline/test_ligandmpnn_encoder_state_fn.py::test_ligandmpnn_encoder_state_fn_identity_carry_matches_vmap -q` |
| `test_ligandmpnn_encoder_state_fn_none_uses_vmap_path` | unit | `encoder_state_fn=None` | no error; falls back to `jax.vmap` | same file |
| `test_ligandmpnn_encoder_state_fn_guard_blocks_non_none_before_wired` | edge case | non-None `encoder_state_fn` before threading complete | `NotImplementedError` or `StageSchemaError` with clear message | same file |
| `test_ligandmpnn_encoder_state_fn_carry_shape_fixed_at_trace` | unit | carry with wrong shape at second call | `jax.errors.TracerArrayConversionError` or shape-check error | same file |
| `test_ligandmpnn_encoder_state_fn_jit_compiles_once` | unit | jitted path called twice | trace count = 1 (carry structure fixed) | same file |
| `test_ligandmpnn_score_unconditional_with_encoder_state_fn` | integration | `PrxteinLigandMPNN.score_unconditional` + `EncoderStateFn` | output shape `(n_flat, 21)`, all finite | same file |

**HARD BLOCKER:** `test_ligandmpnn_encoder_state_fn_identity_carry_matches_vmap` — If LigandMPNN scan path diverges from vmap beyond atol=1e-5, this is semantic regression. Do not merge until root-caused.

---

## Fixer 6: Test Suite (StageSet Unit + Executor Integration)

**New file:** `tests/pipeline/test_integration.py`

This fixer **unmarks xfails from Fixers 1–5** and adds cross-fixer integration tests.

| Test name | Kind | Input | Expected | Gate |
|---|---|---|---|---|
| `test_stageset_roundtrip_with_unconditional_pipeline` | integration | `StageSet.default()` → `UnconditionalPipeline` → `Aminx` | logits shape `(L, 21)`, all finite | `pytest tests/pipeline/test_integration.py::test_stageset_roundtrip_with_unconditional_pipeline -q` |
| `test_stageset_roundtrip_with_autoregressive_pipeline` | integration | `StageSet.default()` → `AutoregressivePipeline` → `Aminx` | sequences `(L,)`, logits `(L, 21)`, all finite | same file |
| `test_custom_logit_transform_threaded_end_to_end` | integration | custom `LogitTransformFn` registered via StageSet | output differs from default; no error | same file |
| `test_ar_logit_transform_threaded_end_to_end` | integration | custom `ARLogitTransformFn` via StageSet | AR sampling uses it; output differs | same file |
| `test_all_fixer1_5_unit_tests_no_longer_xfail` | smoke | run prior xfail tests | all now pass | `pytest tests/pipeline/ -q --tb=short` |
| `test_pipeline_registry_isolation_per_test` | unit | two tests registering conflicting UIDs | no cross-test bleed with `registry_snapshot` | same file |

**HARD BLOCKER:** `test_all_fixer1_5_unit_tests_no_longer_xfail` — If Fixer 6 fails to unmark xfails, at least one prior fixer did not land correctly.

---

## Fixer 7: Parity Test Migration

Update existing parity tests to use `StageSet.default()` instead of `PipelineFns.default()`.

**Existing parity tests:**

| Test name | Kind | What to assert | Gate |
|---|---|---|---|
| `test_golden_parity` | parity | logits via `StageSet.default()` match baseline within atol=1e-5 | `pytest tests/parity/test_golden_parity.py -q` |
| `test_jax_pytorch_parity` | parity | same | `pytest tests/parity/test_jax_pytorch_parity.py -q` |
| `test_averaging_pipeline_parity` | parity | averaging pipeline scores unchanged | `pytest tests/parity/test_averaging_pipeline_parity.py -q` |
| `test_tied_multistate_apples_to_apples` | parity | multistate logits unchanged | `pytest tests/parity/test_tied_multistate_apples_to_apples.py -q` |
| `test_pipelinefns_deprecation_warning_at_parity_callsites` | unit | all parity tests now use `StageSet.default()`; `PipelineFns` version warns | `pytest tests/parity/ -q -W error::DeprecationWarning` |

**Heavy parity (needs REFERENCE_PATH):**
```bash
export REFERENCE_PATH=/absolute/path/to/ligandmpnn_reference_assets
cd aminx && PYTHONPATH=scripts:src uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v
```

**Tolerance contract:** `atol=1e-5` on logit values, `atol=0` on sampled token indices (deterministic).

**⚠️ FIX #6:** Pre-flight checklist must include parity baseline capture:
```bash
PYTHONPATH=aminx/src uv run pytest aminx/tests/parity/ -q 2>&1 | tee /tmp/parity_baseline.log
```
Before Fixer 7 merge: `pytest aminx/tests/parity/ -q 2>&1 | diff /tmp/parity_baseline.log -` (must be identical)

**HARD BLOCKER:** If any Fixer 7 parity test fails by >1e-4, this is score regression. Do not merge.

---

## Execution Order

```
Fast first (no JAX):
  Fixer 1 unit tests (protocol checks)
  Fixer 2 unit tests (registry, StageSet)
  Fixer 3 unit tests (stage_schema shape/type)
  Fixer 4 unit tests (validation + error messages)
  Fixer 5 unit tests (encoder_state_fn equivalence)

Then integration:
  Fixer 6 integration tests (cross-fixer wiring)

Parity last (slow, needs REFERENCE_PATH):
  Fixer 7 parity verification
```

---

## Hard Blockers Summary

1. `test_stageset_default_does_not_re_trace_across_calls` fails → StageSet stores callables in pytree instead of UIDs
2. `test_ligandmpnn_encoder_state_fn_identity_carry_matches_vmap` fails beyond atol=1e-5 → LigandMPNN scan path diverges numerically
3. Any Fixer 7 parity test fails by >1e-4 → Score regression
4. `test_stage_schema_error_message_helpful` fails → Error messages opaque
5. Fixer 6 fails to unmark xfails → Prior fixer did not land correctly
6. `DeprecationWarning` suppressed at parity call sites → PipelineFns shim incomplete

---

## Test File Locations

| File | Fixer | Purpose |
|---|---|---|
| `tests/pipeline/conftest.py` | 6 | `registry_snapshot`, `mini_mpnn_model`, `mini_ligand_model`, `trace_counter` fixtures |
| `tests/pipeline/test_tier_protocols.py` | 1 | Tier 1/2 protocol runtime-checkable tests |
| `tests/pipeline/test_stageset.py` | 2 | StageSet unit + compile-counter |
| `tests/pipeline/test_stage_schema.py` | 3 | Model.stage_schema() shape/type/frozen tests |
| `tests/pipeline/test_executor_validation.py` | 4 | Executor validation + error message quality |
| `tests/pipeline/test_ligandmpnn_encoder_state_fn.py` | 5 | LigandMPNN encoder_state_fn threading + equivalence |
| `tests/pipeline/test_integration.py` | 6 | Cross-fixer integration: StageSet → Pipeline → Model |

Reuse existing conftest fixtures (`model_inputs`, `rng_key`, `mock_model_parameters`, `apply_jit`) from `tests/conftest.py`.
