# Specification: Tier 1/Tier 2 Protocol Hierarchy + StageSet Refactoring

## Context & Oracle Recommendation

**Oracle chose Option A:** Extend PipelineFns into StageSet with UID fields for featurize/encode/decode. Reuse the existing registry+UID mechanism. Backwards compatible defaults.

**Scope from oracle:**
- Rename PipelineFns → StageSet, add featurize_uid, encode_uid, decode_uid
- Add register_featurize_fn, register_encode_fn, register_decode_fn to registry
- Refactor model methods (PrxteinMPNN.score_*_from_payload) to resolve stages
- Add Model.stage_schema() classmethod per variant
- Thread encoder_state_fn through LigandMPNN.score_conditional
- Update executors to validate stage_set against schema
- Move parity tests onto executor path
- ~6-8 files, 250-400 LOC

## Specification Requirements

### 1. Tier 1 Protocols (generic, reusable across projects)
- `TransformFn[In, Out]` — stateless transformation
- `RollingFn[Carry, In, Out]` — scan-body with init
- `FuseFn[PerItem, Combined]` — reduce-across-axis

### 2. Tier 2 Aliases (MPNN-specific type aliases)
- `FeaturizeFn = TransformFn[BackboneGeometry, FeaturizedGraph]`
- `EncoderStepFn = TransformFn[FeaturizedGraph, EncoderOutput]`
- `EncoderStateFn = RollingFn[EncoderCarry, BackboneGeometry, EncoderOutput]`
- `ConditionalDecodeFn`, `UnconditionalDecodeFn` (asymmetric, keep distinct)
- `LogitTransformFn = FuseFn[(S,L,V), (L,V)]`
- `ARLogitTransformFn = FuseFn[(S,V), (V)]`
- Per-variant: `ProteinEncodeFn`, `LigandEncodeFn`

### 3. StageSet dataclass (replaces PipelineFns)
```python
@dataclass(frozen=True)
class StageSet:
    featurize_uid: str = DEFAULT_FEATURIZE_UID
    encode_uid: str = DEFAULT_ENCODE_UID
    decode_uid: str = DEFAULT_DECODE_UID
    logit_transform_uid: str = DEFAULT_LOGIT_TRANSFORM_UID
    ar_logit_transform_uid: str = DEFAULT_AR_LOGIT_TRANSFORM_UID
    encoder_state_fn_uid: str | None = None

    @classmethod
    def from_callables(featurize=None, encode=None, decode=None, ...) -> StageSet
    
    @classmethod
    def default() -> StageSet
    
    def resolve_all(self) -> dict[str, Callable]
```

### 4. Model.stage_schema() classmethod
```python
@classmethod
def stage_schema(cls) -> dict[str, type]:
    """Returns {stage_name: type_alias} for this model variant."""
    return {
        'featurize': FeaturizeFn,
        'encode': ProteinEncodeFn,  # or LigandEncodeFn for ligand model
        'decode': ConditionalDecodeFn | UnconditionalDecodeFn,
        'logit_transform': LogitTransformFn,
        'ar_logit_transform': ARLogitTransformFn,
        'encoder_state_fn': EncoderStateFn | None,
    }
```

### 5. Model method refactoring (PrxteinMPNN + PrxteinLigandMPNN)
- `score_unconditional_from_payload(payload, stage_set: StageSet, ...)`
- `score_conditional_from_payload(payload, stage_set: StageSet, ...)`
- `sample_autoregressive_from_payload(payload, stage_set: StageSet, ...)`
- Each resolves featurize/encode/decode from stage_set UIDs at host time
- Defaults to current behavior (backwards compatible)

### 6. Executor updates (UnconditionalExecutor, ConditionalExecutor, AutoregressiveExecutor)
- Signature: `__call__(module, key, inputs, stage_set: StageSet, **kwargs)`
- Host-time validation: `stage_set.validate_for(module.stage_schema())`
- Raises `StageSchemaError` if mismatch
- Keep multistate routing in executor (do not push into stages)

### 7. Registry extensions
- `register_featurize_fn(featurize: FeaturizeFn) -> str (UID)`
- `register_encode_fn(encode: EncoderStepFn | EncoderStateFn) -> str`
- `register_decode_fn(decode: ConditionalDecodeFn | UnconditionalDecodeFn) -> str`
- Reuse cloudpickle-hash UID mechanism; idempotent

### 8. Deprecations
- `EncoderPreFn`, `EncoderPostFn` → removed from protocols.py
- `PipelineFns` → deprecated alias for StageSet (one release)
- 2026-05-08 plan → marked SUPERSEDED in docs

### 9. Backwards compatibility

**Public API preservation guarantee:** All existing `PipelineFns` public methods are preserved on `StageSet` with identical signatures and return types: `default()`, `from_callables()`, `resolve_logit_transform()`, `resolve_ar_logit_transform()`, `resolve_encoder_state_fn()`. No breaking changes to the public API in this release.

**Implementation pattern:**
- `PipelineFns = StageSet (alias, emits DeprecationWarning)`
- Old calls like `UnconditionalPipeline(..., fns=PipelineFns(...))` still work
- New calls like `UnconditionalExecutor(..., stage_set=StageSet(...))`
- One release overlap; hard-break in next sprint

### 10. Test surface
- Unit tests: `stage_schema()`, `validate_for()`, `resolve_all()`
- Integration: parity tests via executor + `StageSet.default()`
- Smoke test: one direct-method call per variant (regression)
- Custom stage tests: wrap encoder_state_fn, verify executor routes correctly

## Constraints & Non-goals

**Constraints:**
- Maintain JIT safety (UIDs in static_argnames, not Callables)
- No Optional[Array] on stage inputs/outputs (resolve on host)
- Carry pytrees have fixed structure at trace time
- Equinox pytree semantics for SamplingInputs/payloads
- Multistate routing stays in executor, not in stages

**Non-goals:**
- Do not introspect payloads to auto-select stages
- Do not make Executor accept raw callable dict (must be registered UIDs)
- Do not change Equinox integration; work within its pytree model
- Do not move stage sequence into a declarative config file

## Deliverable

1. **Protocol definitions** (Tier 1: TransformFn, RollingFn, FuseFn)
2. **Type aliases** (Tier 2: per-variant FeaturizeFn, EncoderStepFn, EncoderStateFn, DecodeFn, etc.)
3. **StageSet dataclass** with `from_callables()`, `default()`, `resolve_all()`, `validate_for()`
4. **Model.stage_schema()** signature per variant
5. **Model method refactoring** logic (where UIDs resolve, what defaults to)
6. **Executor validation** logic (how stage_set is validated at host time)
7. **Registry extensions** (register_featurize_fn, register_encode_fn, register_decode_fn)
8. **Deprecation strategy** (PipelineFns alias, warning, one-release overlap)
9. **Test strategy** (what to verify at each layer)
10. **Migration guide** for users and for parity tests

This is a precise, unambiguous specification for fixer implementation.
