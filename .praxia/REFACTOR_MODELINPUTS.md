# ModelInputs / ModelStaticConfig Refactor Plan

**Goal:** Collapse 30-45 positional argument signatures into a clean `ModelInputs`
(pytree dataclass) + `ModelStaticConfig` (frozen hashable) hierarchy that mirrors
the existing `RunSpec` hierarchy and supports StableHLO export.

**Oracle verdict:** APPROVED. Outside-in migration, Protocol for dispatch interface.

**Inference model:** Tied inference — each state encodes and decodes independently
(vmap over states), then a registered batch post-processing step transforms logits
across the state set. No flat supergraph. Memory scales linearly with state count,
not quadratically. The "flat" multistate path is deprecated in favor of this.

**Updated 2026-05-07:** Expanded to include pluggable decode fn registry, richer
hierarchical groupings (state → chain → residue → kNN), and state identity metadata.

---

## Inference Model

```
For each state s independently:
    encode(backbone_s)  →  node_features_s, edge_features_s   # vmap over states
    decode(node_features_s, ...)  →  logits_s  # (L_s, vocab)

Post-process:
    registered_batch_fn(logits_stack, state_metadata, ...) → combined_logits
    # operates on (n_states, L, vocab) or equivalent
    # supports rolling ops, weighted combination, arbitrary transforms
```

The `registered_batch_fn` is a Python callable registered with `RunSpec`, resolved
on the host before JIT, and passed as a static argument. It is NOT stored in the JAX
pytree — only its UID (a str) lives in `ModelStaticConfig`.

---

## Proposed Type Hierarchy

```
RunSpec (host-side, already exists)
├── SamplingSpecification
│   └── decode_fn: Callable | None  ← NEW: fn arg registered locally with UID
└── ScoringSpecification

DecodeFnRegistry (host-side ONLY — never enters JAX)
├── uid: str               ← hash of cloudpickle bytes (deterministic)
├── name: str              ← user-provided label
├── fn: Callable           ← the actual callable, passed as static arg to JIT
├── cloudpickle_bytes: bytes  ← for reproducibility / re-hydration
└── env_trace: dict        ← python version, jax version, package versions

ModelStaticConfig (NEW — frozen hashable, static_argnames at JIT)
├── SamplingStaticConfig
│   └── decode_fn_uid: str        ← registry key, resolved on host
│       n_samples: int
│       temperature: float
│       multistate_mode: Literal["tied", "independent"]
│       max_group_size: int
└── ScoringStaticConfig
    └── pass_mode: Literal[...]
        ar_mask_is_eye: bool

ModelInputs (NEW — pytree dataclass, NO Optional[Array])
├── SamplingInputs
│   ├── backbone: BackboneGeometry    # single-state geometry
│   │   ├── coords: Float[Array, "L 4 3"]
│   │   ├── mask: Float[Array, "L"]
│   │   ├── residue_index: Int[Array, "L"]
│   │   └── chain_index: Int[Array, "L"]
│   ├── state_stack: MultistateStackPayload   # (n_states, n_pad, ...) stacked
│   │   ├── coords_stack, mask_stack, residue_index_stack, chain_index_stack
│   │   ├── tie_group_map_stack
│   │   ├── fixed_mask_stack, fixed_tokens_stack
│   │   ├── state_flat_rows, flat_row_offsets
│   │   ├── state_index: Int[Array, "n_states"]   # NEW: per-state ordinal identity
│   │   ├── state_embedding: Float[Array, "n_states D"] | None → resolved to zeros on host
│   │   └── n_states, n_canonical, n_flat (static)
│   ├── wave_parallel: WaveParallelPayload    # decode schedule for tied AR
│   │   ├── wave_group_ids: Int[Array, "n_waves max_wave_size"]
│   │   ├── wave_group_positions: Int[Array, "n_waves max_wave_size max_group_size"]
│   │   ├── wave_group_valid: Bool[Array, "n_waves max_wave_size"]
│   │   └── wave_position_valid: Bool[Array, "n_waves max_wave_size max_group_size"]
│   └── conditioning: ConditioningFeatures
│       ├── fixed_tokens: Int[Array, "L"]
│       ├── bias: Float[Array, "L vocab"]
│       └── ar_mask: Float[Array, "L L"]
└── ScoringInputs
    ├── backbone: BackboneGeometry
    └── sequences: Int[Array, "n_seqs L"]

Hierarchical grouping metadata (passed through SamplingInputs or separately):
    State level:    state_stack.state_index, state_stack.state_embedding
    Chain level:    backbone.chain_index
    Residue level:  backbone.residue_index, tie_group_map_stack
    kNN level:      WaveParallelPayload (per-wave decode schedule)
    Features/state: state_embedding (learnable or one-hot state ID)
```

---

## Critical Design Rules

1. **NO `Optional[Array]` on `ModelInputs`** — None changes pytree structure,
   causes silent re-traces. Resolve all None on the host adapter before JIT.
   Use `state_embedding = jnp.zeros(...)` when no embedding is provided.

2. **Verify StableHLO pytree support first** — `jax.export` supports pytrees.
   Smoke test: `jax.export.export(jit_fn)(example_SamplingInputs)`.

3. **Outside-in migration order** — define types → host adapter → parity test →
   push one layer deeper per PR → flip JIT boundary last.

4. **One registration mechanism** — `eqx.Module` throughout (matches existing
   `MultistateStackPayload`, `LigandStack`, `SamplingControls` pattern).

5. **Compile-counter regression test** — add before any JIT boundary migration.

6. **DecodeFnRegistry is host-only** — cloudpickle, env trace, UID lookup all
   happen before JIT. The resolved `fn` becomes a `static_argnums` arg. Nothing
   from the registry enters `jax.tree_util.tree_leaves`.

7. **Tied inference only** — no flat supergraph. Each state encodes/decodes
   independently via `jax.vmap`. The registered `batch_fn` post-processes the
   full `(n_states, L, vocab)` logits tensor.

8. **Heterogeneous state counts** — different canonical positions may participate
   in different numbers of states. Use `state_flat_rows[s, i] = -1` sentinel for
   missing (state, position) pairs. Builders must not allocate flat rows for
   missing positions. Logit combination must only aggregate valid states per position.

---

## Migration Order (5 PRs, revised)

### PR-1: Define types (no call-site changes)

- Add `WaveParallelPayload` to `payloads.py`
- Extend `MultistateStackPayload` with `state_index` and `state_embedding`
- Add `BackboneGeometry`, `ConditioningFeatures` eqx.Modules to `model_inputs.py` (NEW file)
- Add `SamplingInputs`, `ScoringInputs` to `model_inputs.py`
- Add `SamplingStaticConfig`, `ScoringStaticConfig` to `model_inputs.py`
- Add `DecodeFnRegistry` dataclass to `run/decode_registry.py` (NEW file, host-only)
- Add pytree-structure + leaf-count unit tests
- Add compile-counter baseline test

### PR-2: Host adapter layer

- Extend `_coerce_loose_to_multistate_stack_host` to build `WaveParallelPayload`,
  fill `state_index` (ordinal per state), and resolve `state_embedding = zeros`
- Fix heterogeneous state count handling: builders must set `state_flat_rows[s, i] = -1`
  for positions not present in state s, and compute `flat_row_offsets` from valid counts only
- New outer wrapper: `make_sampling_inputs_from_spec(spec) -> SamplingInputs`
- New: `make_static_config_from_spec(spec) -> SamplingStaticConfig`
- Register `decode_fn` from `SamplingSpecification` into `DecodeFnRegistry`
- Parity tests: `tests/sampling/test_sample.py`, `tests/sampling/test_state_vmap_exact_jit.py`

### PR-3: Push to `_sample_sequences_jitted` boundary

- `_sample_sequences_jitted(model, key, inputs: SamplingInputs, config: SamplingStaticConfig,
  batch_fn: Callable)` — `batch_fn` is static, resolved from registry on host
- Unpack inside the function (temporary, preserves inner call shapes)
- Eliminate `**kwargs: Any` at sample.py:227, 357, 610
- Replace hardcoded `state_vmap_exact` branch with `batch_fn(logits_stack, ...)` call
- Parity gate

### PR-4: Push to model.__call__ boundary

- `PrxteinMPNN.__call__(self, inputs: SamplingInputs) -> (OneHot, Logits)`
- All three branch methods accept `SamplingInputs` (Protocol interface)
- `jax.lax.switch` at mpnn.py:970 becomes single pytree operand
- Parity gate

### PR-5: StableHLO export verification + cleanup

- Smoke-test `jax.export.export` with `SamplingInputs`
- Delete FBT `*`-marker fixes (bool args gone from JIT boundary)
- Delete remaining `Optional[Array]` params
- Update external scripts (run_design_grid.py, run_unconditional_logits_grid.py)

---

## Files to Touch

| File | Change |
|------|--------|
| `src/prxteinmpnn/payloads.py` | Add `WaveParallelPayload`; extend `MultistateStackPayload` with `state_index`, `state_embedding` |
| `src/prxteinmpnn/model_inputs.py` | NEW — `BackboneGeometry`, `ConditioningFeatures`, `SamplingInputs`, `ScoringInputs`, `SamplingStaticConfig`, `ScoringStaticConfig` |
| `src/prxteinmpnn/run/decode_registry.py` | NEW — `DecodeFnRegistry` (host-only) |
| `src/prxteinmpnn/sampling/state_vmap_prep.py` | Fix heterogeneous state count in builders; add `per_state_canonical_mask` param |
| `src/prxteinmpnn/sampling/sample.py` | Extend coercion adapter; migrate `_sample_sequences_jitted`; wire `batch_fn` |
| `src/prxteinmpnn/model/mpnn.py` | Migrate `__call__`, `_call_unconditional`, `_call_conditional` |
| `src/prxteinmpnn/model/ligand_mpnn.py` | Same pattern |
| `src/prxteinmpnn/run/scoring.py` | Migrate `score_single_pair` |
| `scripts/run_design_grid.py` | Update caller (PR-5) |
| `scripts/run_unconditional_logits_grid.py` | Update caller (PR-5) |

---

## Decode Function Registry

Each `SamplingSpecification` accepts an optional `decode_fn: Callable`. When
provided, it is registered on first use:

```python
uid = hashlib.sha256(cloudpickle.dumps(fn)).hexdigest()[:16]
DECODE_FN_REGISTRY[uid] = DecodeFnRegistry(
    uid=uid,
    name=user_provided_name or fn.__name__,
    fn=fn,
    cloudpickle_bytes=cloudpickle.dumps(fn),
    env_trace={"python": sys.version, "jax": jax.__version__, ...},
)
```

The `uid` is stored in `SamplingStaticConfig.decode_fn_uid`. Before JIT dispatch,
the host resolves `uid → fn` and passes `fn` as a `static_argnums` argument.

When `decode_fn` is None, the default tied-AR batch fn is used (the existing
`apply_multistate_to_all_logits` path, renamed and made pluggable).

---

## Hierarchical Groupings

Groupings are purely metadata in `SamplingInputs` — they are not baked into
the model. The registered `batch_fn` uses them however it chooses:

| Level | Carrier | What it enables |
|-------|---------|----------------|
| State | `state_stack.state_index`, `state_stack.state_embedding` | State-aware combination, ordered rolling ops across states |
| Chain | `backbone.chain_index` | Chain-conditional sampling, cross-chain coupling |
| Residue | `backbone.residue_index`, `tie_group_map_stack` | Tied decoding across states, position identity |
| kNN | `WaveParallelPayload` | Wave-parallel AR decode schedule |
| Features | `state_stack.state_embedding` | Learned or one-hot state conditioning |

Grouping *strategies* (functions that produce a `WaveParallelPayload` from geometry)
are registrable host-side functions — same registry pattern as decode fns.

---

## lax.switch Compatibility

Current: `jax.lax.switch(branch_index, branches, *operands)` at mpnn.py:970
After PR-4: `jax.lax.switch(branch_index, branches, inputs)` — single pytree.
All branches must accept identical pytree structure. `eqx.Module` with no
`Optional[Array]` fields guarantees structure equality.

---

## Revert Note

FBT `*`-marker fixes (commit b1e37dc): do NOT revert — benign for encode/decode
layer `inference` flags. The `__call__`-level bools (multistate_mode, etc.) move to
`SamplingStaticConfig` in PR-4, eliminating them naturally.
