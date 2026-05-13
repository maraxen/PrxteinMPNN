# MPNN / LigandMPNN Duplication Reduction — Design Spec

**Date**: 2026-05-13
**Kind**: refactor (discovery-first; task count finalized after Fixer 1)
**Related**: `2026-05-13-naming-discipline-spec.md`, `2026-05-13-model-protocol-seam-spec.md`, `2026-05-13-model-public-contract-spec.md`

---

## Problem

`mpnn.py` (1406 LOC, `PrxteinMPNN`) and `ligand_mpnn.py` (1389 LOC, `PrxteinLigandMPNN`) implement structurally parallel models. Every change to a shared concept risks needing manual mirroring. The ~2800 combined LOC is misleading: classes are not symmetric copies, but they share extractable logic (~400-600 recoverable LOC). Without systematic extraction, the codebase accumulates silent behavioral divergence.

**Realistic extraction target: 400-600 LOC, not 1400.** The two `__call__` methods take fundamentally different signatures (ligand adds `Y, Y_t, Y_m`) and cannot be merged. Encoder and context-integration paths differ legitimately. Savings come from three places: `_call_unconditional` (~80 LOC, near-identical); ligand `_run_autoregressive_scan` inline (~305 LOC, asymmetric vs protein's external delegate); `__call__` `state_vmap_exact` validation block (~60 LOC, parameterized).

## Goal

Reduce maintainable surface by extracting verified-identical or parameterized-equivalent logic into shared free functions or a common base, leaving both classes as thin shells with **numerically identical behavior confirmed by `parity_heavy`**.

---

## Prerequisites (HARD REQUIREMENT)

**Do not start this spec until all three are at HEAD:**

1. **Naming-discipline rename** — The naming-discipline spec handles **file and inference-function** renames (`mpnn_autoregressive_state_vmap_exact.py` → `ar_exact.py`, `run_sample_autoregressive_state_vmap_exact` → `run_sample_ar_exact`, etc.) and the multistate underscore normalization. This spec (**dedup**) owns the **method rename** on the model classes themselves (`score_unconditional_from_payload` → cleaner name; `sample_autoregressive_state_vmap_exact_from_payload` → cleaner name). The split: naming-discipline = files + inference fns; dedup = method names on model classes. Both must be coordinated to avoid baking deprecated names into shared code, but the rename PRs do not overlap on the same files.

2. **Protocol-seam spec** — The layer above both model classes must depend only on `ModelProtocol`. Without this, refactoring the class hierarchy risks breaking callers. The dedup work is structurally unsafe without the seam. Note: the protocol-seam spec keeps the `_from_payload` names in its protocol definition; **after dedup renames these methods, the protocol seam spec's `ModelProtocol` definition must be updated in the same commit as the rename** (one Fixer task in this spec covers both).

3. **Public/internal boundary** — `model/__init__.py` must declare the public contract explicitly so the implementer knows which methods may be renamed and which require stable aliases.

**Starting early produces:** partial extraction conflicting with pending renames, three-way merge conflicts, broken callers not yet migrated to the protocol layer. Cost of waiting: one sprint. Cost of starting early: rework.

---

## Discovery Requirement

The primary deliverable of **Fixer 1** is a side-by-side comparison table. All subsequent tasks block on this.

### Method-by-method comparison table (format)

| Method | PrxteinMPNN | PrxteinLigandMPNN | Classification | Notes |
|---|---|---|---|---|
| `__init__` | 7 fields | 17 fields | DIVERGENT | Ligand adds 10 fields; A viable if subclass declares extras |
| `stage_schema` | FeaturizeFn / ProteinEncodeFn / ... | FeaturizeFn / LigandEncodeFn / ... | PARAMETERIZED | Differs only in "encode" alias |
| `__call__` | `(coords, mask, ri, ci, ...)` | `(coords, mask, ri, ci, Y, Y_t, Y_m, ...)` | DIVERGENT | Cannot merge |
| `_call_unconditional` | ~80 LOC at 209-289 | ~77 LOC at 617-694 | **IDENTICAL** | Diff: protein uses `self._apply_multistate_to_all_logits` delegate; ligand calls `_shared` directly. Normalize then extract. |
| `_call_conditional` | ~82 LOC at 291-373 | inline in `__call__` | DIVERGENT | Different dispatch topology |
| `_call_autoregressive` | ~91 LOC at 375-466 | inline; dispatches to `_run_autoregressive_scan` | DIVERGENT | Different dispatch |
| `_run_autoregressive_scan` | ~40 LOC shell delegating to `run_autoregressive_scan` | ~305 LOC **inline** | **ASYMMETRIC** | Primary debt. Extract ligand to external module first (strategy-independent) |
| `_process_group_positions` | ~100 LOC at 525-626 | inline inside ligand AR scan | ASYMMETRIC | Move to shared free function |
| `_combine_logits_multistate` | static delegate | not present (calls `_shared` directly) | PROTEIN-ONLY | Remove indirection |
| `_apply_multistate_to_all_logits` | static delegate | not present | PROTEIN-ONLY | Remove indirection |
| `_combine_logits_multistate_idx` | static delegate | not present | PROTEIN-ONLY | Remove indirection |
| `sample_autoregressive_state_vmap_exact` | ~24 LOC shell | ~52 LOC shell (extra y-args) | PARAMETERIZED | Same shape; differ in 3 ligand arrays |
| `*_from_payload` methods | all variants | all variants | DEPRECATED | Removed by naming-discipline spec; do not extract |
| `score_unconditional` | ~113 LOC inline | ~43 LOC shell delegating to external | **ASYMMETRIC** | Consider extracting protein path to `score_exact.py` |
| `score_conditional` | ~130 LOC inline | ~53 LOC shell delegating to external | **ASYMMETRIC** | Same asymmetry |

**Implementer must verify this table against HEAD before proceeding.** Differences override the table.

### Attribute comparison

| Field | PrxteinMPNN | PrxteinLigandMPNN | Strategy A implication |
|---|---|---|---|
| `features` | `ProteinFeatures` | `ProteinFeaturesLigand` | Different types; must redeclare in subclass |
| `encoder` | `Encoder \| PhysicsEncoder` | `Encoder` | Narrower type in ligand; compatible |
| `decoder`, `w_s_embed`, `w_out`, `node_features_dim`, `edge_features_dim`, `num_decoder_layers` | same | same | Inheritable |
| `capabilities` | default `PRXTEIN_MPNN_CAPABILITIES` | default `PRXTEIN_LIGAND_MPNN_CAPABILITIES` | Subclass must override default |
| `context_encoder`, `y_context_encoder` | absent | `tuple[DecoderLayer, ...]` | Ligand-only |
| `w_v`, `w_c`, `w_nodes_y`, `w_edges_y`, `v_c`, `v_c_norm`, `dropout` | absent | `eqx.nn.Linear` / `LayerNorm` / `Dropout` | Ligand-only |
| `hidden_features_dim`, `ligand_mpnn_use_side_chain_context` | absent | static field | Ligand-only |

**Equinox pytree leaf-order constraint**: field declaration order determines flattening. Reordering invalidates checkpointed weights indexed by leaf position. Verify with `io/weights.py` loading path before Strategy A.

---

## Strategy Evaluation Framework

Three strategies are viable. **Implementer chooses after completing the comparison table; writes a one-page rationale before extraction.**

### Strategy A — Inheritance: `PrxteinLigandMPNN(PrxteinMPNN)`

Subclass declares extra fields after parent's, inherits identical methods, overrides divergent ones. **Precedent**: `DiffusionPrxteinMPNN(PrxteinMPNN)` already works.

Reject if: weight-loading is leaf-order-sensitive, OR `__init__` PRNG-key splits differ structurally (ligand's `context_encoder` and `y_context_encoder` need extra keys).

Specific risk: `features` field type differs — subclass must explicitly redeclare `features: ProteinFeaturesLigand` to shadow parent type, or pytree registration uses parent annotation silently.

### Strategy B — Composition / Mixin: `_BaseMPNN` abstract

Extract `_BaseMPNN(eqx.Module)` holding shared fields/methods. Both classes compose it. Most boilerplate; with Equinox there's no real Python mixin — composition means embedding `_BaseMPNN` as a field with indirection.

Likely verdict: avoid unless A rejected and C insufficient.

### Strategy C — Functional Core: shared free functions

Dominant existing pattern. Extract logic to free functions in `_shared.py` or new modules. Both classes remain independent `eqx.Module` subclasses with no inheritance. Method bodies become thin shells.

Precedent: `mpnn_autoregressive_scan.py` and the rest of the inference family already follow this. Strategy C extends the norm.

Primary extractions under C:
1. `_call_unconditional` body → `_shared.call_unconditional_impl(model, ...)` — both classes call it
2. Ligand inline `_run_autoregressive_scan` body → `mpnn_autoregressive_scan_ligand.py` (mirrors protein's external delegate)
3. `_process_group_positions` equivalent → shared function

**JIT note**: Strategy C is safest for JIT trace-cache stability — free functions called by reference, same JAX function object across protein/ligand.

### Decision matrix

| Criterion | Weight | A | B | C |
|---|---|---|---|---|
| Pytree safety | HIGH | Risky | Safe | Safe |
| Weight-loading compat | HIGH | Risky | Safe | Safe |
| JIT cache stability | HIGH | Uncertain | Uncertain | Highest |
| LOC reduction | MEDIUM | 200-400 | 100-200 | 400-600 |
| Implementation complexity | MEDIUM | Low | High | Medium |
| Consistency with existing patterns | HIGH | Partial | No | Yes |

**Rationale must answer (1) weight-loading safety for A, (2) JIT trace-cache test result, (3) `features` redeclaration question for A. If any unanswered for A → default to C.**

---

## Migration Sequence Template

**Phase 0**: Verify prerequisites at HEAD. Run fast + parity baselines.

**Phase 1**: Fixer 1 produces comparison table + strategy rationale. No code changes.

**Phase 2**: Pre-step — normalize asymmetry. Extract ligand inline scan to `mpnn_autoregressive_scan_ligand.py`. Strategy-independent (required for A, B, and C). One atomic commit; fast tests gate.

**Phase 3+**: Strategy-specific extractions. One method per commit. Fast tests per commit; `parity_heavy` before trunk merge.

Each commit independently revertable. Max LOC per task: 400.

---

## API Stability Assertion

These imports must remain working:
```python
from prxteinmpnn.model import PrxteinMPNN, PrxteinLigandMPNN
from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
```

`DiffusionPrxteinMPNN(PrxteinMPNN)` must continue to work. If Strategy A makes `PrxteinLigandMPNN` a subclass of `PrxteinMPNN`, verify Diffusion still inherits correctly.

Public method surface (via `ModelProtocol`): `features`, `encoder`, `decoder`, `w_out`, `w_s_embed`, `capabilities`, `__call__`, `stage_schema`, `score_unconditional`, `score_conditional`, `sample_autoregressive_state_vmap_exact` (post-rename).

`*_from_payload` methods are deprecated; do not stabilize.

---

## Equinox-Specific Risks

| Risk | Mitigation |
|---|---|
| Pytree leaf order change | Before Strategy A: `jax.tree_util.tree_leaves(model)` count baseline; assert unchanged after refactor |
| `features` field type shadow | Explicitly redeclare `features: ProteinFeaturesLigand` in subclass; test `isinstance(model.features, ProteinFeaturesLigand)` |
| `capabilities` default override | Redeclare with `eqx.field(static=True, default=PRXTEIN_LIGAND_MPNN_CAPABILITIES)`; test `model.capabilities.is_ligand_model == True` |
| JIT trace cache invalidation | Compile-time smoke test before/after; if >15% slower investigate with `jax.make_jaxpr` |
| Equinox version drift | Pin Equinox version in `pyproject.toml`; do not upgrade mid-refactor |
| `inference=` parameter | Ligand `_run_autoregressive_scan` has `inference: bool` kwarg; protein doesn't. Shared free function must plumb it through |

---

## Fixer Tasks

### Fixer 1 — Side-by-Side Comparison Table (analysis only, 0 LOC)

Files: new `docs/superpowers/comparisons/2026-05-13-mpnn-ligand-method-table.md`

- Produce complete method comparison table from HEAD
- Produce attribute comparison
- Produce one-page strategy rationale addressing the three A-specific questions
- Default to C if A's questions unanswered

Gate:
```bash
test -f docs/superpowers/comparisons/2026-05-13-mpnn-ligand-method-table.md && \
  wc -l docs/superpowers/comparisons/2026-05-13-mpnn-ligand-method-table.md | awk '{if ($1 < 30) exit 1}'
```

**All subsequent tasks blocked on this approval.**

### Fixer 2 — Extract Ligand Autoregressive Scan (~305 LOC moved, strategy-independent)

Files: new `src/prxteinmpnn/model/ar_scan_ligand.py` (post-naming-discipline; if naming-discipline is not yet HEAD, use `mpnn_autoregressive_scan_ligand.py`); modify `ligand_mpnn.py`.

Create function `run_sample_ar_scan_ligand(model, prng_key, ...)` (post-naming form). Move `PrxteinLigandMPNN._run_autoregressive_scan` body verbatim; replace `self` with `model`. Replace ligand's method body with shell delegating to the new function. **Preserve `inference: bool` kwarg** — it is part of the ligand AR API and tests rely on it for dropout activation.

Gate (fast + targeted parity + dropout assertion):
```bash
# Fast suite
PYTHONPATH=prxteinmpnn/src uv run pytest \
  prxteinmpnn/tests/sampling/test_sample.py \
  prxteinmpnn/tests/model/test_ligand_wave_parallel.py \
  prxteinmpnn/tests/sampling/test_state_vmap_exact_jit.py \
  prxteinmpnn/tests/sampling/test_sample_call_kw_contract.py -q

# parity_heavy on ligand path — this is exactly where silent numerical drift would appear
export REFERENCE_PATH=/absolute/path/to/ligandmpnn_reference_assets
cd prxteinmpnn && PYTHONPATH=scripts:src uv run pytest \
  tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v

# inference= kwarg plumbing — confirm dropout activates under inference=False
PYTHONPATH=prxteinmpnn/src uv run python -c "
# Build a tiny ligand model, invoke the new run_sample_ar_scan_ligand twice
# with inference=False and check that outputs differ across PRNG keys
# (dropout is non-deterministic in training mode). If outputs are bit-identical,
# inference= was silently dropped.
# Spec leaves the exact assertion to the implementer; the requirement is that
# the gate includes evidence the kwarg made it through.
print('inference= plumb-through test: see implementer notes')
"
```

### Fixer 3 — Extract `_call_unconditional` to shared free function (~80 LOC)

Files: `_shared.py`, `mpnn.py`, `ligand_mpnn.py`.

Add `_shared.call_unconditional_impl(decoder, w_out, w_s_embed, node_features, edge_features, neighbor_indices, mask, prng_key, tie_group_map, multi_state_strategy_idx, state_weights, state_mapping) -> tuple[OneHotProteinSequence, Logits]`.

Take sub-modules explicitly (no `self`) to remain JAX-tracing-safe — matches existing convention in `mpnn_autoregressive_scan.py`.

Normalize: drop `self._apply_multistate_to_all_logits` delegate; call `apply_multistate_to_all_logits` from `_shared` directly. Both classes use the new function.

Gate: same fast tests.

### Fixer 4+ — Strategy-specific extractions

Filled in after Fixer 1 rationale. Each ≤400 LOC. Fast-test gate per commit. `parity_heavy` before trunk merge.

```bash
export REFERENCE_PATH=/absolute/path/to/ligandmpnn_reference_assets
cd prxteinmpnn && PYTHONPATH=scripts:src uv run pytest \
  tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v
```

Compile-time smoke test (run before and after each strategy task):
```bash
time PYTHONPATH=prxteinmpnn/src uv run python -c "
import jax, equinox as eqx, jax.numpy as jnp
from prxteinmpnn.model import PrxteinMPNN
m = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=jax.random.PRNGKey(0))
f = eqx.filter_jit(m.__call__)
coords = jnp.ones((10, 4, 3))
mask = jnp.ones((10,)); ri = jnp.arange(10); ci = jnp.zeros(10, int)
f(coords, mask, ri, ci, 'unconditional', prng_key=jax.random.PRNGKey(0)).block_until_ready()
"
```

If >15% slower post-refactor, investigate with `jax.make_jaxpr`.

---

## Risks

| Risk | Mitigation |
|---|---|
| `parity_heavy` regression on ligand path | Pre-extraction: save baseline output; post-extraction: assert `jnp.allclose` |
| Pytree leaf-order invalidates checkpoints (Strategy A) | Baseline leaf count; assert unchanged; add regression test |
| JIT recompilation regression | Compile-time smoke test before/after each task |
| `inference=` parameter dropped silently | Test both models with `inference=False`; verify dropout is active |
| Circular import after extraction | `_shared.py` adds `Decoder`, `eqx.nn.Linear` imports — verify with `python -c "from prxteinmpnn.model._shared import call_unconditional_impl"` |
| `DiffusionPrxteinMPNN` broken by Strategy A field reorder | Smoke-test `DiffusionPrxteinMPNN(128, 128, 128, 3, 3, 30, key=...)` after every Strategy A commit |

---

## Gates Summary

| Level | Command | When |
|---|---|---|
| Per-commit (fast) | `pytest test_sample test_ligand_wave_parallel test_state_vmap_exact_jit test_sample_call_kw_contract -q` | Every commit |
| Per-trunk-merge (parity) | `pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v` | Before merging branch |
| Compile-time regression | Time smoke test | Before/after each strategy task |
| Import stability | `python -c "from prxteinmpnn.model import PrxteinMPNN, PrxteinLigandMPNN; from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN"` | Every commit |
| Leaf-count regression (A only) | `assert len(tree_leaves(model))` unchanged | First Strategy A commit |

---

## Out of Scope

- `encoder.py`, `decoder.py` — no duplication; composed not duplicated
- `ProteinFeaturesLigand` vs `ProteinFeatures` — different by design
- `mpnn_autoregressive_state_vmap_exact*.py` — already external, already Strategy C
- `mpnn_scoring_state_vmap_exact_ligand.py` — already external
- Training loop — uses protocol layer, no changes
- `DiffusionPrxteinMPNN` — clean subclass, no changes
- `capabilities.py` constants — intentionally separate

---

## Why This Is Last

This refactor REQUIRES all three preceding specs at HEAD:

1. **Naming discipline** must land first: `_from_payload` methods use `MultistateStackPayload` in signatures; extracting before rename bakes deprecated names into shared code.

2. **Protocol seam** must land first: the refactor changes the class hierarchy. Callers must depend on `ModelProtocol`, not concrete classes, for the change to be safe.

3. **Public/internal boundary** must land first: the refactor moves methods between classes and into free functions. Without a declared public contract, the implementer cannot know which methods need stable aliases.

Starting early produces partial extraction that conflicts with pending renames, requires rework, and risks breaking callers not yet migrated to the protocol layer. **Cost of waiting: one sprint. Cost of starting early: three-way merge conflict + rework.**
