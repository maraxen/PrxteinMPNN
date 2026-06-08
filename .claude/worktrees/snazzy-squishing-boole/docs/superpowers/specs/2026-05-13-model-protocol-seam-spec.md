# Model-Protocol Seam — Design Spec

**Date**: 2026-05-13
**Kind**: wiring (8 fixer tasks)
**Related**: `2026-05-13-naming-discipline-spec.md`, `2026-05-13-mpnn-dedup-spec.md` (this is a prerequisite for that)

---

## Problem

`sampling/sample.py`, `sampling/conditional_logits.py`, `sampling/unconditional_logits.py`, `sampling/state_vmap_payload_logits.py`, and `scoring/score.py` all import concrete model classes (`PrxteinMPNN`, `PrxteinLigandMPNN`) directly and branch on `isinstance` checks. Every model variant must be known at the sampling/scoring layer, making it impossible to introduce a new variant without editing files outside `model/`. This is also the structural reason `mpnn.py` and `ligand_mpnn.py` each approach 1400 LOC with 80%+ overlap.

## Goal

Replace all concrete-class imports and `isinstance` checks in `sampling/` and `scoring/` with a tightened `ModelProtocol` so those layers depend on a contract, not a class.

---

## 1. Existing-Protocols Audit

| Name | Purpose | Used where | Decision |
|---|---|---|---|
| `ConditionalLogitsFn` | Return type for `make_conditional_logits_fn` | `sampling/conditional_logits.py` | Keep |
| `UnconditionalLogitsFn` | Return type for `make_unconditional_logits_fn` | `sampling/unconditional_logits.py` | Keep |
| `StateVmapExactLogitsFn` | Wide-open factory for stacked multistate logits | `sampling/*` | Keep — escape hatch |
| `SamplerFn` | Sampler closure signature | `sampling/sample.py` | Keep |
| `ScoreFn` | Flat-graph scorer signature | `scoring/score.py` | Keep |
| `StateVmapExactScoreFn` | Stacked multistate scorer | `scoring/score.py` | Keep |
| `DesignSink` | Streaming I/O sink | `io/designs.py` | Keep — unrelated |
| `EncoderPreFn` | Marked deprecated; unused | nowhere | Flag for removal (separate sprint) |
| `EncoderPostFn` | Marked deprecated; unused | nowhere | Flag for removal (separate sprint) |
| `ModelProtocol` | Structural protocol over sub-modules; all `Any` | `Pipeline` references it | **Extend** — tighten types + add method signatures |
| `Pipeline` | Pipeline callable taking `ModelProtocol` | `pipeline/*.py` | Keep unchanged — benefits automatically |

---

## 2. Design Decision: Method Names

This spec uses **current method names** (`score_unconditional_from_payload`, etc.). The naming-discipline spec does not rename these public `_from_payload` methods — that rename is deferred to the dedup spec. The protocol seam is shippable and testable independently.

---

## 3. Design Decision: Ligand Overload Resolution

Current `_from_payload` signatures are structurally incompatible:

```python
# PrxteinMPNN
def score_unconditional_from_payload(self, prng_key, stack, *, tie_group_map, ...): ...

# PrxteinLigandMPNN — extra positional arg
def score_unconditional_from_payload(self, prng_key, stack, ligand, *, ...): ...
```

**Resolution: make `ligand` keyword-only on `PrxteinLigandMPNN`.**

**WARNING — call sites are NOT all keyword today.** The following sites currently pass `ligand` positionally and must be migrated to keyword in the same commit that changes the signature (else `ValueError: too many positional args` at runtime):

- `src/prxteinmpnn/sampling/state_vmap_payload_logits.py:53-62` — positional `ligand` (3rd arg, after `prng_key, stack`)
- `src/prxteinmpnn/sampling/state_vmap_payload_logits.py:110-122` — positional `ligand`
- `src/prxteinmpnn/sampling/sample.py:612-616` — passes `ls` (a `LigandStack`) positionally

Fixer 2 must update these three sites in the same atomic commit as the signature change. Same change applies to `score_conditional_from_payload` and `sample_autoregressive_state_vmap_exact_from_payload`.

Unified signature:
```python
def score_unconditional_from_payload(
    self, prng_key, stack: MultistateStackPayload,
    *, ligand: LigandStack | None = None, tie_group_map, ...,
) -> Logits: ...
```

`PrxteinLigandMPNN` raises `TypeError` if `ligand is None` (preserving existing guard).

---

## 4. Design Decision: `isinstance` → `capabilities`

All five `isinstance` sites convert to `model.capabilities.is_ligand_model`. `ModelCapabilities` already exists (`src/prxteinmpnn/model/capabilities.py`) as a static field on both classes with `PRXTEIN_MPNN_CAPABILITIES` and `PRXTEIN_LIGAND_MPNN_CAPABILITIES` constants. No new flags needed.

---

## 5. Design Decision: Static-Method Calls

Two files call `PrxteinMPNN._apply_multistate_to_all_logits` and `PrxteinMPNN._combine_logits_multistate_idx` directly on the class (with `# noqa: SLF001`). Both functions already exist as module-level exports in `model/_shared.py` as `apply_multistate_to_all_logits` and `combine_logits_multistate_idx`. Migration: import the module functions directly and call them — drop the `_shared`-via-class indirection. These functions do NOT belong on `ModelProtocol` (they are logit utilities, not model methods).

---

## 6. Target Protocol Design

Extend `ModelProtocol` in `protocols.py`. Current definition declares only attribute names with `Any`. Extension: (a) tightened attribute types, (b) three `_from_payload` method signatures, (c) `stage_schema` classmethod.

```python
@runtime_checkable
class ModelProtocol(Protocol):
    """Structural protocol over prxteinmpnn model modules.

    Concrete implementations: PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN.

    Equinox compatibility: all concrete implementations are eqx.Module subclasses
    and are valid JAX pytrees. Protocol satisfaction is structural — no ABC
    registration. jit/vmap/scan work unchanged because they trace through
    eqx.Module leaves, not through the protocol.

    Do NOT add __abstractmethods__ or ABCMeta — that breaks eqx.Module.

    The _from_payload methods form the stable seam: sampling and scoring layers
    must call only these, never raw stack-kwargs or concrete-class methods.
    """

    features: Any          # ProteinFeatures | LigandFeatures — kept Any to avoid circular import
    encoder: Any           # Encoder | PhysicsEncoder | LigandEncoder
    decoder: Any           # Decoder | ConditionalProteinDecoder
    w_out: Any             # eqx.nn.Linear
    w_s_embed: Any         # eqx.nn.Embedding
    capabilities: ModelCapabilities  # TIGHTENED

    # NOTE: __call__ is intentionally NOT part of the protocol.
    # Its signature differs fundamentally between protein (coords, mask, ri, ci, ...)
    # and ligand (coords, mask, ri, ci, Y, Y_t, Y_m, ...). Callers that need the raw
    # forward pass should depend on the concrete class; callers using the seam should
    # go through the *_from_payload methods declared below.

    @classmethod
    def stage_schema(cls) -> dict[str, type | None]: ...

    def score_unconditional_from_payload(
        self, prng_key: PRNGKeyArray, stack: MultistateStackPayload,
        *, ligand: LigandStack | None = None,
        tie_group_map: TieGroupMap | None, multi_state_strategy_idx: Int,
        state_weights: jnp.ndarray | None, state_mapping: jnp.ndarray | None,
        inference: bool = True,
        logit_transform_fn: LogitTransformFn | None = None,
        encoder_state_fn: EncoderStateFn | None = None,
    ) -> Logits: ...

    def score_conditional_from_payload(
        self, prng_key, stack: MultistateStackPayload,
        seq_oh_stack: jax.Array, ar_mask_stack: jax.Array,
        *, ligand: LigandStack | None = None,
        tie_group_map, multi_state_strategy_idx,
        state_weights, state_mapping,
        bias_flat: jax.Array | None = None,
        inference: bool = True,
        logit_transform_fn: LogitTransformFn | None = None,
        encoder_state_fn: EncoderStateFn | None = None,
    ) -> Logits: ...

    def sample_autoregressive_state_vmap_exact_from_payload(
        self, prng_key, stack: MultistateStackPayload,
        autoregressive_mask_stack: jax.Array, bias_stack: jax.Array,
        temperature: float, multi_state_strategy_idx,
        state_weights, wave_group_ids_local, wave_group_positions_local,
        wave_group_valid_local, wave_position_valid_local,
        *, ligand: LigandStack | None = None,
        ar_logit_transform_fn: ARLogitTransformFn | None = None,
    ) -> tuple[OneHotProteinSequence, Logits]: ...
```

**Import additions to `protocols.py`** (in `TYPE_CHECKING` block):
- `TieGroupMap`, `Int`, `OneHotProteinSequence` from `utils.types`
- `MultistateStackPayload`, `LigandStack` from `payloads`
- `ModelCapabilities` runtime import (safe — leaf module)

---

## 7. Call-Site Inventory

### Runtime imports (must change)

| File | Current | Change |
|---|---|---|
| `sampling/sample.py:11` | imports `PrxteinLigandMPNN, PrxteinMPNN` | annotate `model: ModelProtocol`; replace `cast("PrxteinMPNN", ...)` line 317 with `cast("ModelProtocol", ...)` |
| `sampling/conditional_logits.py:33-34` | two model imports | replace with `from prxteinmpnn.protocols import ModelProtocol`; line 182 `isinstance(model, PrxteinLigandMPNN)` → `model.capabilities.is_ligand_model`; line 243 guard adjusted |
| `sampling/unconditional_logits.py:32-33` | same pair | same pattern; lines 115, 164 |
| `sampling/state_vmap_payload_logits.py:19-20` | same pair | lines 46, 103 `isinstance` → capability check |
| `scoring/score.py:12-13` | same pair | line 91 isinstance; line 67 `PrxteinMPNN._apply_multistate_to_all_logits` → `apply_multistate_to_all_logits` (from `model._shared`); remove SLF001 |
| `run/averaging.py:12` | `PrxteinMPNN` | annotations + line 339 `PrxteinMPNN._combine_logits_multistate_idx` → `combine_logits_multistate_idx` (from `model._shared`); remove SLF001 |
| `run/conformational_inference.py:22` | `PrxteinMPNN` | annotate lines 62, 84, 147, 171 → `ModelProtocol` |

### TYPE_CHECKING imports

`sampling/__init__.py:22`, `sampling/ste_optimize.py:31`, `run/sampling.py:48`, `run/scoring.py:30`, `run/_dispatcher.py:11` — change annotations to `ModelProtocol`.

### Unchanged

`io/weights.py` (factory must return concrete types), `training/*` (gradient computation needs concrete class), `parity/*` (reference parity tests).

---

## 8. Fixer Tasks

### Fixer 1 — Extend `ModelProtocol` (~55 LOC)

`src/prxteinmpnn/protocols.py`: add ModelCapabilities runtime import; add TYPE_CHECKING imports; replace `ModelProtocol` body with §6 definition.

Gate:
```bash
PYTHONPATH=src uv run python -c "
from prxteinmpnn.protocols import ModelProtocol
from prxteinmpnn.model import PrxteinMPNN
import jax
m = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=jax.random.PRNGKey(0))
assert isinstance(m, ModelProtocol)
print('Fixer 1 gate passed')
"
```

### Fixer 2 — Unify `PrxteinLigandMPNN` `_from_payload` signatures + migrate positional call sites (~50 LOC, single atomic commit)

This task has TWO sub-changes that MUST land in the same commit (the signature change without the call-site updates produces a runtime break):

**A) `src/prxteinmpnn/model/ligand_mpnn.py`**: change `ligand: LigandStack` positional to `*, ligand: LigandStack | None = None` on three methods (`score_unconditional_from_payload` ~line 1063, `score_conditional_from_payload` ~line 1151, `sample_autoregressive_state_vmap_exact_from_payload` ~line 1246). Add `if ligand is None: raise TypeError("PrxteinLigandMPNN.<name> requires ligand= keyword")` guards.

**B) Migrate positional callers to keyword** (verified via grep at spec authoring):
- `src/prxteinmpnn/sampling/state_vmap_payload_logits.py:53-62` — `(prng_key, stack, ligand, ...)` → `(prng_key, stack, ligand=ligand, ...)`
- `src/prxteinmpnn/sampling/state_vmap_payload_logits.py:110-122` — same shape
- `src/prxteinmpnn/sampling/sample.py:612-616` — change positional `ls` → `ligand=ls`

The implementer must re-run the grep before editing to catch any new positional sites added since spec authoring:
```bash
grep -rn "from_payload" src/prxteinmpnn/sampling/ src/prxteinmpnn/scoring/ src/prxteinmpnn/run/
# Inspect each invocation; any positional 3rd-arg LigandStack must move to keyword
```

Gate:
```bash
PYTHONPATH=src uv run python -c "
import inspect
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
for name in ['score_unconditional_from_payload', 'score_conditional_from_payload',
             'sample_autoregressive_state_vmap_exact_from_payload']:
    sig = inspect.signature(getattr(PrxteinLigandMPNN, name))
    assert sig.parameters['ligand'].kind == inspect.Parameter.KEYWORD_ONLY, f'{name}: ligand not keyword-only'
print('Fixer 2 gate passed')
"
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  -q
```

### Fixer 3 — Migrate `sampling/conditional_logits.py` (~25 LOC)

Replace concrete imports with `from prxteinmpnn.protocols import ModelProtocol`. Replace `isinstance` checks with `model.capabilities.is_ligand_model`. Update all annotations.

### Fixer 4 — Migrate `sampling/unconditional_logits.py` (~20 LOC)

Same pattern, lines 115, 164.

### Fixer 5 — Migrate `sampling/state_vmap_payload_logits.py` (~20 LOC)

Same pattern, lines 46, 103. Adjust ligand guards to use `m.capabilities.is_ligand_model`.

### Fixer 6 — Migrate `scoring/score.py` (~30 LOC)

Same pattern + migrate SLF001 call to `apply_multistate_to_all_logits` import from `model._shared`. Remove SLF001 comment.

### Fixer 7 — Migrate `run/averaging.py` and `run/conformational_inference.py` (~30 LOC)

Same pattern + `combine_logits_multistate_idx` migration.

### Fixer 8 — Add structural-satisfaction typing test (~35 LOC)

Create `tests/typing/test_protocol_satisfies.py`:

```python
import inspect
import jax
from prxteinmpnn.model.capabilities import ModelCapabilities
from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.model.ligand_mpnn import PrxteinLigandMPNN
from prxteinmpnn.protocols import ModelProtocol

KEY = jax.random.PRNGKey(42)

def test_protein_mpnn_satisfies_protocol():
    m = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=KEY)
    assert isinstance(m, ModelProtocol)
    assert isinstance(m.capabilities, ModelCapabilities)
    for method in ["score_unconditional_from_payload",
                   "score_conditional_from_payload",
                   "sample_autoregressive_state_vmap_exact_from_payload"]:
        assert hasattr(m, method)

def test_ligand_mpnn_keyword_only_ligand():
    for method in ["score_unconditional_from_payload",
                   "score_conditional_from_payload",
                   "sample_autoregressive_state_vmap_exact_from_payload"]:
        sig = inspect.signature(getattr(PrxteinLigandMPNN, method))
        assert "ligand" in sig.parameters
        assert sig.parameters["ligand"].kind == inspect.Parameter.KEYWORD_ONLY
```

Gate:
```bash
PYTHONPATH=src uv run pytest tests/typing/test_protocol_satisfies.py -v
```

---

## 9. Dependency Note

**Depends on**: nothing — shippable at HEAD. The naming-discipline spec is parallel.

**Is prerequisite for**: the model deduplication spec (`2026-05-13-mpnn-dedup-spec.md`). The dedup work cannot proceed safely without the protocol seam in place.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| `DiffusionPrxteinMPNN` doesn't satisfy extended `ModelProtocol` | Fixer 8 typing test includes diffusion check |
| `is_ligand_model` branching mismatches a new model variant | Document in `ModelCapabilities` that `is_ligand_model=True` implies `_from_payload` accepts `ligand: LigandStack` keyword |
| Ligand positional→keyword change breaks callers | **THREE positional sites exist today** (state_vmap_payload_logits.py:53, :110; sample.py:612). Fixer 2 migrates them in the same atomic commit as the signature change. A new positional caller introduced between spec authoring and execution would surface as `TypeError`. Implementer must re-run the discovery grep before editing |
| `apply_multistate_to_all_logits` from `_shared` has subtle delta vs class delegate | The class delegates are thin wrappers calling `_shared` directly — verify with `git grep` before Fixer 6 |
| `@runtime_checkable` only checks attribute existence, not signatures | Expected — signature mismatch is mypy/pyright concern, not runtime |

---

## 11. Gates

```bash
PYTHONPATH=src uv run pytest \
  tests/sampling/test_sample.py \
  tests/model/test_ligand_wave_parallel.py \
  tests/sampling/test_state_vmap_exact_jit.py \
  tests/sampling/test_sample_call_kw_contract.py \
  tests/typing/test_protocol_satisfies.py \
  -q

PYTHONPATH=src uv run python -m mypy \
  src/prxteinmpnn/protocols.py \
  src/prxteinmpnn/sampling/conditional_logits.py \
  src/prxteinmpnn/sampling/unconditional_logits.py \
  src/prxteinmpnn/scoring/score.py \
  --ignore-missing-imports
```

---

## 12. Architecture Note

Post-implementation dependency at the seam:
```
sampling/*, scoring/*
    ↓ import
  protocols.ModelProtocol
    ↑ satisfies (structural)
  model/mpnn.py, ligand_mpnn.py, diffusion_mpnn.py
```

`isinstance(model, PrxteinLigandMPNN)` disappears. Any future model class works in sampling/scoring without changes outside `model/`, provided it satisfies `ModelProtocol` structurally (enforced by Fixer 8 typing test).

Add a one-paragraph note to `protocols.py` top docstring when Fixer 1 lands.

---

## 13. Out of Scope

- Deduplication of `mpnn.py` / `ligand_mpnn.py` (separate spec; this is its prerequisite)
- Renaming `sample_autoregressive_state_vmap_exact` → `sample_ar_exact` on model classes (naming-discipline spec)
- Removing `EncoderPreFn` / `EncoderPostFn` (cleanup sprint)
- Typing `run/_dispatcher.py` (optional, no behavioral consequence)
- Protocol coverage of `Packer` (I/O utility, not model variant)
- `io/weights.py` (factory must remain concretely typed)
