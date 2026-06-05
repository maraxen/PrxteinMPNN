---
title: Potts Model Architecture — Parallel Family, Not StageSet Consumer
task_id: 260605_multistate-potts
date: 260605
status: Accepted 2026-06-05
---

## Status: Accepted 2026-06-05

**Effective**: All new Potts model code must follow the parallel architecture pattern. Enforcement via lint rule #1304 (forbidden imports from `aminx.inference.decode`, `aminx.host.plan`, `aminx.types.stages`, `aminx.inference.logits`).

## Context: Integration of PottsTRWStructureModel into aminx.potts.PottsModel

### Background
The mistypotts project provides `PottsTRWStructureModel` (structure inference via TRW), which is being integrated into aminx as a **multistate Potts model family** (`aminx.potts.PottsModel`). The integration requires a fundamental architectural decision: should `PottsModel` be a **parallel, self-contained model** operating on its own currency and schedule, or should it be embedded as a **StageSet consumer** within the existing inference pipeline (`aminx.inference.decode` → `aminx.host.plan` → `aminx.types.stages`)?

### Currencies and Phases

**Existing inference pipeline (decode):**
- **Currency**: `(S, L, V)` logits → `EncoderOutput` + `decode_step` adapter
- **Phase**: Autoregressive per-position scan: for each position i, receive `(node_f, edge_f, nei, mask)` → emit logits for position i
- **Grain**: One position per step; cumulative across scan (Markov chain)

**Potts model (TRW):**
- **Currency**: Raw backbone coordinates → `ProteinFeatures` → `DifferentiableTRW` → `(marginals, h, J, rho)`
- **Phase**: One-shot global optimization (TRW messages saturate in ~15 iterations; no per-position feedback)
- **Grain**: Whole structure at once; no Markovian decomposition
- **Algebraic contract**: Returns `(marginals, h, J, rho)` where:
  - `marginals[i, k]` = posterior probability of state k at position i
  - `h[i, k]` = external field (unary term)
  - `J[i, j, k, l]` = pairwise interaction (binary term)
  - `rho` = spectral reweighting (TRW artifact)

### Two Currency Mismatch Examples

#### Example 1: decode_step Adapter Signature
```python
# Existing signature
def decode_step(
    node_f: jax.Array,     # shape (L, D)
    edge_f: jax.Array,     # shape (N_edges, D)
    neighbors: jax.Array,  # shape (L, K)
    mask: jax.Array,       # shape (L,)
) -> jax.Array:           # shape (L, V) — logits
    # Autoregressive: emit logits for next position(s)
    ...

# Potts would need:
def potts_step(
    backbone_coords: jax.Array,  # shape (L, 3) — RAW coordinates
    mask: jax.Array,              # shape (L,)
) -> tuple[
    jax.Array,  # marginals (L, V)
    jax.Array,  # h (L, V)
    jax.Array,  # J (L, L, V, V)
    jax.Array,  # rho (...)
]:
    # Global: solve TRW once, no per-position feedback
    ...
```
The Potts signature requires **raw structural input** (backbone coordinates), not `(node_f, edge_f, nei)`. Those features are **internal to Potts** (computed by `ProteinFeatures` within the model). The decode_step contract assumes those features come from upstream; Potts computes them internally.

#### Example 2: Phase Mismatch
- **Decode**: Scan across positions, accumulate context
  ```python
  for i in range(L):
      logits_i = decode_step(node_f[i], edge_f[i], nei[i], mask[i])
  ```
- **Potts**: Solve once, return all posteriors
  ```python
  marginals, h, J, rho = potts_model(backbone_coords, mask)  # whole structure, once
  ```
  Potts has no per-position step; TRW converges globally in O(iterations), not O(L).

## Decision: Parallel Model Family, NOT StageSet Consumer

### Chosen Architecture

**Potts model is a parallel, self-contained family operating at a different abstraction level:**

1. **Namespace**: `aminx.potts.{model, poe, sampling, designer}` — **not** `aminx.inference.decode` submodule
2. **Input contract**: Raw backbone coordinates, mask; optionally calibration parameters
3. **Output contract**: Marginals, Potts energy (h, J, rho), sampled sequences
4. **Execution model**: Global optimization (TRW), not autoregressive scan
5. **Composition**: Parallel pipeline (PottsModel → PoE → Sampling), not StageSet layers

### Rejected Alternatives

#### Option II: decode_step Adapter (Rejected)
**Proposal**: Wrap Potts as a `decode_step` adapter in the existing autoregressive pipeline.

```python
class PottsDecodeAdapter(eqx.Module):
    def __call__(self, node_f, edge_f, nei, mask, accumulated_context=None):
        # Somehow extract coords from accumulated_context?
        # Ignore node_f, edge_f, nei — recompute internally?
        # Per-position logits only — lose h, J, rho?
        ...
```

**Problems**:
1. **Currency mismatch**: `decode_step` receives `(node_f, edge_f, nei)` (features) but Potts needs raw coordinates. Either:
   - Store coordinates in `accumulated_context` (invasive to pipeline)
   - Invert feature extraction (computationally wasteful)
   - Ignore upstream features and recompute (redundant computation)
2. **Phase mismatch**: `decode_step` is called L times (per position); Potts solves once. Adapting either means:
   - Call Potts L times (wrong — TRW doesn't operate per-position)
   - Cache and reuse (Potts result static across scan — defeats the purpose of scan)
3. **Output currency loss**: Potts returns `(marginals, h, J, rho)` but `decode_step` returns only `(L, V)` logits. Discarding h, J, rho wastes the Potts model's algebraic structure.
4. **Load-bearing pipeline mutation**: `EncoderOutput` would need a new Potts cargo field (h, J, rho, etc.). This is load-bearing stack-wide (see Option III).

**Verdict**: Adds complexity, loses information, and couples Potts to an incompatible contract.

#### Option III: EncoderOutput Widening (Rejected)
**Proposal**: Extend `EncoderOutput` to carry Potts-specific tensors (h, J, rho).

```python
@dataclass
class EncoderOutput:
    edge_logits: jax.Array        # existing
    neighbor_indices: jax.Array   # existing
    ...
    potts_h: jax.Array | None = None      # NEW
    potts_J: jax.Array | None = None      # NEW
    potts_rho: jax.Array | None = None    # NEW
    potts_marginals: jax.Array | None = None  # NEW
```

**Problems**:
1. **Load-bearing stack-wide**: `EncoderOutput` (defined in `aminx/types/encoder.py` or `host/output.py`) is used throughout the stack:
   - `aminx.inference.decode` reads/writes it
   - `aminx.host.plan` schedules on it
   - `aminx.types.stages` wraps it
   - Every `decode_step` implementation expects specific fields
   - Adding Potts fields breaks all downstream code (BREAKING CHANGE to stable interface)
2. **Optional fields pattern is fragile**: Downstream code must check `is not None` on every Potts field. High error surface.
3. **Mixes concerns**: EncoderOutput becomes a "universal container" for all model outputs (Potts, AF2-style, etc.), not a clean abstraction.

**Verdict**: Modifying a load-bearing, widely-used type introduces fragility and breaks the principle of separation of concerns.

#### Option IV: ProductOfProbabilities for PoE (Rejected)
**Proposal**: Represent Potts marginals in the existing logits currency and combine via `ProductOfProbabilities` layer.

```python
# Decode produces (L, V) logits
decode_logits = ...  # shape (L, V)

# Potts produces marginals
potts_marginals = ...  # shape (L, V) — marginals [0, 1]

# Combine via PoE (product of experts)
combined = ProductOfProbabilities(decode_logits, log(potts_marginals))
```

**Problems**:
1. **J tensor has no representation**: Potts J tensor is `(N, N, V, V)` — pairwise interactions. The logits currency `(L, V)` has no room for pairwise terms. Either:
   - Discard J (loses all covariance, pair-interaction information)
   - Store J separately (same as Option III — load-bearing type extension)
2. **Marginals != logits**: Potts returns calibrated marginals `p(s_i = k | data)` in [0, 1]. Logits are unbounded. Mixing requires lossy conversion:
   - logit(marginal) ≈ log(p / (1-p)) — only invertible if marginal not 0 or 1
   - Numerical instability near boundaries
   - Loses calibration information
3. **PoE interpretation is wrong**: Product of Probabilities (PoE) combines independent experts. Potts and decode are not independent — Potts uses structural features to condition on sequence context.

**Verdict**: Discards load-bearing Potts outputs (h, J, rho) and forces a mathematically inappropriate combination model.

## Enforcement: Boundary Rules for aminx.potts

**All Potts model code must follow these rules:**

### Forbidden Imports (in aminx.potts.*)
```python
# FORBIDDEN in aminx/potts/model.py
from aminx.inference import decode      # ✗
from aminx.inference import logits      # ✗
from aminx.host import plan             # ✗
from aminx.types import stages          # ✗
from aminx.types import EncoderOutput   # ✗ (part of inference.* stack)

# ALLOWED in aminx/potts/model.py
from aminx.model import features        # ✓ (ProteinFeatures)
from aminx.model import backbone        # ✓ (if needed)
from mistypotts.structure_potts import PottsTRWStructureModel  # ✓
```

### Exception: Designer Only
**`aminx.potts.designer` is permitted to import from `aminx.inference`** (and `aminx.types.stages`) because it must:
- Inspect decode outputs to decide Potts invocation
- Read EncoderOutput.edge_logits to infer structure features
- Compose Potts + decode in planning scenarios

```python
# ALLOWED in aminx/potts/designer.py ONLY
from aminx.types import EncoderOutput    # ✓ (for composition metadata)
from aminx.inference import decode       # ✓ (for introspection)
from aminx.host import plan              # ✓ (for scheduling Potts + decode)
```

### Enforcement
- **Lint rule #1304**: AST-grep scans `aminx/potts/{model,poe,sampling}.py` and fails if any forbidden import found.
- **Code review gate**: Any PR touching `aminx/potts/` must pass #1304 before merge.
- **CI/CD**: #1304 runs on every commit to main; no bypassability.

## Module Docstring Requirement

Every file in `aminx/potts.{model, poe, sampling}` must include a docstring referencing this ADR:

```python
"""
Potts model family — parallel architecture, not StageSet consumer.

This module is part of the parallel Potts pipeline (PottsModel → PoE → Sampling).
It operates at a different abstraction level than aminx.inference.decode.

ARCHITECTURE: Parallel model family
- Input: Raw backbone coordinates + mask (no features from upstream)
- Output: (marginals, h, J, rho)
- Phase: Global optimization (TRW), not autoregressive scan

BOUNDARY RULE: aminx.potts must NOT import from:
  - aminx.inference.decode
  - aminx.inference.logits
  - aminx.host.plan
  - aminx.types.stages

See: .praxia/docs/decisions/260605_potts-parallel-not-stageset.md
"""
```

`aminx/potts/designer.py` is the exception and must note it explicitly:

```python
"""
Potts × Decode composition designer — orchestrates parallel pipelines.

EXCEPTION TO BOUNDARY RULE: designer.py is permitted to import from:
  - aminx.inference.decode (for output introspection)
  - aminx.host.plan (for scheduling)
  - aminx.types.stages (for metadata)

This enables composing Potts and decode results without violating the boundary.
See: .praxia/docs/decisions/260605_potts-parallel-not-stageset.md
"""
```

## Risk Mitigations

### Risk 1: Alphabet Index Collision (X vs Gap)
**Risk**: Potts alphabet `k ∈ [0, 20]` uses 20 standard amino acids. What about gaps (padding)?
- Mistypotts: Gap = index 20 (not in Potts encoding)
- Aminx: Alphabet includes gap at index 20 (standard AF2 convention)

**Mitigation**:
- Static `alphabet_map` in `aminx.potts.designer` maps aminx indices → mistypotts indices
- At Potts invocation, reindex to mistypotts space (gap → masked)
- After sampling, reindex back to aminx space

**Enforcement**: Synthetic test with known alphabet (20 + gap), verify round-trip reindexing.

### Risk 2: h/J Scale Directed-Slot Convention
**Risk**: Potts h, J tensors have scale factors that depend on directed vs. undirected graph interpretation.
- Mistypotts: Directed slots (h[i], J[i→j])
- Combine via sum (not max) in inference

**Mitigation**: Synthetic check at PoE init — verify h/J scales match expected directed sum.

### Risk 3: k (Alphabet Size) Baked into Checkpoint
**Risk**: Potts checkpoints encode fixed k=20. Future multistate variants (k>20) cannot be loaded.
- Mistypotts: k from checkpoint metadata
- Aminx: Must read k from checkpoint, not assume

**Mitigation**: All weight loading reads k from metadata (not hardcoded). Specify k in RunSpec.

### Risk 4: fori Loop OOM During Training
**Risk**: TRW fori loop materializes all message-passing iterations. On large structures (L>500), OOM.
- Mistypotts: No training loop (structure inference only)
- Aminx: May fine-tune Potts weights — TRW fori is not checkpointed

**Mitigation**: At spec construction time, enforce `backend='scan'` + `checkpoint=True` for training. Document constraint in `potts_trw_spec.py`.

### Risk 5: caliby_path=None Default
**Risk**: Potts expects calibrated marginals. If calibration curve is not available, defaults to identity.
- Mistypotts: No calibration (returns raw marginals)
- Aminx: Must handle None gracefully

**Mitigation**: `caliby_path=None` is identity transform (no-op). Document this. Tests with synthetic data should verify no numerical artifacts.

### Risk 6: Method Name Semantics (infer_params vs. encode/decode)
**Risk**: Potts uses `infer_params`, `sample`, `score` — NOT `encode`, `decode`.
- Existing pipeline uses encode/decode for feature extraction
- Potts uses different names to avoid confusion

**Mitigation**: Lint rule flags any call to `potts_model.encode()` or `.decode()` (will fail at runtime anyway).

## Related Decisions

- **ProteinFeatures sourcing**: See `260605_protein-features-shared-or-local.md` — mistypotts imports `ProteinFeatures` from `aminx.model.features`.
- **PoE combination**: Separate ADR for Potts + decode composition (not this one).

## Timeline

- **2026-06-05**: ADR accepted
- **2026-06-12**: All Potts code must include docstring reference + pass lint #1304
- **2026-06-19**: Full boundary enforcement in CI (no exceptions)

## References

- Mistypotts source: `/home/marielle/projects/mistypotts/src/mistypotts/`
- Aminx Potts stub: `/home/marielle/projects/aminx/src/aminx/potts/`
- Recon findings: `task_id=260605_multistate-potts` transduction log
