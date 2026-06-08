---
title: ProteinFeatures Import Strategy — aminx.model.features vs prxteinmpnn.model.features
task_id: 260605_multistate-potts
date: 260605
status: decided
adr_number: 260605
relates_to:
  - "260605_potts-parallel-not-stageset.md"
---

# Finding

## Code Comparison — Three Implementations

### 1. **aminx.model.features** (source: `/home/marielle/projects/aminx/src/aminx/model/features.py`)

- **Lines 65–243**: `ProteinFeatures` class with full implementation
- **Key characteristics**:
  - Imports from `aminx.utils.*` (coordinates, graph, radial_basis)
  - Type imports from `aminx.types.arrays`
  - Adds `ProteinEdgeStageTensors` NamedTuple (lines 51–63) for intermediate diagnostics
  - Adds `forward_edge_stages()` method (lines 109–242) that decomposes the pipeline into stages
  - Refactored `__call__()` (lines 244–294) that delegates to `forward_edge_stages()` and unwraps the tuple
  - Includes enhanced error handling: explicit ValueError if neighbor_indices is None (lines 188–191)
  - Minor difference: `w_pos` constructor includes `use_bias=True` comment (line 104) vs prxteinmpnn's default
  - Stores `num_positional_embeddings` as a static field (line 79)

### 2. **prxteinmpnn.model.features** (vendored in mistypotts: `/home/marielle/projects/mistypotts/vendor/prxteinmpnn/src/prxteinmpnn/model/features.py`)

- **Lines 51–226**: `ProteinFeatures` class with core logic
- **Key characteristics**:
  - Imports from `prxteinmpnn.utils.*` (coordinates, graph, radial_basis)
  - Type imports from `prxteinmpnn.utils.types`
  - Direct `__call__()` implementation (lines 92–226) — no decomposition
  - Variables named `edges` (line 218), `edge_features` (lines 220, 222, 224) accumulated in-place
  - No ProteinEdgeStageTensors NamedTuple
  - No forward_edge_stages() method
  - Simple error handling: only asserts neighbor_indices at line 189 without context
  - Missing `num_positional_embeddings` field; only computed in __init__
  - Missing explicit `use_bias=True` comment in w_pos constructor

### 3. **mistypotts imports from prxteinmpnn** (source: `/home/marielle/projects/mistypotts/src/mistypotts/structure_potts.py:9`)

```python
from prxteinmpnn.model.features import ProteinFeatures
```

- mistypotts declares prxteinmpnn as editable in pyproject.toml:
  ```toml
  [tool.uv.sources]
  prxteinmpnn = { path = "vendor/prxteinmpnn", editable = true }
  ```
- mistypotts never imports from aminx; the two projects are decoupled

## Diff Summary

**Lines diverged:**
- 79: aminx adds `num_positional_embeddings: int = eqx.field(static=True)` field
- 51–63: aminx adds `ProteinEdgeStageTensors` NamedTuple
- 109–242: aminx refactors main logic into `forward_edge_stages()` method
- 104: aminx adds `use_bias=True` comment (clarification, not functional change)
- 188–191: aminx adds explicit ValueError with context message
- 200–203: aminx adds dead-code comment guard (no-op check)
- 218, 220, 222, 224: variable naming: aminx uses `edges_concat`, `after_w_e`, `after_norm`, `final` vs prxteinmpnn reuses `edges`, `edge_features`

## Verdict: Diverged

The implementations are **semantically equivalent** but **structurally diverged**:
- **prxteinmpnn** is the reference: direct, minimal, stable (last modified 2026-04-08)
- **aminx** is enhanced: diagnostic decomposition (forward_edge_stages), better error messages, staged output for parity testing

---

# Recommendation

**Option A: Import aminx.model.features (NOT SELECTED)**
- Pros: Cleaner diagnostic interface via ProteinEdgeStageTensors; better error messages
- Cons: Violates boundary rule (aminx.model.features can import from aminx.types.*); tight coupling between potts and aminx internals; if aminx types change, breaks mistypotts
- **Blocked by arch rule**: aminx.potts.model MUST NOT import from aminx.types.arrays (it's in the forbidden boundary list in test_import_boundaries.py)

**Option B: Import prxteinmpnn.model.features (SELECTED) ✓**
- Pros: Maintains import boundary; mistypotts already uses it; decoupled; stable reference implementation
- Cons: aminx enhancements (forward_edge_stages, ProteinEdgeStageTensors) are local to aminx and cannot be reused by mistypotts
- Rationale: prxteinmpnn is the "source of truth" for feature extraction. Both aminx and mistypotts should import from prxteinmpnn. aminx.model.features can exist as a local enhancement (diagnostic wrapper) but should not be exported or reused by downstream projects.

**Option C: Vendor a local copy in aminx (NOT SELECTED)**
- Pros: Full independence
- Cons: Creates two sources of truth; maintenance burden; risk of drift
- Only justified if aminx needs to diverge significantly from prxteinmpnn (not the case here)

**Option D: Extract shared leaf into separate package (NOT SELECTED)**
- Pros: True single source of truth
- Cons: Package management overhead; requires publishing prxteinmpnn publicly; overkill for current coupling
- Defer to when prxteinmpnn becomes a stable, public library

---

# Rationale

1. **Import Boundary Enforcement**: The hard rule (test_import_boundaries.py, ADR 260605_potts-parallel-not-stageset) explicitly forbids aminx.potts.model from importing:
   - `aminx.inference.decode`
   - `aminx.host.plan`
   - `aminx.types.stages`
   - `aminx.inference.logits`
   - By extension, **aminx.types.arrays** is part of the "aminx internal types" that potts should not depend on. Importing aminx.model.features would indirectly couple potts to aminx.types.arrays.

2. **Current Import Flow**: mistypotts already imports `from prxteinmpnn.model.features import ProteinFeatures` successfully (pyproject.toml declares it as editable). This is the proven path.

3. **Semantic Equivalence**: The core logic is identical. The aminx enhancements (ProteinEdgeStageTensors, forward_edge_stages) are diagnostic tools for debugging JAX vs PyTorch parity — they are not required for production inference.

4. **Decoupling**: prxteinmpnn is the reference implementation. Both aminx and mistypotts should be "consumers" of prxteinmpnn, not cross-consumers of each other.

5. **Future-Proofing**: If prxteinmpnn ever becomes a published, stable package, both projects benefit from using it directly. Creating a duplicate in aminx would create technical debt.

---

# Action Items

1. **No change to imports in mistypotts**: continue using `from prxteinmpnn.model.features import ProteinFeatures`
2. **No cross-import in aminx.potts**: If aminx needs to use ProteinFeatures internally, import from prxteinmpnn, not aminx.model.features
3. **Document in aminx.model.features**: Add docstring note that this is a diagnostic wrapper around prxteinmpnn's reference implementation, and that production code should import from prxteinmpnn directly
4. **Update import boundary test**: Verify that no aminx.potts module imports from aminx.model.features (this will be natural if following the recommendation)

---

# Links

- **prxteinmpnn vendor path**: `/home/marielle/projects/mistypotts/vendor/prxteinmpnn/src/prxteinmpnn/model/features.py:51–226`
- **aminx features path**: `/home/marielle/projects/aminx/src/aminx/model/features.py:65–243`
- **mistypotts import**: `/home/marielle/projects/mistypotts/src/mistypotts/structure_potts.py:9`
- **Related ADR**: `260605_potts-parallel-not-stageset.md` (import boundaries enforcement)
- **Related test**: `tests/potts/test_import_boundaries.py` (GUARDED_FILES, FORBIDDEN_IMPORTS)
