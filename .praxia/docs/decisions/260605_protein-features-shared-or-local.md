---
title: ProteinFeatures Sharing Decision — Aminx vs. Prxteinmpnn vs. Mistypotts
task_id: 260605_potts-integration
date: 260605
status: final
---

## Finding

Three implementations of `ProteinFeatures` exist across the codebase:

### 1. **Aminx** (`/home/marielle/projects/aminx/src/aminx/model/features.py`, lines 65–294)
- **Field**: `num_positional_embeddings: int = eqx.field(static=True)` (line 79)
- **W_pos bias**: `use_bias=True` (line 104)
- **Extra method**: `forward_edge_stages()` (lines 109–242) — diagnostic intermediates for JAX/PyTorch parity testing
- **Extra NamedTuple**: `ProteinEdgeStageTensors` (lines 51–62) — diagnostic tensors
- **Returns from `__call__`**: `(edge_features, neighbor_indices, node_features_out, prng_key)` (line 294)
- **Docstring**: Explicit multi-state support via `structure_mapping` parameter (lines 268–270)

### 2. **Prxteinmpnn** (mistypotts vendor, `/home/marielle/projects/mistypotts/vendor/prxteinmpnn/src/prxteinmpnn/model/features.py`, lines 51–226)
- **Field**: NO `num_positional_embeddings` field (parameter only, not stored)
- **W_pos bias**: NO `use_bias` argument — defaults to `False` (line 87)
- **No diagnostic method**: No `forward_edge_stages()` equivalent
- **No diagnostic NamedTuple**
- **Returns from `__call__`**: `(edge_features, neighbor_indices, node_features, prng_key)` (line 226)
- **Docstring**: Identical multi-state support language

### 3. **Mistypotts usage** (`/home/marielle/projects/mistypotts/src/mistypotts/structure_potts.py`, lines 70–75)
```python
self.features = ProteinFeatures(
  node_features=128,
  edge_features=edge_features_dim,
  k_neighbors=k_neighbors,
  key=k0,
)
```
- **Dependency**: Imports from `prxteinmpnn.model.features` (line 9)
- **Call sites**: Lines 98, 127 — calls with minimal args (`prng_key`, `structure_coordinates`, `mask`, `residue_index`, `chain_index`, `backbone_noise=None`)
- **Usage**: Treats output as `(edge_knn, nei, _node_unused, _)` — ignores node features

## Core Differences

| Aspect | Aminx | Prxteinmpnn |
|--------|-------|------------|
| **Stored metadata** | Stores `num_positional_embeddings` | Discards it after `__init__` |
| **W_pos layer** | `use_bias=True` + comment | No bias argument (False default) |
| **Diagnosis capability** | `forward_edge_stages()` + `ProteinEdgeStageTensors` | None |
| **Installed as package** | Yes (in aminx project) | Vendored in mistypotts only |
| **ABI compatibility** | ✓ Both accept `num_positional_embeddings=16` default | ✓ Calls work with both |
| **Serialization risk** | High — stored field will differ in saved weights | None — field never stored in prxteinmpnn |

## Recommendation

### Three-Option Evaluation

**(a) Import from aminx.model.features**
- **Path**: Mistypotts imports `from aminx.model.features import ProteinFeatures` (allowed by boundary rule; `aminx.model.features` not in forbidden list).
- **Status**: This is the forward path for mistypotts. Aminx carries extended diagnostics (`forward_edge_stages()`, `ProteinEdgeStageTensors`) unused by mistypotts, but ABI is stable — all call sites work with both versions.
- **Why selected**: Consolidates on the full-featured version; avoids duplication; simplifies mistypotts' dependency graph.

**(b) Import from prxteinmpnn as live package dependency**
- **Current state**: Mistypotts currently vendors prxteinmpnn and uses its stripped-down version (no `num_positional_embeddings` field, `use_bias=False` hardcoded).
- **Risk identified**: The prxteinmpnn version has `use_bias=False` baked into weight loading and serialization. Existing checkpoints trained on this divergent behavior cannot be loaded directly into aminx's `use_bias=True` version without retraining. Additionally, keeping prxteinmpnn as a live package dependency (rather than vendored) creates an external maintenance burden and versioning risk.
- **Why not selected**: (1) Checkpoints incompatibility — use_bias divergence breaks existing models; (2) No stable, installable prxteinmpnn package (vendored in mistypotts only); (3) Keeps mistypotts coupled to an external, unmaintained reference codebase.

**(c) Vendor copy of aminx into mistypotts**
- **Approach**: Copy aminx's full `ProteinFeatures` implementation (with diagnostics) into mistypotts' vendored tree.
- **Duplication risk**: Creates a second copy of the same logic. Future changes to feature extraction or diagnostics must be applied in two places (aminx + mistypotts), risking divergence and hidden bugs.
- **Why not selected**: Violates DRY principle; maintenance burden outweighs the marginal isolation gain; cross-project coordination overhead.

### Chosen Path: **(a) with legacy compatibility layer**

- **Aminx**: `ProteinFeatures` remains in `src/aminx/model/features.py` with full diagnostics (`forward_edge_stages()`, `ProteinEdgeStageTensors`).
- **Mistypotts**: Change import from vendored prxteinmpnn to `from aminx.model.features import ProteinFeatures` (option a).
- **Legacy checkpoint support**: Keep prxteinmpnn's vendored copy in mistypotts for reference only — existing checkpoints trained on `use_bias=False` behavior remain readable. Document this explicitly.
- **Forward path**: New models use aminx's version (option a provides bias + diagnostics + ABI stability).

### Rationale for Selection

1. **Boundary Rule Compliance**: `aminx.model.features.ProteinFeatures` is allowed to import (not in forbidden list); mistypotts should use aminx's version, not maintain a separate copy.

2. **Divergence is intentional**:
   - Aminx's `forward_edge_stages()` exists for **diagnostic parity testing between JAX and PyTorch reference models**. It's not used in production inference.
   - Prxteinmpnn's stripped-down version is optimized for weight loading and production inference (mistypotts' original use case).
   - Merging avoids duplication (option c rejected); using aminx (option a) provides one source of truth while legacy prxteinmpnn remains available for checkpoint compatibility.

3. **Metadata field risk mitigation**:
   - Prxteinmpnn doesn't store `num_positional_embeddings`, so checkpoints trained on prxteinmpnn can't be directly loaded and introspected in aminx (would need static annotation at load time).
   - Keeping prxteinmpnn vendored as reference preserves the metadata distinction and documents legacy behavior.
   - Option (b) rejected because it recreates the divergence at runtime; option (a) moves mistypotts to the canonical version.

4. **Stability and trust**:
   - Prxteinmpnn's checkpoint weights may have been created with the exact `use_bias=False` behavior. Option (a) eliminates future confusion by pinning mistypotts to aminx's version for all new work.
   - Prxteinmpnn stays vendored for legacy checkpoint compatibility only (option b's rationale for NOT using as live dep).
   - Option (c) rejected because vendoring a full copy duplicates maintenance burden without additional benefit.

## Action Items

1. Update mistypotts `structure_potts.py` line 9: `from aminx.model.features import ProteinFeatures`
2. Verify mistypotts checkpoint compatibility by running `graph_weights()` call with aminx version and comparing output to known reference.
3. Document that aminx carries extended diagnostics (forward_edge_stages); mistypotts uses minimal subset.
4. No code merge — keep both implementations (vendored prxteinmpnn as legacy, aminx as forward path).

## Boundary Enforcement

All usage in aminx conforms to the LOCKED ARCH rule:
- [ ] `aminx.potts.model`: migrate import from `prxteinmpnn.model.features` → `aminx.model.features` (action item #1 above; not yet implemented. Current state: lines 204, 320 import from prxteinmpnn)
- ✓ Will NOT import `aminx.inference.decode`, `aminx.host.plan`, `aminx.types.stages`, `aminx.inference.logits`
- ✓ Designer only exception remains in scope

Mistypotts (cross-project) is permitted to import `aminx.model.features` per this decision; boundary rule applies only within aminx.
