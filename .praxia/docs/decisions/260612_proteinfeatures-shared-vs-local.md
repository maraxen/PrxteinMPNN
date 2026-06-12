---
title: ProteinFeatures sourcing confirmed — aminx.model.features is canonical
decision_id: 260612_proteinfeatures-shared-vs-local
date: 2026-06-12
status: Accepted
decision_type: architectural
supersedes: adr/260605_protein-features-shared-or-local.md
relates_to: 260605_protein-features-shared-or-local
---

## Status: Accepted

Confirms and closes the question opened in `260605_protein-features-shared-or-local`.
The migration to Option A (import from `aminx.model.features`) is complete.

## Context

Three implementations of `ProteinFeatures` exist across the broader codebase:

1. **`aminx.model.features.ProteinFeatures`** — canonical, at `src/aminx/model/features.py:65–294`.
   Has `forward_edge_stages()` diagnostic method, `use_bias=True` on `w_pos`, and explicit
   `structure_mapping` multi-state support.

2. **`prxteinmpnn.model.features.ProteinFeatures`** (mistypotts vendor) — diverged reference
   implementation. `use_bias=False` on `w_pos`, no diagnostic method, used by
   `mistypotts.structure_potts`.

3. **`aminx.potts` usage** — `PottsModel` imports `ProteinFeatures` from `aminx.model.features`
   at two call sites in `model.py` (lines 210, 274) via a deferred `PLC0415` import.
   No vendor copy exists inside the `aminx.potts` tree.

## Decision

**Option A** (import from `aminx.model.features`) is confirmed as the implemented path.
No code change is required — the migration is already complete.

Key divergence that does NOT block this decision: `use_bias=True` vs `use_bias=False` on `w_pos`
affects raw edge embeddings but is irrelevant to PottsModel checkpoint compatibility because
`ProteinFeatures` is instantiated transiently in `PottsModel.__call__` and
`PottsModel._build_adjacency` — its weights are never serialized to the PottsModel checkpoint.

## Consequences

- `aminx.potts.model` is boundary-compliant: it imports from `aminx.model.features`, not
  from any prxteinmpnn vendor copy.
- The `forward_edge_stages()` diagnostic capability is available to `PottsModel` via the
  aminx import but currently unused — can be leveraged for parity testing.
- Any future comparison of raw edge embeddings between aminx and prxteinmpnn must account for
  the `use_bias` divergence.

## Out of scope (deferred)

Migrating `mistypotts/src/mistypotts/structure_potts.py` to import from `aminx.model.features`
is a cross-project change. Track separately when mistypotts is in scope.

## Supersedes

`adr/260605_protein-features-shared-or-local.md` chose Option B (import from prxteinmpnn).
That document is superseded — it predates the Option A migration and contradicts the current
code state. See status update in that file.
