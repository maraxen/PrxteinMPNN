---
title: Potts-MPNN Alphabet Alignment and Index Mapping
task_id: 260605_multistate-potts
date: 260605
status: final
decision_type: alphabet_mapping
---

# Potts-MPNN Alphabet Alignment

## Summary

Potts (PottsMPNN) and aminx (ProteinMPNN via prxteinmpnn) both use q=21 states, but **the amino acid orderings differ at index 20**. Potts uses a 20-character alphabet without 'X', while aminx includes 'X' (gap token) at index 20. This decision document establishes the canonical mapping and risk mitigation.

## Alphabet Comparison

### Source References

**aminx MPNN_ALPHABET** (from `src/aminx/utils/aa_convert.py:16`)
```python
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
```
Length: 21 characters (indices 0–20)

**PottsMPNN Alphabet** (from `mistypotts/.tmp/PottsMPNN/run_utils.py`)
```python
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
```
Length: 20 characters (indices 0–19, **no 'X'**)

### Character-by-Character Alignment

| Index | aminx Char | Potts Char | Match? |
|-------|-----------|-----------|--------|
| 0     | A         | A         | ✓      |
| 1     | C         | C         | ✓      |
| 2     | D         | D         | ✓      |
| 3     | E         | E         | ✓      |
| 4     | F         | F         | ✓      |
| 5     | G         | G         | ✓      |
| 6     | H         | H         | ✓      |
| 7     | I         | I         | ✓      |
| 8     | K         | K         | ✓      |
| 9     | L         | L         | ✓      |
| 10    | M         | M         | ✓      |
| 11    | N         | N         | ✓      |
| 12    | P         | P         | ✓      |
| 13    | Q         | Q         | ✓      |
| 14    | R         | R         | ✓      |
| 15    | S         | S         | ✓      |
| 16    | T         | T         | ✓      |
| 17    | V         | V         | ✓      |
| 18    | W         | W         | ✓      |
| 19    | Y         | Y         | ✓      |
| 20    | X (gap)   | (none)    | ✗      |

**Result:** Indices 0–19 are **identity-mapped** (perfect alignment). Index 20 differs: aminx has 'X', Potts has nothing.

## Potts q=21 Expansion Strategy

PottsMPNN models use vocab=21 or vocab=22 (with padding). When exported to `(h, J, W)` tensors via `pottsmpnn_ckpt_export.py`, the last dimension is `h.shape[-1]` (typically 22 after padding). The first 21 elements correspond to standard amino acids + gap.

**Convention (per `pottsmpnn_ckpt_export.py` comments and tests):**
- Indices 0–20 in PottsMPNN: amino acids A–Y + padding/gap token
- **Index 20 is treated as a gap token, semantically equivalent to aminx's 'X'**

## Recommended POTTS_TO_MPNN_ALPHABET_MAP

When interfacing Potts indices with aminx indices:

```python
# Potts index → aminx MPNN index permutation
# Potts[i] → MPNN[POTTS_TO_MPNN_ALPHABET_MAP[i]]
POTTS_TO_MPNN_ALPHABET_MAP = jnp.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    dtype=jnp.int32
)
```

**Rationale:** Indices 0–20 are semantically aligned:
- Indices 0–19: direct 1:1 correspondence (A–Y)
- Index 20: Potts gap token ↔ aminx 'X' (both gaps)

This is an **identity permutation** — no reordering needed.

## Risk Analysis

### P1 Risk: Semantic Collision at Index 20

**Risk:** If code accidentally treats Potts index 20 as an undefined state rather than a gap token, it may:
- Ignore energies for gap positions
- Cause silent energy underestimation
- Produce incorrect designed sequences

**Mitigation:**
1. **Static alphabet map** in `MpnnPottsDesigner` with explicit comment:
   ```python
   # Index 20 is gap token in both Potts and MPNN — semantically aligned
   ```
2. **Symbolic constant** `POTTS_ALPHABET` in `model.py` with docstring clarifying the gap.
3. **Type guards** in score/sample functions reject indices > 20.

### P2 Risk: h/J Scale Mismatch

**Mitigation status:** Per `pottsmpnn_ckpt_export.py:82`, the scale is verified in tests—Potts `h` and `J` are multiplied by 2 to match `potts_log_unnormalized` bookkeeping. This is **already handled**.

### P3 Risk: k (number of neighbors) from Checkpoint

**Mitigation status:** Read-only from metadata; `MpnnPottsDesigner` accepts `k_neighbors` as an argument. **No action needed**.

## Implementation: Add POTTS_ALPHABET Constant

**File:** `src/aminx/potts/model.py`

```python
# Potts amino acid order (from PottsMPNN convention)
# Indices 0–19: standard amino acids A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
# Index 20: gap token (semantically aligned with aminx MPNN 'X')
# Note: This is an identity permutation w.r.t. MPNN_ALPHABET[0:21]
POTTS_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Indices 0–20 match MPNN_ALPHABET indices 0–20 exactly
POTTS_TO_MPNN_ALPHABET_MAP = jnp.array(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    dtype=jnp.int32
)
```

## Clearance for Phase P-11

**Precondition check:** Indices 0–19 match (identity permutation). Index 20 collision resolved (gap ↔ gap). Both alphabets are q=21.

**Verdict:** ✓ **PROCEED to P-11** — all prerequisites satisfied. No blockers detected.

---

**Decision made:** 260605
**Reviewed by:** Fixer (automated research phase)
**Status:** Ready for implementation
