---
title: Potts-MPNN Alphabet Alignment — No Collision Risk
task_id: 260605_multistate-potts
date: 260605
adr_id: 260605_potts-alphabet-alignment
status: approved
risk_level: low
---

## Context

Integration of mistypotts (Potts model) with aminx (MPNN-based design) requires confirming alphabet alignment. The task specification flagged a potential semantic collision at index 20 (X vs gap), where both q=21 systems could disagree on character-to-index mapping.

## Investigation

### aminx MPNN_ALPHABET
Source: `/src/aminx/utils/aa_convert.py:16`
```python
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
```
- Length: 21 characters
- Index 20: 'X' (gap/unknown indicator)

### mistypotts MPNN_ALPHABET  
Source: `/vendor/prxteinmpnn/src/prxteinmpnn/utils/aa_convert.py:16`  
```python
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
```
- Length: 21 characters
- Index 20: 'X' (gap/unknown indicator)

### Comparison Table

| Index | aminx | mistypotts | Match |
|-------|-------|------------|-------|
| 0     | A     | A          | ✓     |
| 1     | C     | C          | ✓     |
| 2     | D     | D          | ✓     |
| 3     | E     | E          | ✓     |
| 4     | F     | F          | ✓     |
| 5     | G     | G          | ✓     |
| 6     | H     | H          | ✓     |
| 7     | I     | I          | ✓     |
| 8     | K     | K          | ✓     |
| 9     | L     | L          | ✓     |
| 10    | M     | M          | ✓     |
| 11    | N     | N          | ✓     |
| 12    | P     | P          | ✓     |
| 13    | Q     | Q          | ✓     |
| 14    | R     | R          | ✓     |
| 15    | S     | S          | ✓     |
| 16    | T     | T          | ✓     |
| 17    | V     | V          | ✓     |
| 18    | W     | W          | ✓     |
| 19    | Y     | Y          | ✓     |
| 20    | X     | X          | ✓     |

**Result:** Identity permutation across all 21 indices.

## Decision

### Alignment Status
✅ **No collision risk.** Both aminx and mistypotts use the identical MPNN alphabet and identical index mappings. The 'X' at index 20 is unambiguous in both systems — a gap/unknown indicator.

### Permutation Array
Since the alphabets are identical (identity permutation), no remapping is required:

```python
POTTS_TO_MPNN_ALPHABET_MAP = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
```

This is effectively a no-op; any code using it will confirm "what you put in is what you get out" for Potts indices.

### Character-to-Index Binding

For reference, the canonical binding in both systems:

```python
POTTS_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
```

Index mapping (e.g., for MpnnPottsDesigner):
```python
A=0, C=1, D=2, E=3, F=4, G=5, H=6, I=7, K=8, L=9, M=10, N=11, P=12, Q=13, R=14, S=15, T=16, V=17, W=18, Y=19, X=20
```

## Implementation

### Addition to `src/aminx/potts/model.py`

Add module-level constant after imports:

```python
# Canonical Potts amino acid alphabet (identity with MPNN)
POTTS_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
POTTS_TO_MPNN_ALPHABET_MAP = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
```

Update module docstring to reference this constant.

### Impact on MpnnPottsDesigner

When `MpnnPottsDesigner` seeds Gibbs sampling from MPNN logits:
- MPNN logits: shape (N, 21) in MPNN_ALPHABET order
- Potts input: shape (N, 21) must be in POTTS_ALPHABET order
- **Action:** Since alphabets are identical, no index permutation needed. Direct pass-through.

This static alignment contract should be documented in the designer's docstring.

## Risk Assessment

| Category | Risk | Confidence | Mitigation |
|----------|------|-----------|-----------|
| Alphabet mismatch | None — identity permutation | High | Constants + docstring enforces binding |
| Index 20 collision | None — both use X | High | Explicit comparison table in this ADR |
| Potts-to-MPNN conversion | Not needed (identity) | High | Permutation array provided for future safety |

## Next Steps

1. ✅ Add `POTTS_ALPHABET` and `POTTS_TO_MPNN_ALPHABET_MAP` to `model.py`
2. ✅ Update module docstring to reference alphabet binding
3. ⏭️ When `MpnnPottsDesigner` is implemented (P-10), document the identity binding in its docstring
4. ⏭️ In P-11 testing, verify Gibbs sampling respects alphabet on synthetic test cases

---

**Verdict:** P-10 and P-11 may proceed without alphabet mapping logic. The identity binding is stable and enforced by constants.
