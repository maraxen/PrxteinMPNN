# Proxide RBF Atom Ordering Discrepancy

## Issue

Proxide computes RBF features using the standard Atom37 backbone order:

- `0: N`
- `1: CA`
- `2: C`
- `3: CB`
- `4: O`

PrxteinMPNN expects (and ColabDesign uses) a permuted order for RBF features:

- `0: N`
- `1: CA`
- `2: C`
- `3: O`
- `4: CB`

## Evidence

- Direct comparison of `proxide.rbf_features` against Python-computed RBFs showed lower error when assuming the Rust/Atom37 order (indices 3 and 4 swapped).
- Current Parity Tests pass only when using the internal Python fallback (which explicitly constructs the N-CA-C-O-CB order correctly).

## Required Fix (When updating binary)

When the `proxide` binary is updated to expose `neighbor_indices`, we must **permute the RBF features** before passing them to the model.

### Implementation Plan

In `prxteinmpnn/io/parsing/dispatch.py` (or where output is processed):

```python
# Swap indices 3 (CB in Rust, O in Python) and 4 (O in Rust, CB in Python)
# RBF features shape: (N, K, 25 * 16) flattened.
# Pairs are: (0,0), (1,1), ... (3,3), (4,4), (1,0), (1,2), (1,3), (1,4), ...
```

Since the RBF encoding flattens the pair dimension, reordering is non-trivial (requires mapping indices).

**Recommended Action**:
Modify `proxide` (Rust) to accept an option for output atom order OR update `PrxteinMPNN` to use standard Atom37 order `(N, CA, C, CB, O)` internally for `compute_backbone_coordinates`, aligning it with `proxide`.

**Wait**, if we change `PrxteinMPNN`, we change the input to the trained model weights.

1. **If the weights were trained with O=3, CB=4**, we MUST feed inputs with O=3, CB=4.
2. Therefore, `proxide` inputs MUST be reordered to match the weights.

**Conclusion**: The Rust output needs to be remapped to `[N, CA, C, O, CB]` *before* RBF computation, OR we need a permutation layer after RBF computation. Since RBF depends on distances, permuting the *atom coordinates* in Rust before RBF calc is the cleanest solution.

**Request**: Update `proxide`'s `compute_rbf` to use `(N, CA, C, O, CB)` order for backbone.
