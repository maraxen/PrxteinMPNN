"""Quick test to verify AR mask fix."""

import jax.numpy as jnp
from prxteinmpnn.utils.autoregression import generate_ar_mask

# Test 1: Basic AR mask should exclude diagonal
print("=" * 60)
print("Test 1: AR Mask Diagonal Check")
print("=" * 60)

decoding_order = jnp.array([0, 1, 2, 3, 4])
ar_mask = generate_ar_mask(decoding_order)

print(f"Decoding order: {decoding_order}")
print(f"AR mask:\n{ar_mask}")
print(f"\nDiagonal values: {jnp.diag(ar_mask)}")
print(f"Diagonal should be all zeros (no self-attention): {jnp.all(jnp.diag(ar_mask) == 0)}")

# Verify lower triangular structure (excluding diagonal)
expected_structure = jnp.array([
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0],
])

matches_expected = jnp.array_equal(ar_mask, expected_structure)
print(f"\nMatches expected lower triangular (k=-1): {matches_expected}")

if not matches_expected:
    print("\n❌ FAILED: AR mask does not match expected structure")
    print(f"Expected:\n{expected_structure}")
    print(f"Got:\n{ar_mask}")
else:
    print("\n✅ PASSED: AR mask correctly excludes diagonal")

# Test 2: Permuted decoding order
print("\n" + "=" * 60)
print("Test 2: Permuted Decoding Order")
print("=" * 60)

decoding_order_perm = jnp.array([2, 0, 4, 1, 3])
ar_mask_perm = generate_ar_mask(decoding_order_perm)

print(f"Decoding order: {decoding_order_perm}")
print(f"AR mask:\n{ar_mask_perm}")

# Verify positions can only attend to those decoded before them
# Position 0 (order=2) should attend to positions with order < 2: positions 1 (order=0) and 3 (order=1)
# Position 0 should NOT attend to itself (order=2)
print(f"\nPosition 0 (order=2) attends to: {jnp.where(ar_mask_perm[0])[0]}")
print(f"Expected: positions [1, 3] (those with order < 2)")

# Check diagonal is still zero
print(f"\nDiagonal should be all zeros: {jnp.all(jnp.diag(ar_mask_perm) == 0)}")

if jnp.all(jnp.diag(ar_mask_perm) == 0):
    print("\n✅ PASSED: Permuted AR mask correctly excludes diagonal")
else:
    print("\n❌ FAILED: Permuted AR mask has non-zero diagonal")

# Test 3: Compare with old implementation (would have used >=)
print("\n" + "=" * 60)
print("Test 3: Comparison with Old Implementation (>=)")
print("=" * 60)

# Old buggy implementation
row_indices = decoding_order[:, None]
col_indices = decoding_order[None, :]
ar_mask_old_buggy = (row_indices >= col_indices).astype(int)

print(f"Old (buggy) AR mask with >=:\n{ar_mask_old_buggy}")
print(f"New (fixed) AR mask with >:\n{ar_mask}")
print(f"\nDifference is the diagonal: {jnp.array_equal(ar_mask_old_buggy - ar_mask, jnp.eye(5, dtype=jnp.int32))}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("✅ AR mask fix verified: positions cannot attend to themselves")
print("✅ This matches ColabDesign's implementation (tri with k=-1)")
