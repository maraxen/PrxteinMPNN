# Fix for vmap Dimension Mismatch in Tied Positions Sampling

## Problem

When using tied positions in the sampling pipeline with `batch_size=1`, a `ValueError` was raised:

```python
ValueError: vmap got inconsistent sizes for array axes to be mapped:
* most axes (5 of them) had size 1, e.g. axis 0 of argument coords of type float32[1,182,37,3];
* one axis had size 182: axis 0 of argument current_tie_map of type int32[182]
```

## Root Cause

The `resolve_tie_groups()` function in `src/prxteinmpnn/utils/autoregression.py` returns a 1D array of shape `(n_residues,)` containing the residue-level grouping information for tied positions.

However, in `src/prxteinmpnn/run/sampling.py`, this array is passed to `jax.vmap()` with `in_axes=0`, which tells vmap to map over the first dimension. When the batch size is 1, the coordinates have shape `[1, 182, 37, 3]`, but `tie_group_map` has shape `[182]` (missing the leading batch dimension).

JAX's vmap expects all arrays with `in_axes=0` to have the same size along axis 0 (the batch dimension).

## Solution

Use `jnp.atleast_2d()` combined with `jnp.broadcast_to()` to ensure `tie_group_map` and `mapping` have proper batch dimensions that match the actual batch size before passing them to `jax.vmap()`.

### Changes Made

In `src/prxteinmpnn/run/sampling.py`, in the `_sample_batch()` function:

**Before:**

```python
tie_map_in_axis = 0 if tie_group_map is not None else None
mapping_in_axis = 0 if batched_ensemble.mapping is not None else None

vmap_structures = jax.vmap(
  internal_sample,
  in_axes=(0, 0, 0, 0, None, tie_map_in_axis, mapping_in_axis),
)

sampled_sequences, sampled_logits, _ = vmap_structures(
  batched_ensemble.coordinates,
  batched_ensemble.mask,
  batched_ensemble.residue_index,
  batched_ensemble.chain_index,
  keys,
  tie_group_map,
  batched_ensemble.mapping,
)
```

**After:**

```python
# Ensure tie_group_map and mapping have batch dimensions for vmap
# tie_group_map comes from resolve_tie_groups with shape (n_residues,)
# but vmap expects (batch_size, n_residues) when in_axes=0
batch_size = batched_ensemble.coordinates.shape[0]

tie_map_for_vmap = None
if tie_group_map is not None:
  # Add batch dimension and broadcast: (n,) -> (1, n) -> (batch_size, n)
  tie_map_for_vmap = jnp.broadcast_to(
    jnp.atleast_2d(tie_group_map), 
    (batch_size, tie_group_map.shape[0])
  )

mapping_for_vmap = batched_ensemble.mapping
if batched_ensemble.mapping is not None and batched_ensemble.mapping.ndim == 1:
  # Add batch dimension and broadcast if needed: (n,) -> (1, n) -> (batch_size, n)
  mapping_for_vmap = jnp.broadcast_to(
    jnp.atleast_2d(batched_ensemble.mapping),
    (batch_size, batched_ensemble.mapping.shape[0])
  )

tie_map_in_axis = 0 if tie_map_for_vmap is not None else None
mapping_in_axis = 0 if mapping_for_vmap is not None else None

vmap_structures = jax.vmap(
  internal_sample,
  in_axes=(0, 0, 0, 0, None, tie_map_in_axis, mapping_in_axis),
)

sampled_sequences, sampled_logits, _ = vmap_structures(
  batched_ensemble.coordinates,
  batched_ensemble.mask,
  batched_ensemble.residue_index,
  batched_ensemble.chain_index,
  keys,
  tie_map_for_vmap,
  mapping_for_vmap,
)
```

## Why `jnp.atleast_2d()` + `jnp.broadcast_to()`?

This two-step approach ensures proper batch dimension handling:

1. **`jnp.atleast_2d()`**: Ensures the array has at least 2 dimensions
   - If the input is 1D with shape `(n,)`, it returns a 2D array with shape `(1, n)`
   - If the input is already 2D or higher, it returns the input unchanged

2. **`jnp.broadcast_to()`**: Broadcasts the array to match the actual batch size
   - Takes the `(1, n)` array and broadcasts it to `(batch_size, n)`
   - This ensures the tie_group_map is replicated for each item in the batch
   - JAX handles this efficiently without actually copying the data

This is necessary because:

- `tie_group_map` is the same for all items in a batch (it describes the tying structure)
- But vmap expects all arrays with `in_axes=0` to have the same batch size
- Broadcasting allows us to satisfy vmap's requirements while being memory-efficient

## Alternative Approaches Considered

1. **`jnp.expand_dims(tie_group_map, axis=0)`**: This would work but is less robust - it always adds a dimension even if one already exists.

2. **Modifying `resolve_tie_groups()` to return 2D arrays**: This would require changes to all callers and might break other code.

3. **Using `jnp.reshape(tie_group_map, (1, -1))`**: Similar to expand_dims, but less clear in intent.

## Impact

This fix resolves the dimension mismatch error for:

- Tied positions sampling with any batch size
- Structure mapping with 1D mapping arrays
- Any other similar vmap operations in the sampling pipeline

The fix is minimal, localized, and follows existing patterns in the codebase.

## Testing

To test this fix, run sampling with tied positions:

```python
from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.specs import SamplingSpecification

spec = SamplingSpecification(
    inputs="path/to/structure.pdb",
    tied_positions="direct",  # or other tied position modes
    pass_mode="inter",
    num_samples=1,
    batch_size=1,
)

results = sample(spec)
```

The error should no longer occur, and sampling should complete successfully.
