# Proxide Upgrade Specification

## Objective

Update `proxide` to support parity-critical features for `PrxteinMPNN` and dynamic updates for `prolix`.

## 1. Expose Neighbor Indices

To resolve the conditional decoding parity failure (~0.35 correlation), `PrxteinMPNN` must use the **exact same** neighbor graph that was used to compute the RBF features.

### Rust (`oxidize`) implementation

- **Modify `compute_rbf`**:
  - Update to return a tuple: `(rbf_features, neighbor_indices)`.
  - Currently, it likely does `neighbors = tree.query(...)` internally and drops the indices.
- **Update Structures**:
  - Add `neighbor_indices` field to `Protein` and `AtomicSystem` Rust structs.
  - Type: `Array2<i32>` (Shape: `[num_residues, k_neighbors]`).
- **Python Conversion**:
  - Ensure `neighbor_indices` are converted to numpy array and passed in the `rust_dict`.

### Python (`proxide`) implementation

- **Update `Protein` Dataclass**:
  - Add `neighbor_indices: jnp.ndarray | None = None`.
  - Update `from_rust_dict` to populate this field.

## 2. Dynamic Feature Updates & Gaussian Noising

To support noise-based training (Coordinate Augmentation) while maintaining feature consistency, `proxide` must handle the noise application and feature re-computation together.

### Rust (`oxidize`) Binding

Create a new method `update_features` or `apply_noise` on the `Protein` object (or a standalone function).

#### `update_with_noise(sigma: f32, seed: u64)`

1. **Apply Noise**: Add Gaussian noise with std dev `sigma` to `coordinates` (and `full_coordinates` if present).
2. **Recompute Topology** (Optional/Advanced): If noise is large enough to break topology assumptions (unlikely for typical training noise), rebuild neighbor lists.
3. **Recompute Masked Features**:
    - If `rbf_features` were present, recompute them using the *new* coordinates (and potentially new neighbors if re-running KNN). *Crucially: Return the new keys/indices.*
    - If `electrostatic_features` were present, recompute.
    - If `vdw_features` were present, recompute.
4. **Mutate/Return**: Update the struct in-place or return a new one.

#### `update_coordinates(new_coords: Array)`

- Allow external updates (e.g. from `prolix` MD steps or Manual changes).
- Trigger the same feature re-computation logic as above.

## 3. Integration Plan

### For `PrxteinMPNN`

1. **Training Loop**: Moving noise application from JAX (inside model) to Data Loading (proxide/Rust).
    - *Why*: JAX cannot easily call Rust to recompute RBFs mid-batch without significant overhead. Scaling noise in the data loader is efficient.
    - *Flow*: `DataLoader` calls `protein.update_with_noise(sigma)` -> Yields `Protein` with noisy coords + consistent RBFs + consistent Neighbors.
    - *Model*: Accepts `rbf_features` and `neighbor_indices`. Skips internal `apply_noise` and `compute_neighbor_graph`.

### For `Prolix`

1. **AtomicSystem**: Ensure `AtomicSystem` is the primary data structure.
2. **Simulation Loop**:
    - Step MD (OpenMM/JaxMD).
    - Update `AtomicSystem.coordinates`.
    - Call `proxide.update_features(system)`.
    - Retrieve fresh energies/forces/features for analysis or ML-biasing.

## 4. Immediate Action Items for `proxide` Maintainers

1. [ ] Modify `compute_rbf` in `src/features.rs` (or equivalent) to return indices.
2. [ ] Add `neighbor_indices` to `Protein` struct in `src/containers.rs`.
3. [ ] Implement `PyProtein::update_with_noise` in `src/python_bindings.rs`.
4. [ ] Release `proxide` version `0.x.y`.
