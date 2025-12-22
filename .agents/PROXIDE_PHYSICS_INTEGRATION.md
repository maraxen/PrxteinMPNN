# Proxide Physics Integration Plan

## Objective

Migrate the computation of physics functionality (Van der Waals interactions, Electrostatics) from Python (`PrxteinMPNN`) to Rust (`proxide`) to improve performance and consistency.

## Motivation

- **Performance**: Rust implementation will be significantly faster, especially for neighbor-list dependent calculations.
- **Consistency**: Centralizing feature computation in `proxide` ensures that `PrxteinMPNN` receives fully pre-processed `Protein` objects, simplifying the Python codebase.
- **Modularity**: Allows `proxide` to support different atom ordering schemes (e.g. MPNN vs Standard) internally without complex Python shuffling.

## Proposed Changes

### 1. Proxide (Rust)

- **Extend `OutputSpec`**: Add flags for enabling physics features.

    ```rust
    pub struct OutputSpec {
        // ... existing
        pub compute_vdw: bool,
        pub compute_electrostatics: bool,
    }
    ```

- **Implement Feature Calculation**:
  - Port VdW calculation logic (Lennard-Jones) to Rust.
  - Port Electrostatics calculation logic (Coulomb/Debye-Huckel) to Rust.
  - Ensure calculations respect the requested `output_format_target` (e.g. if MPNN order is requested, features are emitted in that order if applicable, or documented as atom-indexed).

### 2. PrxteinMPNN (Python)

- **Update `dispatch.py`**:
  - Configure `OutputSpec` to enable `compute_vdw` and `compute_electrostatics` when requested.
- **Update `ProteinFeatures`**:
  - Accept `vdw_features` and `electrostatic_features` as input arguments (already present in `Protein` container).
  - Bypass internal Python calculation if these fields are populated.
- **Deprecate `physics` Modules**:
  - Once verified, remove the legacy Python implementations in `prxteinmpnn.physics`.

## Verification Strategy

1. **Parity Tests**:
    - Create a test comparing Python-computed mechanics vs Rust-computed mechanics on the same structure.
    - Verify numerical agreement (< 1e-5 difference).
2. **Integration Tests**:
    - Run `test_colabdesign_equivalence.py` with Rust physics enabled.
    - Ensure model logits/perplexity remain unaffected.

## Questions/Risks

- **Parameterization**: How will forcefield parameters (sigma, epsilon, charges) be loaded? Proxide already has `load_forcefield` but we need to ensure it's seamless for the default `PrxteinMPNN` usage.
