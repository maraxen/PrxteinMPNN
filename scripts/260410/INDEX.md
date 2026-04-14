# Verification Scripts - 2026-04-10

This directory contains scripts used to verify the implementation of MultiState LigandMPNN design pool generation and efficient ArrayRecord storage.

## Scripts

### 1. `verify_weighted_multistate.py`
- **Purpose**: Verifies the core logit combination logic in `multi_state_sampling.py`.
- **Coverage**:
    - Weighted Arithmetic Mean (LogSumExp).
    - Weighted Geometric Mean (Log probability average).
    - Weighted Product (Sum of weighted logits).
    - Numerical stability check for large logit values.

### 2. `verify_ligand_tied_multistate.py`
- **Purpose**: Validates that `PrxteinLigandMPNN` correctly handles tied-position sampling.
- **Coverage**:
    - Ensures tied residues sample identical amino acids.
    - Verifies that `state_weights` and `state_mapping` correctly influence the consensus sequence.
    - Confirms tied positions receive identical combined logits.

### 3. `verify_design_storage.py`
- **Purpose**: Confirms the functionality of `DesignArrayRecordWriter` and the Msgpack serialization layer.
- **Coverage**:
    - Writing design payloads (sequence, logits, scores, weights, metadata) to compressed ArrayRecords.
    - Round-trip verification: Reading records back and ensuring data integrity.
    - Verification of JAX-to-NumPy conversion for serialization compatibility.

### 4. `verify_massive_sampling.py`
- **Purpose**: Documents the high-performance parallelization strategy.
- **Strategy**:
    - `jax.vmap`: Used to parallelize sequence generation across GPU cores (intra-batch).
    - `jax.lax.map`: Used to sequence batches through the device without OOM (inter-batch).
    - `jax.experimental.io_callback`: Used to stream batched results out of the JIT loop asynchronously.
