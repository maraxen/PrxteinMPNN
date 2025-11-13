# Physics Trainer Integration Guide

This document explains how to incorporate the physics trainer (PhysicsEncoder) into the overfit test and general training pipeline.

## Overview

The physics training integration works through **model surgery** - replacing the standard encoder with a `PhysicsEncoder` that can optionally accept physics-based electrostatic features as initial node representations.

## Key Components

### 1. **PhysicsEncoder** (`src/prxteinmpnn/model/physics_encoder.py`)

- Wraps the standard `Encoder` class
- Accepts optional `initial_node_features` parameter
- Projects physics features (5D) to the model's node feature dimension
- Falls back to zero initialization if no physics features provided

### 2. **TrainingSpecification** (`src/prxteinmpnn/training/specs.py`)

- Already has `use_physics_features: bool = False` flag
- Already has `physics_feature_weight: float = 1.0` for future use

### 3. **Physics Features** (`src/prxteinmpnn/physics/features.py`)

- `compute_electrostatic_node_features(protein: ProteinTuple) -> jax.Array`
  - Returns shape `(n_residues, 5)` electrostatic features
- `compute_electrostatic_features_batch(proteins: list[ProteinTuple]) -> tuple[jax.Array, jax.Array]`
  - Returns padded features `(batch_size, max_length, 5)` and mask

### 4. **Protein Data Structure** (`src/prxteinmpnn/utils/data_structures.py`)

- `Protein` dataclass already stores:
  - `charges`: Partial charges from PQR files
  - `full_coordinates`: All atomic coordinates  
  - `estat_*`: Electrostatic-related metadata

## Integration Steps

### Step 1: Enable Physics Features in Training Specification

```python
spec = TrainingSpecification(
    # ... other parameters ...
    use_physics_features=True,
    physics_feature_weight=1.0,
)
```

### Step 2: Model Surgery in `_init_checkpoint_and_model()`

The trainer now applies physics encoder surgery after model initialization:

```python
if spec.use_physics_features:
    physics_encoder = create_physics_encoder(
        model.encoder,
        use_initial_features=True,
    )
    model = eqx.tree_at(
        lambda m: m.encoder,
        model,
        physics_encoder,
    )
```

### Step 3: Pass Physics Features in Training Steps

When the encoder is replaced with `PhysicsEncoder`, you can pass `initial_node_features`:

```python
# In train_step and eval_step, compute physics features from batch
if use_physics_features and batch.charges is not None:
    physics_features = compute_electrostatic_node_features(batch)
else:
    physics_features = None

# Pass to model forward pass
node_features, edge_features = encoder(
    edge_features,
    neighbor_indices,
    mask,
    initial_node_features=physics_features,  # NEW parameter
)
```

## Data Requirements

For physics features to work, your protein data must include:

1. **PQR Format**: Standard PQR files with atomic coordinates and partial charges
2. **Protein Dataclass**: Must have `charges` and `full_coordinates` populated
3. **Batch Format**: Batches must contain the physics-related fields

## Overfit Test Integration

The `overfit_check.py` script now:

1. Sets `use_physics_features=True` in the specification
2. Relies on the trainer to apply physics encoder surgery
3. Requires PQR files in the `scripts/overfit/data/` directory

## Current Implementation Status

✅ **Completed:**

- Physics encoder model surgery in `_init_checkpoint_and_model()`
- TrainingSpecification supports `use_physics_features` flag
- Overfit test specification updated to enable physics features

⚠️ **TODO - Requires Further Implementation:**

1. **Update `train_step()` to compute and pass physics features**
   - Add `compute_electrostatic_node_features()` call for batches
   - Handle batching of physics features
   - Pass `initial_node_features` to encoder

2. **Update `eval_step()` similarly**
   - Compute physics features for validation batches
   - Pass features to encoder during evaluation

3. **Verify batch data structure**
   - Ensure PQR data (charges, coordinates) flows through data loader
   - May need updates to `create_protein_dataset()` if physics features aren't loaded

4. **Test the full pipeline**
   - Run overfit test with actual PQR files
   - Verify physics features are properly integrated
   - Check loss convergence with physics features

## Code Changes Summary

### Modified Files

1. **`src/prxteinmpnn/training/trainer.py`**
   - Added import: `from prxteinmpnn.model.physics_encoder import create_physics_encoder`
   - Updated `_init_checkpoint_and_model()` to apply physics encoder surgery

2. **`scripts/overfit/overfit_check.py`**
   - Set `use_physics_features=True`
   - Removed TODO comment
   - Added logging for physics features

### Recommended Next Steps

1. Implement physics feature computation in `train_step()`
2. Update encoder call signature to pass `initial_node_features`
3. Run tests to verify integration
4. Add unit tests for physics feature computation in training loop

## Reference Architecture

```
Data Loading (PQR files)
        ↓
    Batch Creation (charges, full_coordinates)
        ↓
    Trainer: _init_checkpoint_and_model()
        ↓
    Model Surgery: Standard Encoder → PhysicsEncoder
        ↓
    Training Loop: train_step()
        ↓
    Physics Feature Computation (electrostatic forces)
        ↓
    Encoder Forward Pass (with initial_node_features)
        ↓
    Loss Computation & Backprop
```

## Usage Example

```python
# Create specification with physics features
spec = TrainingSpecification(
    inputs="data/train/",  # PQR files
    batch_size=4,
    num_epochs=10,
    use_physics_features=True,  # Enable physics features
    physics_feature_weight=1.0,
)

# Run training - model surgery is applied automatically
result = train(spec)

# Model now uses electrostatic features as initial node representations
```
