# Physics Trainer Integration Summary

## Overview

This document summarizes the integration of the physics trainer (PhysicsEncoder) into the overfitting test and training pipeline.

## What Was Done

### 1. Model Surgery in Trainer (`src/prxteinmpnn/training/trainer.py`)

**Added import:**

```python
from prxteinmpnn.model.physics_encoder import create_physics_encoder
```

**Updated `_init_checkpoint_and_model()` function:**

- After model initialization/loading, checks if `spec.use_physics_features` is enabled
- If enabled, applies **model surgery** to replace the standard encoder with a `PhysicsEncoder`
- The physics encoder wraps the base encoder and can accept `initial_node_features` parameter
- Uses `eqx.tree_at()` for clean model replacement

**Code:**

```python
if spec.use_physics_features:
    logger.info("Applying physics encoder surgery with use_initial_features=True")
    physics_encoder = create_physics_encoder(
        model.encoder,
        use_initial_features=True,
    )
    model = eqx.tree_at(
        lambda m: m.encoder,
        model,
        physics_encoder,
    )
    logger.info("Physics encoder applied successfully")
```

### 2. Overfit Test Updated (`scripts/overfit/overfit_check.py`)

**Set physics features flag:**

```python
spec = TrainingSpecification(
    # ... other parameters ...
    use_physics_features=True,  # ENABLED
    physics_feature_weight=1.0,
)
```

**Updated logging:**

- Now reports when physics features are enabled
- Provides visibility into which model version is being tested

## Architecture

The physics trainer integration works through this flow:

```
┌─────────────────────────────────────┐
│  TrainingSpecification              │
│  use_physics_features: bool = True  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  _init_checkpoint_and_model()       │
│  - Load standard model              │
│  - Check use_physics_features flag  │
│  - Apply model surgery              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Model Surgery with eqx.tree_at()   │
│  Standard Encoder → PhysicsEncoder  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Training Loop (train_step)         │
│  - Computes physics features        │
│  - Passes to encoder                │
│  - Standard loss computation        │
└─────────────────────────────────────┘
```

## Key Components

### PhysicsEncoder (`src/prxteinmpnn/model/physics_encoder.py`)

- Wraps standard `Encoder`
- **Static field:** `use_initial_features` determines behavior
- **Accepts:** `initial_node_features` parameter (optional, shape: n_residues × 5)
- **Falls back:** to zero initialization if features not provided
- **Projection:** Maps 5D physics features to model's node feature dimension

### Protein Data Structure (`src/prxteinmpnn/utils/data_structures.py`)

The `Protein` dataclass already includes:

- `charges`: Partial charges from PQR files
- `full_coordinates`: All atomic coordinates
- `full_atom_mask`: Mask for full atoms
- `estat_*`: Electrostatic metadata fields

These fields are automatically populated from PQR files through the standard data loading pipeline.

## What Still Needs Implementation

### TODO 1: Update `train_step()` to pass physics features

**File:** `src/prxteinmpnn/training/trainer.py`

The training step needs to:

1. Compute electrostatic features from batch data
2. Pass `initial_node_features` to the encoder

**Pseudocode:**

```python
# In train_step loss_fn
if use_physics_features and batch.charges is not None:
    physics_features = compute_electrostatic_node_features(batch)
    encoder_output = encoder(
        edge_features,
        neighbor_indices,
        mask,
        initial_node_features=physics_features,  # NEW
    )
else:
    encoder_output = encoder(...)
```

### TODO 2: Update `eval_step()` similarly

**File:** `src/prxteinmpnn/training/trainer.py`

Apply the same physics feature computation during validation.

### TODO 3: Verify data loader includes PQR data

**File:** `src/prxteinmpnn/io/loaders.py` (if needed)

Ensure that when creating batches:

- `charges` field is populated (from PQR files)
- `full_coordinates` field is populated
- These flow through to training steps

### TODO 4: Test and validate

- Run overfit test with actual PQR files
- Verify loss converges faster or to lower values with physics features
- Check that model properly uses the electrostatic information

## Benefits of This Architecture

1. **Non-invasive:** Model surgery doesn't require changing the base model definition
2. **Optional:** Physics features can be enabled/disabled via config flag
3. **JAX-compatible:** Uses equinox primitives for clean functional updates
4. **Backward compatible:** Standard training works with `use_physics_features=False`
5. **Scalable:** Can extend to other physics encoders using same pattern

## Testing the Integration

To verify the model surgery works correctly:

```python
import jax.numpy as jnp
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.model.physics_encoder import create_physics_encoder
import equinox as eqx

# Load model
model = load_model("v_48_020", weights=None)

# Create physics encoder
physics_encoder = create_physics_encoder(
    model.encoder,
    use_initial_features=True,
)

# Apply surgery
model_with_physics = eqx.tree_at(
    lambda m: m.encoder,
    model,
    physics_encoder,
)

# Verify surgery
assert isinstance(model_with_physics.encoder, PhysicsEncoder)
assert model_with_physics.encoder.use_initial_features is True
```

## Running the Overfit Test

```bash
# With physics features enabled
cd scripts/overfit/
uv run python overfit_check.py

# Should log:
# INFO - Applying physics encoder surgery with use_initial_features=True
# INFO - Physics encoder applied successfully
# INFO - Physics features enabled: True
```

## References

- `src/prxteinmpnn/model/physics_encoder.py` - PhysicsEncoder implementation
- `src/prxteinmpnn/physics/features.py` - Physics feature computation
- `src/prxteinmpnn/training/trainer.py` - Training loop (modified)
- `scripts/overfit/overfit_check.py` - Overfit test (modified)
