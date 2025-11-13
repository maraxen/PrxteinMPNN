# Physics Trainer Quick Start

## Current Status

✅ **Completed:**

- Model surgery infrastructure in `_init_checkpoint_and_model()`
- Physics encoder replacement when `use_physics_features=True`
- Overfit test configured to use physics features
- All necessary imports in place

## How to Use

### 1. Enable Physics Features in Training

```python
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import train

spec = TrainingSpecification(
    inputs="path/to/pqr/files/",
    use_physics_features=True,  # Enable physics features
    num_epochs=10,
    # ... other parameters ...
)

result = train(spec)
```

The trainer will automatically:

1. Load the model
2. Apply PhysicsEncoder surgery
3. Replace standard encoder with physics-aware encoder

### 2. Run Overfit Test

```bash
cd scripts/overfit/
uv run python overfit_check.py
```

Look for these log messages:

```
INFO - Applying physics encoder surgery with use_initial_features=True
INFO - Physics encoder applied successfully
INFO - Physics features enabled: True
```

### 3. Data Requirements

Your data needs:

- PQR files (protein structures with charges)
- Atomic coordinates for force computation
- Charge information for electrostatic calculations

The data loader will automatically populate these fields.

## What's Happening Behind the Scenes

1. Model Loading: Standard ProteinMPNN model loaded
2. Physics Encoder Creation: create_physics_encoder() wraps the encoder
3. Model Surgery: eqx.tree_at() replaces model.encoder with PhysicsEncoder
4. Training: PhysicsEncoder can now accept physics features

## Next Steps to Fully Enable Physics Features

### Step 1: Update train_step() (~30 mins)

Add physics feature computation and pass to encoder

File: `src/prxteinmpnn/training/trainer.py`

### Step 2: Update eval_step() (~15 mins)

Apply the same physics feature computation for validation

File: `src/prxteinmpnn/training/trainer.py`

### Step 3: Update Training Loop (~10 mins)

Pass use_physics_features flag to train_step and eval_step calls

File: `src/prxteinmpnn/training/trainer.py` (train function)

### Step 4: Test (~20 mins)

Run overfit test with actual PQR files and verify

## Files Modified

1. `src/prxteinmpnn/training/trainer.py`
   - Added import for create_physics_encoder
   - Updated _init_checkpoint_and_model() with model surgery

2. `scripts/overfit/overfit_check.py`
   - Set use_physics_features=True
   - Added logging for physics features

## Files to Modify Next

1. `src/prxteinmpnn/training/trainer.py`
   - train_step() - add physics feature computation
   - eval_step() - add physics feature computation
   - train() - pass flags to step functions
