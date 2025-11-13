"""Implementation guide for completing the physics trainer integration.

This shows the exact code changes needed to fully integrate physics features
into the training loop.
"""

# ============================================================================

# STEP 1: Add physics feature computation to train_step()

# ============================================================================

# File: src/prxteinmpnn/training/trainer.py

# Add to imports

# from prxteinmpnn.physics.features import compute_electrostatic_node_features

# Update train_step signature to accept use_physics_features flag

def train_step(
  model: PrxteinMPNN | eqx.Module,
  opt_state: optax.OptState,
  optimizer: optax.GradientTransformation,
  coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  prng_key: jax.Array,
  label_smoothing: float,
  current_step: int,
  lr_schedule: optax.Schedule,
  use_physics_features: bool = False,  # NEW PARAMETER
  charges: jax.Array | None = None,  # NEW PARAMETER
  full_coordinates: jax.Array | None = None,  # NEW PARAMETER
) -> tuple[PrxteinMPNN, optax.OptState, TrainingMetrics]:
  """Single training step with optional physics features.
  
  Args:
      model: PrxteinMPNN model (may have PhysicsEncoder if enabled)
      opt_state: Optimizer state
      optimizer: Optax optimizer
      coordinates: Backbone coordinates (batch_size, seq_len, 4, 3)
      mask: Valid residue mask (batch_size, seq_len)
      residue_index: Residue indices (batch_size, seq_len)
      chain_index: Chain indices (batch_size, seq_len)
      sequence: Target sequence (batch_size, seq_len)
      prng_key: PRNG key
      label_smoothing: Label smoothing factor
      current_step: Current training step
      lr_schedule: Learning rate schedule
      use_physics_features: Whether to compute and use physics features
      charges: Partial charges for physics features (batch_size, seq_len, 5)
      full_coordinates: Full atomic coordinates (batch_size, n_atoms, 3)

  Returns:
      Tuple of (updated_model, updated_opt_state, metrics)
  """
  batch_size = coordinates.shape[0]

  def loss_fn(model: PrxteinMPNN | eqx.Module) -> tuple[jax.Array, jax.Array]:
    """Compute loss for current batch."""
    # Split PRNG keys for each item in batch
    batch_keys = jax.random.split(prng_key, batch_size)

    def single_forward(
      coords: BackboneCoordinates,
      mask_i: jax.Array,
      res_idx: jax.Array,
      chain_idx: jax.Array,
      key: jax.Array,
      physics_features: jax.Array | None = None,  # NEW PARAMETER
    ) -> Logits:
      """Forward pass for a single protein with optional physics features."""
      # Call model encoder with physics features if available
      if use_physics_features and physics_features is not None:
        # PhysicsEncoder accepts initial_node_features
        encoder = model.encoder  # pyright: ignore[reportAttributeAccessIssue]
        node_features, edge_features = encoder.base_encoder(  # Access the wrapped encoder
          edge_features,  # These are computed earlier
          neighbor_indices,
          mask_i,
          initial_node_features=physics_features,  # Pass physics features
        )
        # Continue with decoder...
      else:
        # Standard forward pass without physics features
        _, logits = model(
          coords,
          mask_i,
          res_idx,
          chain_idx,
          decoding_approach="unconditional",
          prng_key=key,
          backbone_noise=jnp.array(0.0),
        )
      return logits

    # Compute physics features if enabled
    if use_physics_features and charges is not None and full_coordinates is not None:
      # For batched computation, vmap over batch dimension
      def compute_physics_features_single(
        charges_i: jax.Array,
        full_coords_i: jax.Array,
      ) -> jax.Array:
        """Compute physics features for a single structure."""
        # Create minimal ProteinTuple for feature computation
        # This requires creating a ProteinTuple from the batch data
        # For now, assume features are precomputed or we need to extract them
        # from charges and full_coordinates
        return jnp.zeros((coordinates.shape[1], 5))  # Placeholder
      
      physics_features_batch = jax.vmap(compute_physics_features_single)(
        charges, full_coordinates
      )
    else:
      physics_features_batch = None

    logits_batch = jax.vmap(single_forward)(
      coordinates,
      mask,
      residue_index,
      chain_index,
      batch_keys,
      physics_features_batch,
    )

    # Rest of loss computation remains the same...
    def batch_loss(logits: Logits, seq: jax.Array, msk: jax.Array) -> jax.Array:
      return cross_entropy_loss(logits, seq, msk, label_smoothing)

    losses = jax.vmap(batch_loss)(logits_batch, sequence, mask)
    loss = jnp.mean(losses)

    return loss, logits_batch

# Compute gradients

  (loss, logits_batch), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

# Rest remains the same

# ... (metrics computation, optimizer update, etc.)

# ============================================================================

# STEP 2: Update the trainer's train loop to pass physics features

# ============================================================================

# File: src/prxteinmpnn/training/trainer.py

# In the train() function, update the train_step call

for batch in train_loader:
  prng_key, subkey = jax.random.split(prng_key)
  
# NEW: Extract physics-related data from batch

  charges = getattr(batch, 'charges', None)
  full_coordinates = getattr(batch, 'full_coordinates', None)

  model, opt_state, train_metrics = eqx.filter_jit(train_step)(
    model,
    opt_state,
    optimizer,
    batch.coordinates,
    batch.mask,
    batch.residue_index,
    batch.chain_index,
    batch.aatype,
    subkey,
    spec.label_smoothing,
    step,
    lr_schedule,
    use_physics_features=spec.use_physics_features,  # NEW
    charges=charges,  # NEW
    full_coordinates=full_coordinates,  # NEW
  )

  step += 1

# ============================================================================

# STEP 3: Similar updates for eval_step()

# ============================================================================

# File: src/prxteinmpnn/training/trainer.py

def eval_step(
  model: PrxteinMPNN,
  coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  prng_key: jax.Array,
  use_physics_features: bool = False,  # NEW
  charges: jax.Array | None = None,  # NEW
  full_coordinates: jax.Array | None = None,  # NEW
) -> EvaluationMetrics:
  """Single evaluation step with optional physics features."""

# Similar structure to train_step

# Compute physics features if enabled

# Pass to encoder forward pass

# Return metrics

# ============================================================================

# STEP 4: Update validation loop to pass physics features

# ============================================================================

# File: src/prxteinmpnn/training/trainer.py

# In the train() function, update the validation loop

if val_loader and step % spec.eval_every == 0:
  val_metrics_list = []
  for val_batch in val_loader:
    prng_key, subkey = jax.random.split(prng_key)

    # NEW: Extract physics-related data from validation batch
    val_charges = getattr(val_batch, 'charges', None)
    val_full_coords = getattr(val_batch, 'full_coordinates', None)
    
    val_metrics = eqx.filter_jit(eval_step)(
      model,
      val_batch.coordinates,
      val_batch.mask,
      val_batch.residue_index,
      val_batch.chain_index,
      val_batch.aatype,
      subkey,
      use_physics_features=spec.use_physics_features,  # NEW
      charges=val_charges,  # NEW
      full_coordinates=val_full_coords,  # NEW
    )
    val_metrics_list.append(val_metrics)

# ============================================================================

# STEP 5: Verify batch structure includes physics data

# ============================================================================

# File: src/prxteinmpnn/io/loaders.py (create_protein_dataset)

# Ensure that create_protein_dataset includes physics-related fields

# - charges (from PQR files)

# - full_coordinates (all heavy atoms, not just backbone)

# - Other PQR metadata

# If these aren't already included, update the batch creation logic

def create_protein_dataset(...):
  """..."""

# Should already have from Protein.from_tuple()

# - charges: Optional charges from PQR

# - full_coordinates: Full atomic positions

# - full_atom_mask: Mask for full atoms

# - estat_*: Electrostatic metadata

# ============================================================================

# ALTERNATIVE: Pre-compute physics features in data loader

# ============================================================================

# This might be more efficient than computing them during training

# File: src/prxteinmpnn/io/loaders.py

# Add to batch creation

from prxteinmpnn.physics.features import compute_electrostatic_features_batch

def create_protein_dataset(..., compute_physics_features: bool = False):
  """Create dataset with optional pre-computed physics features."""

# ... existing code
  
  if compute_physics_features:
    # Pre-compute electrostatic features for each batch
    # Add to Protein or create a wrapper dataclass
    pass

# ============================================================================

# TESTING: Minimal example

# ============================================================================

# Test the physics encoder integration

import jax
import jax.numpy as jnp
from prxteinmpnn.model.physics_encoder import create_physics_encoder
from prxteinmpnn.io.weights import load_model

# Load model

model = load_model("v_48_020", weights=None)

# Apply physics encoder surgery

physics_encoder = create_physics_encoder(
    model.encoder,
    use_initial_features=True,
)
model_with_physics = eqx.tree_at(
    lambda m: m.encoder,
    model,
    physics_encoder,
)

# Create dummy physics features (batch_size, seq_len, 5)

physics_features = jnp.randn(4, 50, 5)

# The encoder should now accept initial_node_features parameter

# (if you update the model forward pass to pass it through)
