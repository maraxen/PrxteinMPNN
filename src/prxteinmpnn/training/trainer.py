"""Main training loop for PrxteinMPNN."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

from prxteinmpnn.io.loaders import create_protein_dataset
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.training.checkpoint import restore_checkpoint, save_checkpoint
from prxteinmpnn.training.losses import (
  cross_entropy_loss,
  perplexity,
  sequence_recovery_accuracy,
)
from prxteinmpnn.training.metrics import (
  EvaluationMetrics,
  TrainingMetrics,
  compute_grad_norm,
)

if TYPE_CHECKING:
  from chex import ArrayTree

  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.training.specs import TrainingSpecification
  from prxteinmpnn.utils.types import BackboneCoordinates, Logits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def create_optimizer(
  spec: TrainingSpecification,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
  """Create AdamW optimizer with learning rate schedule.

  Args:
      spec: Training specification

  Returns:
      Tuple of (optimizer, learning_rate_schedule)

  """
  # Learning rate schedule with warmup
  if spec.warmup_steps > 0:
    schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=spec.learning_rate,
      warmup_steps=spec.warmup_steps,
      decay_steps=spec.total_steps or (spec.num_epochs * 1000),  # Rough estimate
      end_value=spec.learning_rate * 0.1,
    )
  else:
    schedule = optax.constant_schedule(spec.learning_rate)

  # Create optimizer chain
  optimizer = optax.chain(
    optax.clip_by_global_norm(spec.gradient_clip),  # Gradient clipping
    optax.adamw(
      learning_rate=schedule,
      weight_decay=spec.weight_decay,
    ),
  )

  return optimizer, schedule


@dataclass
class TrainingResult:
  """Container for results returned by :func:`train`.

  Attributes:
    final_model: The trained model instance.
    final_step: The final training step index.
    checkpoint_dir: Path to the checkpoint directory used.

  """

  final_model: PrxteinMPNN | eqx.Module
  final_step: int
  checkpoint_dir: str | Path


def _init_checkpoint_and_model(
  spec: TrainingSpecification,
) -> tuple[PrxteinMPNN | eqx.Module, Any, int, ocp.CheckpointManager]:
  """Initialize or restore model, optimizer state and checkpoint manager.

  Returns (model, opt_state, start_step, checkpoint_manager).
  """
  checkpoint_dir = Path(spec.checkpoint_dir)
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  options = ocp.CheckpointManagerOptions(max_to_keep=spec.keep_last_n_checkpoints)
  checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

  opt_state: ArrayTree | None = None
  if spec.resume_from_checkpoint:
    model_template = load_model(spec.model_version, spec.model_weights)
    model, opt_state, _, start_step = restore_checkpoint(
      checkpoint_manager,
      model_template,
      step=None,  # Load latest
    )
    logger.info("Resumed from checkpoint at step %d", start_step)
  else:
    model = load_model(spec.model_version, spec.model_weights)
    start_step = 0
    # Initialize optimizer state with filtered model parameters
    optimizer_obj, _ = create_optimizer(spec)
    opt_state = optimizer_obj.init(eqx.filter(model, eqx.is_inexact_array))

  return model, opt_state, start_step, checkpoint_manager


def _create_dataloaders(spec: TrainingSpecification) -> tuple[Any, Any]:
  """Create training and validation data loaders based on the spec.

  Returns:
    Tuple of (train_loader, val_loader) where val_loader may be None.

  """
  train_loader = create_protein_dataset(
    spec.inputs,  # pyright: ignore[reportArgumentType]
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database,
  )

  val_loader = None
  if spec.validation_data:
    val_loader = create_protein_dataset(
      spec.validation_data,
      batch_size=spec.batch_size,
      foldcomp_database=spec.foldcomp_database,
    )

  return train_loader, val_loader


def setup_mixed_precision(precision: str) -> None:
  """Configure JAX mixed precision policy.

  Args:
      precision: One of "fp32", "fp16", "bf16"

  """
  if precision == "fp16":
    logger.info("Using FP16 mixed precision (manual casting required)")
  elif precision == "bf16":
    # BF16 is natively supported on TPU/newer GPUs
    logger.info("Using BF16 mixed precision")
  else:
    logger.info("Using FP32 (full precision)")


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
) -> tuple[PrxteinMPNN, optax.OptState, TrainingMetrics]:
  """Single training step.

  Args:
      model: PrxteinMPNN model
      opt_state: Optimizer state
      optimizer: Optax optimizer
      coordinates: Backbone coordinates
      mask: Valid residue mask
      residue_index: Residue indices
      chain_index: Chain indices
      sequence: Target sequence (integer labels)
      prng_key: PRNG key
      label_smoothing: Label smoothing factor
      current_step: Current training step (used for learning rate scheduling)
      lr_schedule: Learning rate schedule function

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
      mask: jax.Array,
      res_idx: jax.Array,
      chain_idx: jax.Array,
      key: jax.Array,
    ) -> Logits:
      """Forward pass for a single protein."""
      _, logits = model(
        coords,
        mask,
        res_idx,
        chain_idx,
        decoding_approach="unconditional",
        prng_key=key,
        backbone_noise=jnp.array(0.0),  # Can add noise during training if desired
      )  # pyright: ignore[reportCallIssue]
      return logits

    logits_batch = jax.vmap(single_forward)(
      coordinates,
      mask,
      residue_index,
      chain_index,
      batch_keys,
    )  # (batch_size, seq_len, 21)

    # Compute loss for each item in batch, then average
    def batch_loss(logits: Logits, seq: jax.Array, msk: jax.Array) -> jax.Array:
      return cross_entropy_loss(logits, seq, msk, label_smoothing)

    losses = jax.vmap(batch_loss)(logits_batch, sequence, mask)
    loss = jnp.mean(losses)

    return loss, logits_batch

  # Compute gradients
  (loss, logits_batch), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

  # Compute metrics (average across batch)
  def batch_metrics(logits: Logits, seq: jax.Array, msk: jax.Array) -> tuple[jax.Array, jax.Array]:
    acc = sequence_recovery_accuracy(logits, seq, msk)
    ppl = perplexity(logits, seq, msk)
    return acc, ppl

  accuracies, perplexities = jax.vmap(batch_metrics)(logits_batch, sequence, mask)
  accuracy = jnp.mean(accuracies)
  ppl = jnp.mean(perplexities)
  grad_norm = compute_grad_norm(grads)
  # Get current learning rate
  current_lr = lr_schedule(current_step)

  # Update parameters
  params = eqx.filter(model, eqx.is_inexact_array)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_model = eqx.apply_updates(model, updates)

  metrics = TrainingMetrics(
    loss=loss,
    accuracy=accuracy,
    perplexity=ppl,
    learning_rate=float(current_lr),
    grad_norm=grad_norm,
  )

  return new_model, new_opt_state, metrics


def eval_step(
  model: PrxteinMPNN,
  coordinates: jax.Array,  # (batch_size, seq_len, 4, 3)
  mask: jax.Array,  # (batch_size, seq_len)
  residue_index: jax.Array,  # (batch_size, seq_len)
  chain_index: jax.Array,  # (batch_size, seq_len)
  sequence: jax.Array,  # (batch_size, seq_len)
  prng_key: jax.Array,
) -> EvaluationMetrics:
  """Single evaluation step with batching.

  Args:
      model: PrxteinMPNN model
      coordinates: Backbone coordinates (batched)
      mask: Valid residue mask (batched)
      residue_index: Residue indices (batched)
      chain_index: Chain indices (batched)
      sequence: Target sequence (batched)
      prng_key: PRNG key

  Returns:
      Evaluation metrics

  """
  batch_size = coordinates.shape[0]
  batch_keys = jax.random.split(prng_key, batch_size)

  def single_forward(
    coords: jax.Array,
    msk: jax.Array,
    res_idx: jax.Array,
    chain_idx: jax.Array,
    key: jax.Array,
  ) -> Logits:
    """Forward pass for a single protein."""
    _, logits = model(
      coords,
      msk,
      res_idx,
      chain_idx,
      decoding_approach="unconditional",
      prng_key=key,
      backbone_noise=jnp.array(0.0),
    )
    return logits

  logits_batch = jax.vmap(single_forward)(
    coordinates,
    mask,
    residue_index,
    chain_index,
    batch_keys,
  )

  def batch_metrics(
    logits: jax.Array,
    seq: jax.Array,
    msk: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    val_loss = cross_entropy_loss(logits, seq, msk, label_smoothing=0.0)
    val_accuracy = sequence_recovery_accuracy(logits, seq, msk)
    val_ppl = perplexity(logits, seq, msk)
    return val_loss, val_accuracy, val_ppl

  losses, accuracies, perplexities = jax.vmap(batch_metrics)(
    logits_batch,
    sequence,
    mask,
  )

  return EvaluationMetrics(
    val_loss=jnp.mean(losses),
    val_accuracy=jnp.mean(accuracies),
    val_perplexity=jnp.mean(perplexities),
  )


def train(spec: TrainingSpecification) -> TrainingResult:
  """Train PrxteinMPNN model.

  Args:
      spec: Training specification

  Returns:
      Dictionary with training results and final model

  Example:
      >>> spec = TrainingSpecification(
      ...     inputs="data/train/",
      ...     validation_data="data/val/",
      ...     batch_size=8,
      ...     num_epochs=10,
      ...     learning_rate=1e-4,
      ... )
      >>> results = train(spec)
      >>> print(f"Final validation accuracy: {results['final_val_accuracy']}")

  """
  # Setup
  setup_mixed_precision(spec.precision)
  logger.info("Starting training with spec: %s", spec)

  optimizer, lr_schedule = create_optimizer(spec)

  # Initialize or restore model and checkpoint manager
  model, opt_state, start_step, checkpoint_manager = _init_checkpoint_and_model(spec)

  # Create data loaders
  train_loader, val_loader = _create_dataloaders(spec)

  step = start_step
  best_val_metric = float("inf")
  patience_counter = 0

  prng_key = jax.random.PRNGKey(spec.random_seed)

  logger.info("Starting training loop...")

  for epoch in range(spec.num_epochs):
    logger.info("Epoch %d/%d", epoch + 1, spec.num_epochs)

    for batch in train_loader:
      prng_key, subkey = jax.random.split(prng_key)

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
      )

      step += 1

      if val_loader and step % spec.eval_every == 0:
        val_metrics_list = []
        for val_batch in val_loader:
          prng_key, subkey = jax.random.split(prng_key)
          val_metrics = eqx.filter_jit(eval_step)(
            model,
            val_batch.coordinates,
            val_batch.mask,
            val_batch.residue_index,
            val_batch.chain_index,
            val_batch.aatype,
            subkey,
          )
          val_metrics_list.append(val_metrics)

        # Average validation metrics
        avg_val_loss = jnp.mean(jnp.array([m.val_loss for m in val_metrics_list]))
        avg_val_acc = jnp.mean(jnp.array([m.val_accuracy for m in val_metrics_list]))

        # Convert JAX arrays to Python floats for logging
        val_loss_float = jax.device_get(avg_val_loss).item()
        val_acc_float = jax.device_get(avg_val_acc).item()

        logger.info(
          "Validation at step %d: val_loss=%.4f, val_acc=%.4f",
          step,
          val_loss_float,
          val_acc_float,
        )

        # Early stopping check
        if spec.early_stopping_patience:
          current_metric = avg_val_loss  # Can switch based on spec.early_stopping_metric
          if current_metric < best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
          else:
            patience_counter += 1

          if patience_counter >= spec.early_stopping_patience:
            logger.info("Early stopping triggered at step %d", step)
            break

      # Checkpointing
      if step % spec.checkpoint_every == 0:
        save_checkpoint(
          checkpoint_manager,
          step,
          model,
          opt_state,
          metrics=train_metrics.to_dict(),
        )

  logger.info("Training complete!")

  checkpoint_manager.close()

  return TrainingResult(final_model=model, final_step=step, checkpoint_dir=spec.checkpoint_dir)
