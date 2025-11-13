"""Checkpointing utilities for training.

This module provides helper functions for saving and restoring Equinox models
using Orbax checkpointing. The main purpose is to handle the Equinox-specific
filtering of model parameters before saving.

For most use cases, you should use Orbax's CheckpointManager directly. These
helpers are provided for convenience when working with Equinox models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import orbax.checkpoint as ocp

if TYPE_CHECKING:
  import optax

  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.training.metrics import TrainingMetrics


def save_checkpoint(
  manager: ocp.CheckpointManager,
  step: int,
  model: PrxteinMPNN,
  opt_state: optax.OptState,
  metrics: TrainingMetrics | None = None,
) -> bool:
  """Save Equinox model and optimizer state using Orbax CheckpointManager.

  This is a convenience function that handles filtering the Equinox model
  to save only trainable parameters (inexact arrays), avoiding serialization
  issues with non-JAX-serializable objects like functools.partial.

  Args:
      manager: Orbax CheckpointManager instance
      step: Current training step
      model: PrxteinMPNN model
      opt_state: Optimizer state (from optax)
      metrics: Optional training metrics to save

  Returns:
      True if checkpoint was saved, False otherwise

  Example:
      >>> import orbax.checkpoint as ocp
      >>> manager = ocp.CheckpointManager(
      ...     path,
      ...     options=ocp.CheckpointManagerOptions(max_to_keep=3),
      ... )
      >>> saved = save_checkpoint(manager, step=0, model=model, opt_state=opt_state)

  """
  model_params = eqx.filter(model, eqx.is_inexact_array)
  metrics_dict: dict[str, float | None] = {}
  if metrics is not None:
    metrics_dict = {
      "loss": float(jax.device_get(metrics.loss)),
      "accuracy": float(jax.device_get(metrics.accuracy)),
      "perplexity": float(jax.device_get(metrics.perplexity)),
      "learning_rate": float(metrics.learning_rate),
      "grad_norm": float(jax.device_get(metrics.grad_norm))
      if metrics.grad_norm is not None
      else None,
    }
  return manager.save(
    step,
    args=ocp.args.Composite(
      model=ocp.args.StandardSave(model_params),  # pyright: ignore[reportCallIssue]
      opt_state=ocp.args.StandardSave(opt_state),  # pyright: ignore[reportCallIssue]
      metrics=ocp.args.JsonSave(metrics_dict),  # pyright: ignore[reportCallIssue]
    ),
  )


def restore_checkpoint(
  manager: ocp.CheckpointManager,
  model_template: PrxteinMPNN,
  step: int | None = None,
) -> tuple[PrxteinMPNN, optax.OptState, TrainingMetrics, int]:
  """Restore Equinox model and optimizer state from Orbax checkpoint.

  This is a convenience function that handles restoring an Equinox model
  from saved parameters and combining them with a template model structure.

  Args:
      manager: Orbax CheckpointManager instance
      model_template: Template model for structure reconstruction
      step: Specific step to load (if None, loads latest checkpoint)

  Returns:
      Tuple of (model, opt_state, metrics, step)

  Raises:
      ValueError: If no checkpoint found

  Example:
      >>> import orbax.checkpoint as ocp
      >>> manager = ocp.CheckpointManager(path)
      >>> model, opt_state, metrics, step = restore_checkpoint(manager, model_template)

  """
  # Determine which checkpoint to restore
  if step is None:
    step = manager.latest_step()
    if step is None:
      msg = f"No checkpoints found in {manager.directory}"
      raise ValueError(msg)

  abstract_model = eqx.filter(model_template, eqx.is_inexact_array)
  restored = manager.restore(
    step,
    args=ocp.args.Composite(
      model=ocp.args.StandardRestore(abstract_model),  # pyright: ignore[reportCallIssue]
      opt_state=ocp.args.StandardRestore(None),  # pyright: ignore[reportCallIssue]
      metrics=ocp.args.JsonRestore(),  # pyright: ignore[reportCallIssue]
    ),
  )

  restored_model = eqx.combine(restored["model"], model_template)

  return restored_model, restored["opt_state"], restored["metrics"], step
