"""Training configuration specification."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from prxteinmpnn.run.specs import RunSpecification


@dataclass
class TrainingSpecification(RunSpecification):
  """Configuration for training PrxteinMPNN.

  Extends RunSpecification with training-specific hyperparameters.

  Attributes:
      # Data
      training_data: Path to training data (PQR files, PDB files, or preprocessed cache)
      validation_data: Optional path to validation data
      cache_preprocessed: Whether to cache preprocessed physics features

      # Optimization
      learning_rate: Peak learning rate (default: 1e-4)
      weight_decay: L2 regularization coefficient (default: 0.01)
      warmup_steps: Number of warmup steps for learning rate schedule
      total_steps: Total number of training steps
      gradient_clip: Max gradient norm for clipping (default: 1.0)

      # Training loop
      batch_size: Number of proteins per batch (inherited from RunSpecification)
      num_epochs: Number of training epochs (alternative to total_steps)
      eval_every: Evaluate on validation set every N steps
      log_every: Log metrics every N steps

      # Precision
      precision: Mixed precision mode ("fp32", "fp16", "bf16")

      # Checkpointing
      checkpoint_dir: Directory to save checkpoints
      checkpoint_every: Save checkpoint every N steps
      keep_last_n_checkpoints: Number of recent checkpoints to keep
      resume_from_checkpoint: Path to checkpoint to resume from

      # Physics features (for Phase 1 experiment)
      use_physics_features: Whether to use physics-augmented features
      physics_feature_weight: Weight for physics features in loss (future use)

      # Regularization
      backbone_noise: Noise level for data augmentation (inherited)
      label_smoothing: Label smoothing factor (default: 0.0)

      # Early stopping
      early_stopping_patience: Stop if no improvement for N eval steps (None = disabled)
      early_stopping_metric: Metric to monitor ("val_loss", "val_accuracy")

  Example:
      >>> spec = TrainingSpecification(
      ...     inputs="data/train/",  # Training PQR files
      ...     validation_data="data/val/",
      ...     learning_rate=1e-4,
      ...     weight_decay=0.01,
      ...     batch_size=8,
      ...     num_epochs=10,
      ...     precision="bf16",
      ...     checkpoint_dir="checkpoints/",
      ...     use_physics_features=True,
      ... )
      >>> from prxteinmpnn.training import train
      >>> results = train(spec)

  """
  model_weights: str | Path | None = None,
  model_version: str | None = None,
  # Data paths
  validation_data: str | Path | None = None
  cache_preprocessed: bool = True

  # Optimizer hyperparameters
  learning_rate: float = 1e-4
  weight_decay: float = 0.01
  warmup_steps: int = 1000
  total_steps: int | None = None
  gradient_clip: float | None = None

  # Training loop
  num_epochs: int = 10
  eval_every: int = 500
  log_every: int = 100

  # Precision
  precision: Literal["fp32", "fp16", "bf16"] = "bf16"

  # Checkpointing
  checkpoint_dir: str | Path = Path("checkpoints/")
  checkpoint_every: int = 1000
  keep_last_n_checkpoints: int = 3
  resume_from_checkpoint: str | Path | None = None
  save_at_epochs: Sequence[int] | None = None

  # Physics features (Phase 1)
  use_electrostatics: bool = False
  use_vdw: bool = False
  physics_feature_weight: float = 1.0

  # Data Augmentation & Truncation
  max_length: int | None = None
  truncation_strategy: Literal["random_crop", "center_crop", "none"] = "none"

  # Regularization
  label_smoothing: float = 0.0
  mask_strategy: Literal["random_order", "bert"] = "random_order"
  mask_prob: float = 0.15

  # Training Mode
  training_mode: Literal["autoregressive", "diffusion"] = "autoregressive"

  # Diffusion parameters
  diffusion_num_steps: int = 1000
  diffusion_schedule_type: Literal["cosine", "linear"] = "cosine"
  diffusion_beta_start: float = 1e-4
  diffusion_beta_end: float = 0.02

  # Early stopping
  early_stopping_patience: int | None = None
  early_stopping_metric: Literal["val_loss", "val_accuracy", "val_perplexity"] = "val_loss"

  # Preprocessed data support
  use_preprocessed: bool = False
  """If True, load from preprocessed array_record files instead of parsing on-the-fly."""

  preprocessed_index_path: Path | None = None
  """Path to index file for preprocessed data (required if use_preprocessed=True)."""

  validation_preprocessed_path: Path | None = None
  """Path to validation array_record file (if using preprocessed validation data)."""

  validation_preprocessed_index_path: Path | None = None
  """Path to validation index file (if using preprocessed validation data)."""

  def __post_init__(self) -> None:
    """Validate training specification."""
    super().__post_init__()

    # Ensure checkpoint_dir is Path
    object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))

    # Ensure validation_data is Path if provided
    if self.validation_data and isinstance(self.validation_data, str):
      object.__setattr__(self, "validation_data", Path(self.validation_data))

    # Validate precision
    if self.precision not in ("fp32", "fp16", "bf16"):
      msg = f"precision must be one of ['fp32', 'fp16', 'bf16'], got {self.precision}"
      raise ValueError(msg)

    # Validate that either total_steps or num_epochs is provided
    if self.total_steps is None and self.num_epochs is None:
      msg = "Either total_steps or num_epochs must be provided"
      raise ValueError(msg)

    # Create checkpoint directory if it doesn't exist
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)  # pyright: ignore[reportAttributeAccessIssue]
