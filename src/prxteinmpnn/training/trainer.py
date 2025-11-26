"""Main training loop for PrxteinMPNN."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tqdm

from prxteinmpnn.io.loaders import create_protein_dataset
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.training.checkpoint import restore_checkpoint, save_checkpoint
from prxteinmpnn.training.diffusion import NoiseSchedule
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
from prxteinmpnn.utils.sharding import create_mesh, shard_pytree

if TYPE_CHECKING:
  from chex import ArrayTree

  from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.training.specs import TrainingSpecification
  from prxteinmpnn.utils.types import BackboneCoordinates, Logits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_optimizer(
  spec: TrainingSpecification,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
  """Create AdamW optimizer with learning rate schedule.

  Args:
      spec: Training specification

  Returns:
      Tuple of (optimizer, learning_rate_schedule)

  """
  if spec.warmup_steps > 0:
    schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=spec.learning_rate,
      warmup_steps=spec.warmup_steps,
      decay_steps=spec.total_steps or (spec.num_epochs * 1000),
      end_value=spec.learning_rate * 0.1,
    )
  else:
    schedule = optax.constant_schedule(spec.learning_rate)

  optimizer = optax.chain(
    optax.clip_by_global_norm(spec.gradient_clip)
    if spec.gradient_clip is not None
    else optax.identity(),
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

  final_model: PrxteinMPNN
  final_step: int
  checkpoint_dir: str | Path


def _init_checkpoint_and_model(
  spec: TrainingSpecification,
) -> tuple[PrxteinMPNN, Any, int, ocp.CheckpointManager, ocp.CheckpointManager]:
  """Initialize or restore model, optimizer state and checkpoint manager.

  Applies physics encoder surgery if use_physics_features is enabled.

  Returns (model, opt_state, start_step, checkpoint_manager).
  """
  checkpoint_dir = Path(spec.checkpoint_dir)
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  options = ocp.CheckpointManagerOptions(max_to_keep=spec.keep_last_n_checkpoints)
  checkpoint_manager = ocp.CheckpointManager(
    checkpoint_dir,
    options=options,
  )

  # Permanent checkpoint manager for specific epochs
  permanent_checkpoint_dir = checkpoint_dir / "kept"
  permanent_checkpoint_dir.mkdir(parents=True, exist_ok=True)
  permanent_options = ocp.CheckpointManagerOptions(max_to_keep=None)  # Keep all
  permanent_manager = ocp.CheckpointManager(
    permanent_checkpoint_dir,
    options=permanent_options,
  )

  opt_state: ArrayTree | None = None
  if spec.resume_from_checkpoint:
    model_template = load_model(
      spec.model_version,  # type: ignore[invalid-argument-type]
      spec.model_weights,  # type: ignore[invalid-argument-type]
      use_electrostatics=spec.use_electrostatics,
      use_vdw=spec.use_vdw,
    )
    model, opt_state, _, start_step = restore_checkpoint(
      checkpoint_manager,
      model_template,
      step=None,  # Load latest
    )
    logger.info("Resumed from checkpoint at step %d", start_step)
  else:
    model = load_model(
      spec.model_version,  # type: ignore[invalid-argument-type]
      spec.model_weights,  # type: ignore[invalid-argument-type]
      use_electrostatics=spec.use_electrostatics,
      use_vdw=spec.use_vdw,
      training_mode=spec.training_mode,
    )
    start_step = 0
    optimizer_obj, _ = create_optimizer(spec)
    opt_state = optimizer_obj.init(eqx.filter(model, eqx.is_inexact_array))

  return model, opt_state, start_step, checkpoint_manager, permanent_manager


def _create_dataloaders(spec: TrainingSpecification) -> tuple[Any, Any]:
  """Create training and validation data loaders based on the spec."""
  train_loader = create_protein_dataset(
    spec.inputs,  # type: ignore[invalid-argument-type]
    batch_size=spec.batch_size,
    foldcomp_database=spec.foldcomp_database if not spec.use_preprocessed else None,
    use_electrostatics=spec.use_electrostatics,
    estat_noise=spec.estat_noise,
    estat_noise_mode=spec.estat_noise_mode,
    use_vdw=spec.use_vdw,
    vdw_noise=spec.vdw_noise,
    vdw_noise_mode=spec.vdw_noise_mode,
    use_preprocessed=spec.use_preprocessed,
    preprocessed_index_path=spec.preprocessed_index_path,
    split="train",
    max_length=spec.max_length,
    truncation_strategy=spec.truncation_strategy,
  )

  val_loader = None
  if spec.validation_data:
    val_use_preprocessed = spec.use_preprocessed
    val_inputs = spec.validation_data
    val_index_path = spec.validation_preprocessed_index_path

    if spec.validation_preprocessed_path is not None:
      val_use_preprocessed = True
      val_inputs = spec.validation_preprocessed_path
      if val_index_path is None:
        val_index_path = Path(val_inputs).with_suffix(".index.json")

    val_loader = create_protein_dataset(
      val_inputs,
      batch_size=spec.batch_size,
      foldcomp_database=spec.foldcomp_database if not val_use_preprocessed else None,
      use_preprocessed=val_use_preprocessed,
      use_electrostatics=spec.use_electrostatics,
      estat_noise=spec.estat_noise,
      estat_noise_mode=spec.estat_noise_mode,
      use_vdw=spec.use_vdw,
      vdw_noise=spec.vdw_noise,
      vdw_noise_mode=spec.vdw_noise_mode,
      preprocessed_index_path=val_index_path,
      split="valid",
      max_length=spec.max_length,
      truncation_strategy=spec.truncation_strategy,
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
    logger.info("Using BF16 mixed precision")
  else:
    logger.info("Using FP32 (full precision)")


def train_step(  # noqa: PLR0913
  model: PrxteinMPNN,
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
  physics_features: jax.Array | None = None,
  backbone_noise_std: float = 0.0,
  mask_strategy: str = "random_order",
  mask_prob: float = 0.15,
  training_mode: str = "autoregressive",
  training_mode: str = "autoregressive",
  noise_schedule: NoiseSchedule | None = None,
  full_coordinates: jax.Array | None = None,
  md_params: dict[str, jax.Array] | None = None,
  md_config: dict[str, float | int] | None = None,
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
      physics_features: Optional physics features (if used)
      backbone_noise_std: Standard deviation of Gaussian noise added to backbone coordinates
      mask_strategy: Strategy for autoregressive masking ("random_order" or "bert")
      mask_prob: Probability of masking a token if using "bert" strategy
      training_mode: "autoregressive" or "diffusion"
      noise_schedule: Noise schedule for diffusion training
      full_coordinates: Full atomic coordinates for MD
      md_params: MD parameters
      md_config: MD configuration

  Returns:
      Tuple of (updated_model, updated_opt_state, metrics)

  """
  batch_size = coordinates.shape[0]

  def loss_fn(model: PrxteinMPNN) -> tuple[jax.Array, jax.Array]:
    """Compute loss for current batch."""
    batch_keys = jax.random.split(prng_key, batch_size)

    def single_forward(
      coords: BackboneCoordinates,
      mask: jax.Array,
      res_idx: jax.Array,
      chain_idx: jax.Array,
      seq: jax.Array,
      key: jax.Array,
      phys_feat: jax.Array | None = None,
      full_coords: jax.Array | None = None,
      md_p: dict[str, jax.Array] | None = None,
    ) -> Logits:
      """Forward pass for a single protein."""
      key, subkey = jax.random.split(key)

      n_nodes = mask.shape[0]
      if mask_strategy == "random_order":
        decoding_order = jax.random.permutation(subkey, jnp.arange(n_nodes))
        ranks = jnp.argsort(decoding_order)
        ar_mask = ranks[None, :] < ranks[:, None]
      elif mask_strategy == "bert":
        mask_prob_mask = jax.random.bernoulli(subkey, mask_prob, shape=(n_nodes,))
        can_see = 1.0 - mask_prob_mask
        ar_mask = jnp.tile(can_see[None, :], (n_nodes, 1))
      else:
        ar_mask = jnp.ones((n_nodes, n_nodes))

      one_hot_seq = jax.nn.one_hot(seq, 21)

      if training_mode == "diffusion":
        if noise_schedule is None:
          msg = "noise_schedule required for diffusion training"
          raise ValueError(msg)

        t = jax.random.randint(subkey, (), 0, noise_schedule.num_steps)
        noise = jax.random.normal(subkey, one_hot_seq.shape)
        noisy_seq, _ = noise_schedule.sample_forward(one_hot_seq, t, noise)

        diff_model = cast("DiffusionPrxteinMPNN", model)
        _, logits = diff_model(
            coords,
            mask,
            res_idx,
            chain_idx,
            decoding_approach="diffusion",
            timestep=t,
            noisy_sequence=noisy_seq,
            physics_features=phys_feat,
            full_coordinates=full_coords,
            md_params=md_p,
            md_config=md_config,
        )
        return logits

      _, logits = model(
        coords,
        mask,
        res_idx,
        chain_idx,
        decoding_approach="conditional",
        prng_key=key,
        ar_mask=ar_mask,
        one_hot_sequence=one_hot_seq,
        backbone_noise=jnp.array(backbone_noise_std),
        initial_node_features=phys_feat,
        full_coordinates=full_coords,
        md_params=md_p,
        md_config=md_config,
      )
      return logits


    logits_batch = jax.vmap(single_forward)(
      coordinates,
      mask,
      residue_index,
      chain_index,
      sequence,
      batch_keys,
      physics_features,
      full_coordinates,
      md_params,
    )

    def batch_loss(logits: Logits, seq: jax.Array, msk: jax.Array) -> jax.Array:
      return cross_entropy_loss(logits, seq, msk, label_smoothing)

    losses = jax.vmap(batch_loss)(logits_batch, sequence, mask)
    loss = jnp.mean(losses)

    return loss, logits_batch

  (loss, logits_batch), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

  def batch_metrics(logits: Logits, seq: jax.Array, msk: jax.Array) -> tuple[jax.Array, jax.Array]:
    acc = sequence_recovery_accuracy(logits, seq, msk)
    ppl = perplexity(logits, seq, msk)
    return acc, ppl

  accuracies, perplexities = jax.vmap(batch_metrics)(logits_batch, sequence, mask)
  accuracy = jnp.mean(accuracies)
  ppl = jnp.mean(perplexities)
  grad_norm = compute_grad_norm(grads)
  current_lr = lr_schedule(current_step)

  params = eqx.filter(model, eqx.is_inexact_array)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)
  new_model = eqx.apply_updates(model, updates)

  metrics = TrainingMetrics(
    loss=loss,
    accuracy=accuracy,
    perplexity=ppl,
    learning_rate=current_lr,  # type: ignore[invalid-argument-type]
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
  physics_features: jax.Array | None = None,
  training_mode: str = "autoregressive",
  noise_schedule: NoiseSchedule | None = None,
  full_coordinates: jax.Array | None = None,
  md_params: dict[str, jax.Array] | None = None,
  md_config: dict[str, float | int] | None = None,
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
      physics_features: Optional physics features (if used)
      training_mode: "autoregressive" or "diffusion"
      noise_schedule: Noise schedule for diffusion training
      full_coordinates: Full atomic coordinates for MD
      md_params: MD parameters
      md_config: MD configuration

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
    seq: jax.Array,
    key: jax.Array,
    phys_feat: jax.Array | None = None,
    full_coords: jax.Array | None = None,
    md_p: dict[str, jax.Array] | None = None,
  ) -> Logits:
    """Forward pass for a single protein."""
    inference_model = eqx.nn.inference_mode(model)

    if training_mode == "diffusion":
        if noise_schedule is None:
          msg = "noise_schedule required for diffusion evaluation"
          raise ValueError(msg)

        t = jax.random.randint(key, (), 0, noise_schedule.num_steps)
        one_hot_seq = jax.nn.one_hot(seq, 21)
        noise = jax.random.normal(key, one_hot_seq.shape)
        noisy_seq, _ = noise_schedule.sample_forward(one_hot_seq, t, noise)

        diff_model = cast("DiffusionPrxteinMPNN", inference_model)
        _, logits = diff_model(
            coords,
            msk,
            res_idx,
            chain_idx,
            decoding_approach="diffusion",
            timestep=t,
            noisy_sequence=noisy_seq,
            physics_features=phys_feat,
            full_coordinates=full_coords,
            md_params=md_p,
            md_config=md_config,
        )
        return logits

    _, logits = inference_model(
      coords,
      msk,
      res_idx,
      chain_idx,
      decoding_approach="unconditional",
      prng_key=key,
      backbone_noise=jnp.array(0.0),
      initial_node_features=phys_feat,
      full_coordinates=full_coords,
      md_params=md_p,
      md_config=md_config,
    )
    return logits

  logits_batch = jax.vmap(single_forward)(
    coordinates,
    mask,
    residue_index,
    chain_index,
    sequence,
    batch_keys,
    physics_features,
    full_coordinates,
    md_params,
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


def train(spec: TrainingSpecification) -> TrainingResult:  # noqa: C901, PLR0912, PLR0915
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
  setup_mixed_precision(spec.precision)
  logger.info("Starting training with spec: %s", spec)

  optimizer, lr_schedule = create_optimizer(spec)

  model, opt_state, start_step, checkpoint_manager, permanent_manager = _init_checkpoint_and_model(
    spec,
  )

  train_loader, val_loader = _create_dataloaders(spec)

  step = start_step
  best_val_metric = float("inf")
  patience_counter = 0

  prng_key = jax.random.PRNGKey(spec.random_seed)
  noise_schedule = None
  if spec.training_mode == "diffusion":
      noise_schedule = NoiseSchedule(
          num_steps=spec.diffusion_num_steps,
          beta_start=spec.diffusion_beta_start,
          beta_end=spec.diffusion_beta_end,
          schedule_type=spec.diffusion_schedule_type,
      )

  logger.info("Starting training loop...")

  mesh = None
  if spec.use_sharding:
    mesh = create_mesh()

  def _get_md_args(batch):
      md_params = None
      md_config = None
      if spec.backbone_noise_mode == "md":
          md_params = {
              "bonds": batch.md_bonds,
              "bond_params": batch.md_bond_params,
              "angles": batch.md_angles,
              "angle_params": batch.md_angle_params,
              "backbone_indices": batch.md_backbone_indices,
              "exclusion_mask": batch.md_exclusion_mask,
              "charges": batch.charges,
              "sigmas": batch.sigmas,
              "epsilons": batch.epsilons,
          }
          md_config = {
              "temperature": spec.md_temperature,
              "min_steps": spec.md_min_steps,
              "therm_steps": spec.md_therm_steps,
          }
      return batch.full_coordinates, md_params, md_config

  for epoch in range(spec.num_epochs):
    logger.info("Epoch %d/%d", epoch + 1, spec.num_epochs)
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{spec.num_epochs}")

    for batch in train_loader:
      prng_key, subkey = jax.random.split(prng_key)

      if isinstance(spec.backbone_noise, (float, int)):
        backbone_noise_std = float(spec.backbone_noise)
      else:
        backbone_noise_std = float(spec.backbone_noise[0])

      if mesh is not None and spec.shard_batch:
        batch = shard_pytree(batch, mesh)  # noqa: PLW2901

      full_coords, md_p, md_c = _get_md_args(batch)

      if mesh is not None:
        with mesh:
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
            batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
            backbone_noise_std,
            spec.mask_strategy,
            spec.mask_prob,
            spec.training_mode,
            noise_schedule,
            full_coords,
            md_p,
            md_c,
          )
      else:
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
          batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
          backbone_noise_std,
          spec.mask_strategy,
          spec.mask_prob,
          spec.training_mode,
          noise_schedule,
          full_coords,
          md_p,
          md_c,
        )

      step += 1
      loss_float = jax.device_get(train_metrics.loss).item()
      pbar.set_postfix({"loss": loss_float})

      if val_loader and step % spec.eval_every == 0:
        val_metrics_list = []
        for val_batch in val_loader:
          prng_key, subkey = jax.random.split(prng_key)

          if mesh is not None and spec.shard_batch:
            val_batch = shard_pytree(val_batch, mesh)  # noqa: PLW2901

          val_full_coords, val_md_p, val_md_c = _get_md_args(val_batch)

          if mesh is not None:
            with mesh:
              val_metrics = eqx.filter_jit(eval_step)(
                model,
                val_batch.coordinates,
                val_batch.mask,
                val_batch.residue_index,
                val_batch.chain_index,
                val_batch.aatype,
                subkey,
                val_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
                spec.training_mode,
                noise_schedule,
                val_full_coords,
                val_md_p,
                val_md_c,
              )
          else:
            val_metrics = eqx.filter_jit(eval_step)(
              model,
              val_batch.coordinates,
              val_batch.mask,
              val_batch.residue_index,
              val_batch.chain_index,
              val_batch.aatype,
              subkey,
              val_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
              spec.training_mode,
              noise_schedule,
              val_full_coords,
              val_md_p,
              val_md_c,
            )
          val_metrics_list.append(val_metrics)

        avg_val_loss = jnp.mean(jnp.array([m.val_loss for m in val_metrics_list]))
        avg_val_acc = jnp.mean(jnp.array([m.val_accuracy for m in val_metrics_list]))

        val_loss_float = jax.device_get(avg_val_loss).item()
        val_acc_float = jax.device_get(avg_val_acc).item()

        logger.info(
          "Validation at step %d: val_loss=%.4f, val_acc=%.4f",
          step,
          val_loss_float,
          val_acc_float,
        )

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

      if step % spec.checkpoint_every == 0:
        save_checkpoint(
          checkpoint_manager,
          step,
          model,
          opt_state,
          metrics=train_metrics,
        )

      # Persistent checkpointing
      if spec.save_at_epochs and (epoch + 1) in spec.save_at_epochs:
        logger.info("Saving persistent checkpoint for epoch %d", epoch + 1)
        save_checkpoint(
          permanent_manager,
          step,
          model,
          opt_state,
          metrics=train_metrics,
        )

  logger.info("Training complete!")

  # Final Test Loop
  logger.info("Starting final test evaluation...")
  test_loader = None

  # Determine test data source
  test_inputs = spec.validation_data  # Default to validation data if no separate test set
  test_use_preprocessed = spec.use_preprocessed
  test_index_path = spec.validation_preprocessed_index_path

  # If we are using preprocessed data, we try to load the 'test' split from the same file
  # or a specific test file if one were added to spec
  if spec.use_preprocessed and spec.preprocessed_index_path:
    # If validation path is set, use that, otherwise fall back to training path
    test_inputs = spec.validation_preprocessed_path or spec.inputs

    test_index_path = spec.validation_preprocessed_index_path or spec.preprocessed_index_path

  try:
    test_loader = create_protein_dataset(
      test_inputs,  # type: ignore[invalid-argument-type]
      batch_size=spec.batch_size,
      foldcomp_database=spec.foldcomp_database if not test_use_preprocessed else None,
      use_preprocessed=test_use_preprocessed,
      use_electrostatics=spec.use_electrostatics,
      estat_noise=spec.estat_noise,
      estat_noise_mode=spec.estat_noise_mode,
      use_vdw=spec.use_vdw,
      vdw_noise=spec.vdw_noise,
      vdw_noise_mode=spec.vdw_noise_mode,
      preprocessed_index_path=test_index_path,
      split="test",
    )

    test_metrics_list = []
    for test_batch in tqdm.tqdm(test_loader, desc="Testing"):
      prng_key, subkey = jax.random.split(prng_key)

      if mesh is not None and spec.shard_batch:
        test_batch = shard_pytree(test_batch, mesh)  # noqa: PLW2901

      test_full_coords, test_md_p, test_md_c = _get_md_args(test_batch)

      if mesh is not None:
        with mesh:
          test_metrics = eqx.filter_jit(eval_step)(
            model,
            test_batch.coordinates,
            test_batch.mask,
            test_batch.residue_index,
            test_batch.chain_index,
            test_batch.aatype,
            subkey,
            test_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
            spec.training_mode,
            noise_schedule,
            test_full_coords,
            test_md_p,
            test_md_c,
          )
      else:
        test_metrics = eqx.filter_jit(eval_step)(
          model,
          test_batch.coordinates,
          test_batch.mask,
          test_batch.residue_index,
          test_batch.chain_index,
          test_batch.aatype,
          subkey,
          test_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
          spec.training_mode,
          noise_schedule,
          test_full_coords,
          test_md_p,
          test_md_c,
        )
      test_metrics_list.append(test_metrics)

    if test_metrics_list:
      avg_test_loss = jnp.mean(jnp.array([m.val_loss for m in test_metrics_list]))
      avg_test_acc = jnp.mean(jnp.array([m.val_accuracy for m in test_metrics_list]))
      avg_test_ppl = jnp.mean(jnp.array([m.val_perplexity for m in test_metrics_list]))

      logger.info("=" * 40)
      logger.info("Final Test Results:")
      logger.info("  Loss: %.4f", jax.device_get(avg_test_loss).item())
      logger.info("  Accuracy: %.4f", jax.device_get(avg_test_acc).item())
      logger.info("  Perplexity: %.4f", jax.device_get(avg_test_ppl).item())
      logger.info("=" * 40)
    else:
      logger.warning("Test loader was empty. No test metrics computed.")

  except Exception:  # noqa: BLE001
    logger.warning(
      "Could not create test loader or run testing (possibly no 'test' split found).",
    )

  checkpoint_manager.close()
  permanent_manager.close()

  return TrainingResult(final_model=model, final_step=step, checkpoint_dir=spec.checkpoint_dir)
