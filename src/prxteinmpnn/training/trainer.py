"""Main training loop for PrxteinMPNN with Mixed Precision and Gradient Accumulation."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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

if TYPE_CHECKING:
  from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
  from prxteinmpnn.model.mpnn import PrxteinMPNN
  from prxteinmpnn.training.specs import TrainingSpecification

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=logging.StreamHandler(sys.stdout))


def get_compute_dtype(precision: str) -> jnp.dtype:
  """Map precision string to JAX dtype with hardware checks."""
  if precision == "bf16":
    logger.info("Using bfloat16 precision.")
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    logger.info("Set jax_default_matmul_precision to bfloat16.")
    return jnp.bfloat16
  if precision == "fp16":
    # Gate fp16 on hardware check to prevent issues on TPUs or older GPUs
    platform = jax.default_backend()
    if platform == "tpu":
      logger.warning("FP16 requested on TPU; falling back to bfloat16 for stability.")
      jax.config.update("jax_default_matmul_precision", "bfloat16")
      return jnp.bfloat16
    if platform == "gpu":
      gpu_info = jax.lib.xla_bridge.get_backend().device_description
      if "sm_" in gpu_info:
        sm_version = int(gpu_info.split("sm_")[1].split(" ")[0])
        if sm_version < 70:
          logger.warning(
            f"FP16 requested on GPU with SM {sm_version}; falling back to float32 for stability.",
          )
          return jnp.float32
      jax.config.update("jax_default_matmul_precision", "float16")
    return jnp.float16
  return jnp.float32


def create_optimizer(
  spec: TrainingSpecification,
) -> tuple[optax.GradientTransformation, optax.Schedule]:
  """Create AdamW optimizer with learning rate schedule."""
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
  """Container for results returned by :func:`train`."""

  final_model: PrxteinMPNN
  final_step: int
  checkpoint_dir: str | Path


def _init_checkpoint_and_model(
  spec: TrainingSpecification,
) -> tuple[PrxteinMPNN, Any, int, ocp.CheckpointManager, ocp.CheckpointManager]:
  """Initialize model, cast to precision, and setup checkpoints."""
  checkpoint_dir = Path(spec.checkpoint_dir)
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  options = ocp.CheckpointManagerOptions(max_to_keep=spec.keep_last_n_checkpoints)
  checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)

  permanent_checkpoint_dir = checkpoint_dir / "kept"
  permanent_checkpoint_dir.mkdir(parents=True, exist_ok=True)
  permanent_manager = ocp.CheckpointManager(
    permanent_checkpoint_dir,
    options=ocp.CheckpointManagerOptions(max_to_keep=None),
  )

  optimizer_obj, _ = create_optimizer(spec)

  # 1. Load Model
  model = load_model(
    spec.model_version,
    spec.model_weights,
    use_electrostatics=spec.use_electrostatics,
    use_vdw=spec.use_vdw,
    training_mode=spec.training_mode,
  )

  # 2. Cast Model to Target Precision
  # We only cast inexact arrays (weights/biases), keeping indices/masks intact.
  compute_dtype = get_compute_dtype(spec.precision)
  if compute_dtype != jnp.float32:
    logger.info(f"Casting model parameters to {compute_dtype}")

    def _cast_fn(x):
      return x.astype(compute_dtype) if eqx.is_inexact_array(x) else x

    model = jax.tree_util.tree_map(_cast_fn, model)
  else:
    logger.info("Keeping model in float32")

  # 3. Init Optimizer
  params = eqx.filter(model, eqx.is_inexact_array)
  opt_state = optimizer_obj.init(params)

  start_step = 0

  if spec.resume_from_checkpoint:
    resume_path = Path(spec.resume_from_checkpoint)

    # Check if resume path is a .eqx file (final model)
    if resume_path.suffix == ".eqx" and resume_path.exists():
      logger.info(f"Loading model weights from {resume_path}")
      try:
        # Create a float32 template to avoid dtype mismatch warnings during load
        # (The saved model may be in float32, but our template might be bfloat16)
        float32_model = load_model(
          spec.model_version,
          spec.model_weights,
          use_electrostatics=spec.use_electrostatics,
          use_vdw=spec.use_vdw,
          training_mode=spec.training_mode,
        )
        float32_params = eqx.filter(float32_model, eqx.is_inexact_array)

        # Load the saved weights into the float32 template
        loaded_params = eqx.tree_deserialise_leaves(resume_path, float32_params)
        loaded_model = eqx.combine(loaded_params, float32_model)

        # Now cast the loaded model to the target precision
        if compute_dtype != jnp.float32:
          logger.info(f"Casting loaded model from float32 to {compute_dtype}")

          def _cast_fn(x):
            return x.astype(compute_dtype) if eqx.is_inexact_array(x) else x

          loaded_model = jax.tree_util.tree_map(_cast_fn, loaded_model)

        model = loaded_model
        # Re-init optimizer with loaded params
        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optimizer_obj.init(params)
        logger.info("Successfully loaded model from .eqx file (starting from step 0)")
      except Exception as e:
        logger.error(f"Failed to load from {resume_path}: {e}")
        logger.error(
          "The saved model architecture doesn't match the current configuration. "
          "This usually means the model was saved with different hyperparameters. "
          "Either use the correct checkpoint or start fresh training.",
        )
        logger.info("Starting fresh training")
        # Continue with the freshly initialized model from above
    else:
      # Try to restore from Orbax checkpoint manager
      latest_step = checkpoint_manager.latest_step()
      if latest_step is not None:
        # We restore into the model template which already has the correct dtype
        model, opt_state, _, start_step = restore_checkpoint(
          checkpoint_manager,
          model_template=model,
          abstract_opt_state=opt_state,
          step=None,
        )
        logger.info("Resumed from checkpoint at step %d", start_step)
      else:
        logger.info("No checkpoint found, starting fresh training")

  return model, opt_state, start_step, checkpoint_manager, permanent_manager


def _create_dataloaders(spec: TrainingSpecification) -> tuple[Any, Any]:
  """Create training and validation data loaders."""
  # The spec.batch_size is interpreted as the Total Batch Size.
  # It must be divisible by accum_steps to ensure even micro-batches.
  assert spec.batch_size % spec.accum_steps == 0, (
    f"Batch size {spec.batch_size} must be divisible by accum_steps {spec.accum_steps}"
  )

  train_loader = create_protein_dataset(
    spec.inputs,
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
      batch_size=spec.batch_size,  # Validation usually doesn't need accumulation
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


def train_step(
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
  current_step: jax.Array,
  lr_schedule: optax.Schedule,
  accum_steps: int,
  compute_dtype: jnp.dtype,
  physics_features: jax.Array | None = None,
  backbone_noise_std: float = 0.0,
  mask_strategy: str = "random_order",
  mask_prob: float = 0.15,
  training_mode: str = "autoregressive",
  noise_schedule: NoiseSchedule | None = None,
) -> tuple[PrxteinMPNN, optax.OptState, TrainingMetrics]:
  """Single training step with Gradient Accumulation and Mixed Precision Pipeline."""
  total_batch_size = coordinates.shape[0]
  micro_batch_size = total_batch_size // accum_steps

  # 1. Reshape inputs for micro-batching: (TotalB, ...) -> (Accum, MicroB, ...)
  def reshape_to_micro(x):
    if x is None:
      return None
    return x.reshape((accum_steps, micro_batch_size, *x.shape[1:]))

  # 2. Input Casting: Ensure inputs match model dtype so ops stay in low precision
  def cast_input(x):
    if x is None:
      return None
    # Only cast float arrays; keep integers (indices/masks) as is
    return x.astype(compute_dtype) if eqx.is_inexact_array(x) else x

  # Reshape AND Cast
  micro_coords = cast_input(reshape_to_micro(coordinates))
  micro_mask = reshape_to_micro(mask)  # Keep as bool/int
  micro_res_idx = reshape_to_micro(residue_index)
  micro_chain_idx = reshape_to_micro(chain_index)
  micro_seq = reshape_to_micro(sequence)
  micro_phys = cast_input(reshape_to_micro(physics_features))

  # Pre-generate noise keys for each microbatch
  scan_keys = jax.random.split(prng_key, accum_steps)

  # 3. Define the micro-step (The main compute workload)
  def micro_step(carry, x):
    (accum_grads, accum_metrics) = carry
    (coords, msk, res_idx, chain_idx, seq, phys, key) = x

    # Ensure dynamic scalars are also the correct dtype
    noise_std = jnp.array(backbone_noise_std, dtype=compute_dtype)

    def loss_fn(p_model):
      # p_model is in compute_dtype. Inputs are in compute_dtype.
      batch_keys = jax.random.split(key, micro_batch_size)

      def single_forward(c, m, r, ch, s, p, k):
        k, subkey = jax.random.split(k)

        # Handling Autoregressive / BERT Masking Logic
        n_nodes = m.shape[0]

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

        # One-hot is float, must cast to compute_dtype
        one_hot_seq = jax.nn.one_hot(s, 21, dtype=compute_dtype)

        if training_mode == "diffusion":
          # Ensure diffusion noise is generated in compute_dtype
          t = jax.random.randint(subkey, (), 0, noise_schedule.num_steps)
          noise = jax.random.normal(subkey, one_hot_seq.shape, dtype=compute_dtype)
          noisy_seq, _ = noise_schedule.sample_forward(one_hot_seq, t, noise)

          diff_model = cast("DiffusionPrxteinMPNN", p_model)
          _, logits = diff_model(
            c,
            m,
            r,
            ch,
            decoding_approach="diffusion",
            timestep=t,
            noisy_sequence=noisy_seq,
            physics_features=p,
          )
          return logits

        # Autoregressive / Conditional
        _, logits = p_model(
          c,
          m,
          r,
          ch,
          decoding_approach="conditional",
          prng_key=k,
          ar_mask=ar_mask,
          one_hot_sequence=one_hot_seq,
          backbone_noise=noise_std,
          initial_node_features=p,
        )
        return logits

      logits_batch = jax.vmap(single_forward)(
        coords,
        msk,
        res_idx,
        chain_idx,
        seq,
        phys,
        batch_keys,
      )

      # --- CRITICAL STABILITY STEP ---
      # Cast logits back to Float32 for Loss Calculation
      logits_batch_f32 = logits_batch.astype(jnp.float32)

      losses = jax.vmap(lambda l, s, m: cross_entropy_loss(l, s, m, label_smoothing))(
        logits_batch_f32,
        seq,
        msk,
      )
      loss = jnp.mean(losses)

      # Compute Metrics in Float32
      accuracies = jax.vmap(sequence_recovery_accuracy)(logits_batch_f32, seq, msk)
      perplexities = jax.vmap(perplexity)(logits_batch_f32, seq, msk)

      metrics = (loss, jnp.mean(accuracies), jnp.mean(perplexities))
      return loss, metrics

    (loss, step_metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)

    # Cast Gradients to Float32 for Accumulation (Numerical Stability)
    grads = jax.tree.map(lambda g: g.astype(jnp.float32), grads)

    accum_grads = jax.tree.map(lambda a, g: a + g, accum_grads, grads)

    # Accumulate scalar metrics
    new_metrics = tuple(a + b for a, b in zip(accum_metrics, step_metrics))

    return (accum_grads, new_metrics), None

  # Initialize accumulation containers (Zeros in float32)
  zeros_grads = jax.tree.map(
    lambda p: jnp.zeros_like(p, dtype=jnp.float32),
    eqx.filter(model, eqx.is_inexact_array),
  )
  zeros_metrics = (0.0, 0.0, 0.0)  # Loss, Acc, Ppl

  # SCAN LOOP over microbatches
  (final_grads, final_sums), _ = jax.lax.scan(
    micro_step,
    (zeros_grads, zeros_metrics),
    (micro_coords, micro_mask, micro_res_idx, micro_chain_idx, micro_seq, micro_phys, scan_keys),
  )

  # Average the accumulated gradients and metrics
  grads = jax.tree.map(lambda g: g / accum_steps, final_grads)
  avg_loss = final_sums[0] / accum_steps
  avg_acc = final_sums[1] / accum_steps
  avg_ppl = final_sums[2] / accum_steps

  # Optimizer Step
  # grads are float32, model params are (likely) bfloat16.
  # Optax updates will happen using these float32 grads.
  params = eqx.filter(model, eqx.is_inexact_array)
  updates, new_opt_state = optimizer.update(grads, opt_state, params)

  # Apply updates to model (JAX handles cast back to param dtype if needed)
  new_model = eqx.apply_updates(model, updates)

  grad_norm = compute_grad_norm(grads)
  current_lr = lr_schedule(current_step)

  metrics = TrainingMetrics(
    loss=avg_loss,
    accuracy=avg_acc,
    perplexity=avg_ppl,
    learning_rate=current_lr,
    grad_norm=grad_norm,
  )

  return new_model, new_opt_state, metrics


def eval_step(
  model: PrxteinMPNN,
  coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  sequence: jax.Array,
  prng_key: jax.Array,
  compute_dtype: jnp.dtype,
  physics_features: jax.Array | None = None,
  training_mode: str = "autoregressive",
  noise_schedule: NoiseSchedule | None = None,
) -> EvaluationMetrics:
  """Single evaluation step with batching and correct casting."""
  batch_size = coordinates.shape[0]
  batch_keys = jax.random.split(prng_key, batch_size)

  # Cast inputs to compute_dtype for inference
  def cast_input(x):
    return x.astype(compute_dtype) if x is not None and eqx.is_inexact_array(x) else x

  coordinates = cast_input(coordinates)
  physics_features = cast_input(physics_features)

  def single_forward(c, m, res_idx, chain_idx, seq, key, p):
    inference_model = eqx.nn.inference_mode(model)

    if training_mode == "diffusion":
      t = jax.random.randint(key, (), 0, noise_schedule.num_steps)

      # Cast inputs inside the vmap if generated here
      one_hot_seq = jax.nn.one_hot(seq, 21, dtype=compute_dtype)
      noise = jax.random.normal(key, one_hot_seq.shape, dtype=compute_dtype)
      noisy_seq, _ = noise_schedule.sample_forward(one_hot_seq, t, noise)

      diff_model = cast("DiffusionPrxteinMPNN", inference_model)
      _, logits = diff_model(
        c,
        m,
        res_idx,
        chain_idx,
        decoding_approach="diffusion",
        timestep=t,
        noisy_sequence=noisy_seq,
        physics_features=p,
      )
      return logits

    _, logits = inference_model(
      c,
      m,
      res_idx,
      chain_idx,
      decoding_approach="unconditional",
      prng_key=key,
      backbone_noise=jnp.array(0.0, dtype=compute_dtype),
      initial_node_features=p,
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
  )

  # Cast back to FP32 for metrics
  logits_batch = logits_batch.astype(jnp.float32)

  def batch_metrics(l, s, m):
    val_loss = cross_entropy_loss(l, s, m, label_smoothing=0.0)
    val_accuracy = sequence_recovery_accuracy(l, s, m)
    val_ppl = perplexity(l, s, m)
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
  """Train PrxteinMPNN model."""
  compute_dtype = get_compute_dtype(spec.precision)
  logger.info(f"Starting training with spec: {spec}")
  logger.info(f"Compute precision: {compute_dtype}")

  optimizer, lr_schedule = create_optimizer(spec)

  model, opt_state, start_step, checkpoint_manager, permanent_manager = _init_checkpoint_and_model(
    spec,
  )

  train_loader, val_loader = _create_dataloaders(spec)

  step = np.array(start_step)
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
  loss_float = 1e4

  # Backbone noise: single value; if a tuple is provided we use the first entry
  # (A future improvement would vm ap over noise levels without duplicating data.)
  if isinstance(spec.backbone_noise, (float, int)):
    backbone_noise_std = float(spec.backbone_noise)
  else:
    backbone_noise_std = float(spec.backbone_noise[0])

  filter_jitted_train_step = eqx.filter_jit(train_step)
  filter_jitted_eval_step = eqx.filter_jit(eval_step)

  for epoch in range(spec.num_epochs):
    logger.info(f"Epoch {epoch + 1}/{spec.num_epochs}")
    train_iter = iter(train_loader)
    pbar = tqdm.tqdm(train_iter, desc=f"Epoch {epoch + 1}/{spec.num_epochs}")

    for batch in train_iter:
      # Skip incomplete batches that can't be evenly divided by (accum_steps * noise_levels)
      # These samples will appear in future epochs due to random sampling
      batch_size = batch.coordinates.shape[0]
      if batch_size % spec.accum_steps != 0:
        logger.warning(
          f"Skipping incomplete batch of size {batch_size} "
          f"(not divisible by accum_steps={spec.accum_steps})",
        )
        # Still check if we should run validation/checkpointing based on current step
        # before continuing to next batch
        should_validate = val_loader and step % spec.eval_every == 0
        should_checkpoint = int(step) % spec.checkpoint_every == 0

        if should_validate or should_checkpoint:
          # Perform validation if needed
          if should_validate:
            val_metrics_list = []
            val_iter = iter(val_loader)
            for val_batch in val_iter:
              prng_key, subkey = jax.random.split(prng_key)
              val_metrics = filter_jitted_eval_step(
                model,
                val_batch.coordinates,
                val_batch.mask,
                val_batch.residue_index,
                val_batch.chain_index,
                val_batch.aatype,
                subkey,
                compute_dtype,
                val_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
                spec.training_mode,
                noise_schedule,
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
              current_metric = avg_val_loss
              if current_metric < best_val_metric:
                best_val_metric = current_metric
                patience_counter = 0
              else:
                patience_counter += 1

              if patience_counter >= spec.early_stopping_patience:
                logger.info(f"Early stopping triggered at step {step}")
                break

          # Perform checkpointing if needed (use last known train_metrics if available)
          if should_checkpoint:
            # Use a dummy metrics object since we didn't train this step
            save_checkpoint(
              checkpoint_manager,
              step,
              model,
              opt_state,
              metrics=None,
            )

        continue

      prng_key, subkey = jax.random.split(prng_key)

      model, opt_state, train_metrics = filter_jitted_train_step(
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
        spec.accum_steps,
        compute_dtype,
        batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
        backbone_noise_std,
        spec.mask_strategy,
        spec.mask_prob,
        spec.training_mode,
        noise_schedule,
      )

      step += 1
      if step % 250 == 0:
        loss_float = jax.device_get(train_metrics.loss).item()
      pbar.set_postfix({"loss": loss_float, "step": step})

      if val_loader and step % spec.eval_every == 0:
        val_metrics_list = []
        val_iter = iter(val_loader)
        for val_batch in val_iter:
          prng_key, subkey = jax.random.split(prng_key)
          val_metrics = filter_jitted_eval_step(
            model,
            val_batch.coordinates,
            val_batch.mask,
            val_batch.residue_index,
            val_batch.chain_index,
            val_batch.aatype,
            subkey,
            compute_dtype,
            val_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
            spec.training_mode,
            noise_schedule,
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
          current_metric = avg_val_loss
          if current_metric < best_val_metric:
            best_val_metric = current_metric
            patience_counter = 0
          else:
            patience_counter += 1

          if patience_counter >= spec.early_stopping_patience:
            logger.info(f"Early stopping triggered at step {step}")
            break

      if int(step) % spec.checkpoint_every == 0:
        save_checkpoint(
          checkpoint_manager,
          step,
          model,
          opt_state,
          metrics=train_metrics,
        )

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

  # --- FINAL TEST LOOP ---
  logger.info("Starting final test evaluation...")
  test_loader = None

  # Determine test data source (logic: try explicit test split, fallback to validation inputs)
  test_inputs = spec.validation_data
  test_use_preprocessed = spec.use_preprocessed
  test_index_path = spec.validation_preprocessed_index_path

  if spec.use_preprocessed and spec.preprocessed_index_path:
    test_inputs = spec.validation_preprocessed_path or spec.inputs
    test_index_path = spec.validation_preprocessed_index_path or spec.preprocessed_index_path

  try:
    test_loader = create_protein_dataset(
      test_inputs,
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
    if test_loader:
      test_metrics_list = []
      for test_batch in tqdm.tqdm(test_loader, desc="Testing"):
        prng_key, subkey = jax.random.split(prng_key)
        test_metrics = filter_jitted_eval_step(
          model,
          test_batch.coordinates,
          test_batch.mask,
          test_batch.residue_index,
          test_batch.chain_index,
          test_batch.aatype,
          subkey,
          compute_dtype,
          test_batch.physics_features if (spec.use_electrostatics or spec.use_vdw) else None,
          spec.training_mode,
          noise_schedule,
        )
        test_metrics_list.append(test_metrics)

      avg_test_loss = jnp.mean(jnp.array([m.val_loss for m in test_metrics_list]))
      avg_test_acc = jnp.mean(jnp.array([m.val_accuracy for m in test_metrics_list]))

      logger.info("=" * 40)
      logger.info("Final Test Results:")
      logger.info("  Loss: %.4f", jax.device_get(avg_test_loss).item())
      logger.info("  Accuracy: %.4f", jax.device_get(avg_test_acc).item())
      logger.info("=" * 40)
    else:
      logger.warning("Test loader was empty.")
  except Exception as e:
    logger.warning("Could not run testing: %s", e)

  checkpoint_manager.close()
  permanent_manager.close()

  return TrainingResult(final_model=model, final_step=step, checkpoint_dir=spec.checkpoint_dir)
