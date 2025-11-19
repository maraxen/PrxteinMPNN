"""Training step for Diffusion Model."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from prxteinmpnn.model.diffusion_mpnn import DiffusionPrxteinMPNN
from prxteinmpnn.training.diffusion import NoiseSchedule
from prxteinmpnn.training.losses import cross_entropy_loss, sequence_recovery_accuracy, perplexity
from prxteinmpnn.utils.types import Logits, TrainingMetrics


@eqx.filter_jit
def train_step(
    model: DiffusionPrxteinMPNN,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    coordinates: jax.Array,
    mask: jax.Array,
    residue_index: jax.Array,
    chain_index: jax.Array,
    sequence: jax.Array,  # [B, N] integer labels
    prng_key: jax.Array,
    noise_schedule: NoiseSchedule,
    lr_schedule: optax.Schedule,
    current_step: int,
    physics_features: jax.Array | None = None,
    physics_noise_scale: float = 0.0,
) -> tuple[DiffusionPrxteinMPNN, optax.OptState, TrainingMetrics]:
    """Single training step for the Diffusion Model."""
    batch_size = coordinates.shape[0]
    num_classes = 21

    # Convert integer labels to one-hot for diffusion [B, N, 21]
    seq_one_hot = jax.nn.one_hot(sequence, num_classes)

    def loss_fn(model: DiffusionPrxteinMPNN) -> tuple[jax.Array, jax.Array]:
        """Compute diffusion loss for the current batch."""
        # 1. Sample random timesteps t for each item in the batch
        keys = jax.random.split(prng_key, 2)
        t_key, noise_key = keys[0], keys[1]
        
        timesteps = jax.random.randint(
            t_key,
            shape=(batch_size,),
            minval=0,
            maxval=noise_schedule.num_steps,
        )

        # 2. Create noisy sequences (x_t)
        noise = jax.random.normal(noise_key, shape=seq_one_hot.shape)

        # vmap sample_forward over batch
        noisy_sequence, _ = jax.vmap(noise_schedule.sample_forward)(
            seq_one_hot,
            timesteps,
            noise,
        )

        # 3. Get model prediction (denoise)
        model_keys = jax.random.split(noise_key, batch_size)

        def single_forward(
            coords,
            msk,
            res_idx,
            chain_idx,
            noisy_seq,
            t,
            key,
            phys_feat,
        ):
            # We use decoding_approach="diffusion" which triggers the subclass logic
            _, logits = model(
                coords,
                msk,
                res_idx,
                chain_idx,
                decoding_approach="diffusion",
                prng_key=key,
                physics_features=phys_feat,
                timestep=t,
                noisy_sequence=noisy_seq,
                physics_noise_scale=physics_noise_scale,
            )
            return logits

        logits_batch = jax.vmap(single_forward)(
            coordinates,
            mask,
            residue_index,
            chain_index,
            noisy_sequence,
            timesteps,
            model_keys,
            physics_features,
        )

        # 4. Calculate loss
        # Compare the predicted logits (for x_0) with the original clean sequence
        def batch_loss(logits: Logits, seq: jax.Array, msk: jax.Array) -> jax.Array:
            return cross_entropy_loss(logits, seq, msk, 0.0)

        losses = jax.vmap(batch_loss)(logits_batch, sequence, mask)
        loss = jnp.mean(losses)

        return loss, logits_batch

    (loss, logits_batch), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Metrics
    def batch_metrics(logits: Logits, seq: jax.Array, msk: jax.Array) -> tuple[jax.Array, jax.Array]:
        acc = sequence_recovery_accuracy(logits, seq, msk)
        ppl = perplexity(logits, seq, msk)
        return acc, ppl

    accuracies, perplexities = jax.vmap(batch_metrics)(logits_batch, sequence, mask)
    
    metrics = {
        "loss": loss,
        "accuracy": jnp.mean(accuracies),
        "perplexity": jnp.mean(perplexities),
        "learning_rate": lr_schedule(current_step),
    }

    return new_model, new_opt_state, metrics

