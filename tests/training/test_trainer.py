"""Tests for the main training loop."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import equinox as eqx

from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import (
    create_optimizer,
    eval_step,
    setup_mixed_precision,
    train_step,
)


def test_create_optimizer_basic():
    """Test basic optimizer creation with warmup."""
    spec = TrainingSpecification(
        inputs="data/train/",
        num_epochs=5,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=100,
        total_steps=1000,
    )

    optimizer, schedule = create_optimizer(spec)

    assert optimizer is not None
    assert schedule is not None

    # Test schedule at different steps
    lr_at_0 = schedule(0)
    lr_at_50 = schedule(50)
    lr_at_100 = schedule(100)

    assert lr_at_0 == 0.0  # Warmup starts at 0
    assert 0.0 < lr_at_50 < spec.learning_rate  # During warmup
    assert lr_at_100 == spec.learning_rate  # At peak


def test_create_optimizer_no_warmup():
    """Test optimizer creation without warmup."""
    spec = TrainingSpecification(
        inputs="data/train/",
        num_epochs=5,
        learning_rate=1e-3,
        warmup_steps=0,
    )
    
    optimizer, schedule = create_optimizer(spec)
    
    # Should be constant schedule
    assert schedule(0) == spec.learning_rate
    assert schedule(100) == spec.learning_rate


def test_setup_mixed_precision():
    """Test mixed precision setup (should not raise)."""
    setup_mixed_precision("fp32")
    setup_mixed_precision("fp16")
    setup_mixed_precision("bf16")

@pytest.mark.slow("Takes a while on a CPU-only machine.")
def test_train_step_reduces_loss(small_model: eqx.Module, mock_batch) -> None:
    """Test that multiple training steps reduce loss."""
    optimizer, schedule = create_optimizer(
        TrainingSpecification(
            inputs="data/train/",
            num_epochs=5,
            learning_rate=1e-3,
            warmup_steps=0,
            total_steps=1000,
        )
    )
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))

    key = jax.random.PRNGKey(0)
    model = small_model
    losses: list = []

    # Run 10 training steps
    for step in range(10):
        key, subkey = jax.random.split(key)

        model, opt_state, metrics = train_step(
            model=model,
            opt_state=opt_state,
            optimizer=optimizer,
            coordinates=mock_batch.coordinates,
            mask=mock_batch.mask,
            residue_index=mock_batch.residue_index,
            chain_index=mock_batch.chain_index,
            sequence=mock_batch.aatype,
            prng_key=subkey,
            label_smoothing=0.0,
            current_step=step,
            lr_schedule=schedule,
        )
        # Extract loss value outside JIT context by converting to numpy
        losses.append(float(np.asarray(metrics.loss)))

    # Loss should decrease (overfitting to the single batch)
    assert losses[-1] < losses[0]


def test_eval_step_basic(small_model, mock_batch):
    """Test a single evaluation step."""
    key = jax.random.PRNGKey(0)

    metrics = eval_step(
        model=small_model,
        coordinates=mock_batch.coordinates,
        mask=mock_batch.mask,
        residue_index=mock_batch.residue_index,
        chain_index=mock_batch.chain_index,
        sequence=mock_batch.aatype,
        prng_key=key,
    )

    assert metrics.val_loss.shape == ()
    assert metrics.val_accuracy.shape == ()
    assert metrics.val_perplexity.shape == ()
    assert jnp.isfinite(metrics.val_loss)
    assert 0.0 <= metrics.val_accuracy <= 1.0


def test_eval_step_is_deterministic(small_model, mock_batch):
    """Test that evaluation is deterministic (no dropout)."""
    key = jax.random.PRNGKey(42)

    metrics1 = eval_step(
        model=small_model,
        coordinates=mock_batch.coordinates,
        mask=mock_batch.mask,
        residue_index=mock_batch.residue_index,
        chain_index=mock_batch.chain_index,
        sequence=mock_batch.aatype,
        prng_key=key,
    )

    metrics2 = eval_step(
        model=small_model,
        coordinates=mock_batch.coordinates,
        mask=mock_batch.mask,
        residue_index=mock_batch.residue_index,
        chain_index=mock_batch.chain_index,
        sequence=mock_batch.aatype,
        prng_key=key,
    )

    # Should be exactly the same
    assert metrics1.val_loss == metrics2.val_loss
    assert metrics1.val_accuracy == metrics2.val_accuracy