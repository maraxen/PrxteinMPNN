"""Integration tests for trainer with checkpoint functionality."""
import tempfile
from pathlib import Path
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest
from prxteinmpnn.training.checkpoint import restore_checkpoint, save_checkpoint
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import create_optimizer
from prxteinmpnn.training.metrics import TrainingMetrics
def test_checkpoint_integration_with_trainer_setup(small_model, temp_checkpoint_dir):
    """Test that checkpoint setup works as trainer.py would use it."""
    spec = TrainingSpecification(
        inputs="data/train/",
        checkpoint_dir=str(temp_checkpoint_dir),
        keep_last_n_checkpoints=3,
        num_epochs=1,
        learning_rate=1e-4,
        warmup_steps=100,
        total_steps=1000,
    )
    optimizer, lr_schedule = create_optimizer(spec)
    checkpoint_dir = Path(spec.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=spec.keep_last_n_checkpoints)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    assert checkpoint_manager is not None
    assert opt_state is not None
    checkpoint_manager.close()
def test_checkpoint_save_and_restore_integration(small_model, temp_checkpoint_dir):
    """Test checkpoint save/restore as trainer.py would use it."""
    spec = TrainingSpecification(
        inputs="data/train/",
        checkpoint_dir=str(temp_checkpoint_dir),
        keep_last_n_checkpoints=3,
        num_epochs=1,
        learning_rate=1e-4,
        warmup_steps=100,
        total_steps=1000,
    )
    optimizer, lr_schedule = create_optimizer(spec)
    checkpoint_dir = Path(spec.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=spec.keep_last_n_checkpoints)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    model = small_model
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.8),
        perplexity=jnp.array(5.0),
        learning_rate=1e-4,
    )
    saved = save_checkpoint(
        checkpoint_manager,
        step=100,
        model=model,
        opt_state=opt_state,
        metrics=metrics,
    )
    assert saved
    checkpoint_manager.wait_until_finished()
    restored_model, restored_opt_state, restored_metrics, restored_step = restore_checkpoint(
        checkpoint_manager,
        model_template=small_model,
        step=None,
    )
    assert restored_step == 100
    assert restored_metrics["loss"] == 1.5
    original_params = eqx.filter(small_model, eqx.is_inexact_array)
    restored_params = eqx.filter(restored_model, eqx.is_inexact_array)
    def _check_equal(orig, loaded):
        assert jnp.allclose(orig, loaded, rtol=1e-5, atol=1e-5)
    jax.tree_util.tree_map(_check_equal, original_params, restored_params)
    checkpoint_manager.close()
def test_multiple_checkpoints_cleanup(small_model, temp_checkpoint_dir):
    """Test that old checkpoints are cleaned up as trainer saves multiple checkpoints."""
    spec = TrainingSpecification(
        inputs="data/train/",
        checkpoint_dir=str(temp_checkpoint_dir),
        keep_last_n_checkpoints=2,
        num_epochs=1,
        learning_rate=1e-4,
        warmup_steps=100,
        total_steps=1000,
    )
    optimizer, _ = create_optimizer(spec)
    checkpoint_dir = Path(spec.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=spec.keep_last_n_checkpoints)
    checkpoint_manager = ocp.CheckpointManager(checkpoint_dir, options=options)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    metrics = TrainingMetrics(
        loss=jnp.array(2.0),
        accuracy=jnp.array(0.7),
        perplexity=jnp.array(6.0),
        learning_rate=1e-4,
    )
    steps = [100, 200, 300, 400, 500]
    for step in steps:
        save_checkpoint(
            checkpoint_manager,
            step=step,
            model=small_model,
            opt_state=opt_state,
            metrics=metrics,
        )
    checkpoint_manager.wait_until_finished()
    all_steps = checkpoint_manager.all_steps()
    assert 400 in all_steps
    assert 500 in all_steps
    assert 100 not in all_steps
    assert 200 not in all_steps
    assert 300 not in all_steps
    assert len(all_steps) == 2
    checkpoint_manager.close()
