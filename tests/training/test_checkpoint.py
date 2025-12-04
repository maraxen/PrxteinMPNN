"""Tests for checkpoint management."""
import chex
import equinox as eqx
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest

from prxteinmpnn.training.checkpoint import restore_checkpoint, save_checkpoint
from prxteinmpnn.training.metrics import TrainingMetrics


def test_checkpoint_manager_initialization(temp_checkpoint_dir):
    """Test Orbax CheckpointManager initialization."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)
    assert manager.directory == temp_checkpoint_dir
    assert temp_checkpoint_dir.exists()
    manager.close()
def test_save_checkpoint(temp_checkpoint_dir, small_model):
    """Test saving a checkpoint."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.8),
        perplexity=jnp.array(5.0),
        learning_rate=1e-4,
    )
    saved = save_checkpoint(
        manager=manager,
        step=100,
        model=small_model,
        opt_state=opt_state,
        metrics=metrics,
    )
    assert saved
    manager.wait_until_finished()
    assert (temp_checkpoint_dir / "100").exists()
    manager.close()
def test_load_checkpoint(temp_checkpoint_dir, small_model):
    """Test loading a checkpoint."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.8),
        perplexity=jnp.array(5.0),
        learning_rate=1e-4,
    )
    save_checkpoint(
        manager=manager,
        step=100,
        model=small_model,
        opt_state=opt_state,
        metrics=metrics,
    )
    loaded_model, loaded_opt_state, loaded_metrics, loaded_step = restore_checkpoint(
        manager=manager,
        model_template=small_model,
        step=100,
    )
    assert loaded_step == 100
    assert loaded_metrics is not None
    assert loaded_metrics["loss"] == 1.5
    original_params = eqx.filter(small_model, eqx.is_inexact_array)
    loaded_params = eqx.filter(loaded_model, eqx.is_inexact_array)
    chex.assert_trees_all_close(original_params, loaded_params, rtol=1e-5, atol=1e-5)
    manager.close()
def test_load_nonexistent_checkpoint(temp_checkpoint_dir, small_model):
    """Test loading a checkpoint that doesn't exist."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)
    with pytest.raises(ValueError, match="No checkpoints found"):
        restore_checkpoint(manager=manager, model_template=small_model, step=None)
    manager.close()
def test_cleanup_old_checkpoints(temp_checkpoint_dir, small_model):
    """Test that old checkpoints are cleaned up by Orbax."""
    options = ocp.CheckpointManagerOptions(max_to_keep=2)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    metrics = TrainingMetrics(
        loss=jnp.array(1.5),
        accuracy=jnp.array(0.8),
        perplexity=jnp.array(5.0),
        learning_rate=1e-4,
    )
    for step in [100, 200, 300, 400, 500]:
        save_checkpoint(manager, step, small_model, opt_state, metrics)
    manager.wait_until_finished()
    all_steps = manager.all_steps()
    assert 400 in all_steps
    assert 500 in all_steps
    assert 100 not in all_steps
    assert 200 not in all_steps
    assert 300 not in all_steps
    assert len(all_steps) == 2
    manager.close()
def test_checkpoint_with_no_metrics(temp_checkpoint_dir, small_model):
    """Test checkpointing without metrics."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))
    save_checkpoint(manager, 1, small_model, opt_state, metrics=None)
    loaded_model, loaded_opt_state, loaded_metrics, step = restore_checkpoint(
        manager=manager,
        model_template=small_model,
        step=1,
    )
    assert step == 1
    assert loaded_metrics == {}
    manager.close()
