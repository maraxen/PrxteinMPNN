"""Tests for checkpoint management."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pytest

from prxteinmpnn.training.checkpoint import restore_checkpoint, save_checkpoint


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

    # Create dummy optimizer state
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))

    # Save checkpoint using helper function
    saved = save_checkpoint(
        manager=manager,
        step=100,
        model=small_model,
        opt_state=opt_state,
        metrics={"loss": 1.5, "accuracy": 0.8},
    )

    assert saved

    # Wait for async save to complete
    manager.wait_until_finished()

    # Check checkpoint was saved (Orbax uses step number as directory name)
    assert (temp_checkpoint_dir / "100").exists()

    manager.close()


def test_load_checkpoint(temp_checkpoint_dir, small_model):
    """Test loading a checkpoint."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)

    # Create and save a checkpoint
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))

    save_checkpoint(
        manager=manager,
        step=100,
        model=small_model,
        opt_state=opt_state,
        metrics={"loss": 1.5},
    )

    # Load checkpoint using helper function
    loaded_model, loaded_opt_state, loaded_metrics, loaded_step = restore_checkpoint(
        manager=manager,
        model_template=small_model,
        step=100,
    )

    assert loaded_step == 100
    assert loaded_metrics is not None
    assert loaded_metrics["loss"] == 1.5

    # Check that model parameters match using tree_map for proper leaf ordering
    original_params = eqx.filter(small_model, eqx.is_inexact_array)
    loaded_params = eqx.filter(loaded_model, eqx.is_inexact_array)

    # Use tree_map to ensure leaves are compared in the same order
    def _check_equal(orig, loaded):
      assert jnp.allclose(orig, loaded, rtol=1e-5, atol=1e-5)

    jax.tree_util.tree_map(_check_equal, original_params, loaded_params)

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

    # Save 5 checkpoints
    for step in [100, 200, 300, 400, 500]:
      save_checkpoint(manager, step, small_model, opt_state)

    # Wait for async operations to complete
    manager.wait_until_finished()

    # Check that only the last 2 exist (Orbax uses step numbers as directory names)
    all_steps = manager.all_steps()
    assert 400 in all_steps
    assert 500 in all_steps
    assert 100 not in all_steps
    assert 200 not in all_steps
    assert 300 not in all_steps

    # Should have exactly 2 checkpoints
    assert len(all_steps) == 2

    manager.close()


def test_checkpoint_with_no_metrics(temp_checkpoint_dir, small_model):
    """Test checkpointing without metrics."""
    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    manager = ocp.CheckpointManager(temp_checkpoint_dir, options=options)

    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(small_model, eqx.is_inexact_array))

    # Save checkpoint without metrics
    save_checkpoint(manager, 1, small_model, opt_state, metrics=None)

    # Load checkpoint
    loaded_model, loaded_opt_state, loaded_metrics, step = restore_checkpoint(
        manager=manager, model_template=small_model, step=1,
    )

    assert step == 1
    # When metrics=None is passed, we save an empty dict {}
    assert loaded_metrics == {}

    manager.close()
