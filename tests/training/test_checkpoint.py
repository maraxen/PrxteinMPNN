"""Tests for Track I: checkpoint clean-break to xtrax PyTree format."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.training.types import ResumableState

from aminx.training.checkpoint import (
    get_checkpoint_manager,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture()
def checkpoint_dir(tmp_path: Path) -> Path:
    return tmp_path / "checkpoints"


@pytest.fixture()
def dummy_state() -> ResumableState:
    model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
    return ResumableState(
        step=jnp.int32(5),
        key=jax.random.PRNGKey(42),
        model=model,
        opt_state=None,
        extras={},
    )


def test_get_checkpoint_manager_creates_directory(checkpoint_dir: Path) -> None:
    manager = get_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    assert manager is not None
    manager.close()


def test_save_and_load_roundtrip(
    checkpoint_dir: Path,
    dummy_state: ResumableState,
) -> None:
    """Save a ResumableState and verify we get the same step back after loading."""
    manager = get_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    save_checkpoint(manager, dummy_state)
    manager.close()

    manager2 = get_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    restored = load_checkpoint(manager2, dummy_state, step=None)
    manager2.close()

    assert int(restored.step) == int(dummy_state.step)


def test_save_uses_state_step(
    checkpoint_dir: Path,
    dummy_state: ResumableState,
) -> None:
    """save_checkpoint extracts step from state.step, not a separate arg."""
    manager = get_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    save_checkpoint(manager, dummy_state)

    assert manager.latest_step() == int(dummy_state.step)
    manager.close()


def test_load_restores_model_weights(
    checkpoint_dir: Path,
    dummy_state: ResumableState,
) -> None:
    """Loaded model weights match saved weights leaf-by-leaf."""
    manager = get_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    save_checkpoint(manager, dummy_state)
    manager.close()

    manager2 = get_checkpoint_manager(checkpoint_dir, max_to_keep=3)
    restored = load_checkpoint(manager2, dummy_state, step=None)
    manager2.close()

    orig_leaves = jax.tree_util.tree_leaves(
        eqx.filter(dummy_state.model, eqx.is_array)
    )
    rest_leaves = jax.tree_util.tree_leaves(
        eqx.filter(restored.model, eqx.is_array)
    )
    for orig, rest in zip(orig_leaves, rest_leaves):
        assert jnp.allclose(orig, rest), "Model weights differ after round-trip"
