"""Tests for ResumableState adoption in trainer.py."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from xtrax.training.types import ResumableState


@pytest.fixture()
def dummy_state() -> ResumableState:
    """Minimal ResumableState with a one-layer linear model."""
    model = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))
    return ResumableState(
        step=jnp.int32(0),
        key=jax.random.PRNGKey(1),
        model=model,
        opt_state=None,
        extras={},
    )


def test_resumable_state_construction(dummy_state: ResumableState) -> None:
    assert int(dummy_state.step) == 0
    assert dummy_state.model is not None
    assert dummy_state.extras == {}


def test_resumable_state_step_update(dummy_state: ResumableState) -> None:
    updated = ResumableState(
        step=jnp.int32(int(dummy_state.step) + 1),
        key=dummy_state.key,
        model=dummy_state.model,
        opt_state=dummy_state.opt_state,
        extras=dummy_state.extras,
    )
    assert int(updated.step) == 1


def test_resumable_state_is_pytree(dummy_state: ResumableState) -> None:
    leaves, treedef = jax.tree_util.tree_flatten(dummy_state)
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert int(reconstructed.step) == int(dummy_state.step)


def test_resumable_state_jit_compatible(dummy_state: ResumableState) -> None:
    @eqx.filter_jit
    def identity(state: ResumableState) -> jax.Array:
        return state.step

    result = identity(dummy_state)
    assert int(result) == 0
