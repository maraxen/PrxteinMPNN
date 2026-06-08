"""Tests for safe_scan carry-bearing primitive."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from aminx.utils.safe_scan import safe_scan


def test_safe_scan_accumulates_carry() -> None:
    """Carry accumulates across steps."""

    def transition(carry, x):
        return carry + x, carry + x

    xs = jnp.array([1, 2, 3, 4])
    final_carry, ys = safe_scan(transition, xs, init=jnp.int32(0))

    assert int(final_carry) == 10
    assert ys.tolist() == [1, 3, 6, 10]


def test_safe_scan_output_shape_matches_input() -> None:
    """Output stacks one y per step; shape matches leading axis of xs."""

    def transition(carry, x):
        return carry, x * 2

    xs = jnp.ones((7, 4))
    _, ys = safe_scan(transition, xs, init=jnp.zeros(4))
    assert ys.shape == (7, 4)


def test_safe_scan_pytree_xs() -> None:
    """xs can be a pytree; transition receives one leaf-element per step."""

    def transition(carry, x):
        a, b = x
        return carry + a, a + b

    xs = (jnp.array([1, 2, 3]), jnp.array([10, 20, 30]))
    final_carry, ys = safe_scan(transition, xs, init=jnp.int32(0))

    assert ys.tolist() == [11, 22, 33]
    assert int(final_carry) == 6


def test_safe_scan_no_carry_mutation() -> None:
    """Scan does not mutate init carry."""
    init = jnp.zeros(3)

    def transition(carry, x):
        return carry + x, carry

    safe_scan(transition, jnp.ones((5, 3)), init=init)
    # init should be unchanged (JAX is functional)
    assert jnp.all(init == 0)


def test_safe_scan_empty_xs_raises() -> None:
    """Reject empty pytree."""
    with pytest.raises(ValueError, match="empty"):
        safe_scan(lambda c, x: (c, x), [], init=0)


def test_safe_scan_is_jit_compatible() -> None:
    """safe_scan result is identical inside and outside jit."""

    def transition(carry, x):
        return carry + x, carry + x

    xs = jnp.arange(5, dtype=jnp.float32)
    eager_carry, eager_ys = safe_scan(transition, xs, init=jnp.float32(0))

    jit_fn = jax.jit(
        lambda xs: safe_scan(transition, xs, init=jnp.float32(0))
    )
    jit_carry, jit_ys = jit_fn(xs)

    assert jnp.allclose(eager_carry, jit_carry)
    assert jnp.allclose(eager_ys, jit_ys)
