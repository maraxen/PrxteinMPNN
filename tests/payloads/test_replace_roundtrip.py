"""Phase 3 PR1: scaffold tests for `eqx.tree_at`-style payload `replace` (no production payloads yet)."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


class DummyPayload(eqx.Module):
    """Minimal stand-in for roadmap §3.2 payloads — dynamic array + `static=True` int.

    `eqx.tree_at` only addresses **pytree leaves**; static fields are not leaves, so
    `replace` uses `tree_at` for `data` and reconstructs the module when `tag`
    changes (same split we will document on real payloads in PR2).
    """

    data: jax.Array
    tag: int = eqx.field(static=True)

    def replace(self, **kw: Any) -> DummyPayload:
        """Keyword-only updates."""
        out: DummyPayload = self
        for key, value in kw.items():
            if key not in ("data", "tag"):
                msg = f"DummyPayload.replace: unknown field {key!r}"
                raise TypeError(msg)
            if key == "data":
                out = eqx.tree_at(lambda s: s.data, out, value)
            else:
                out = DummyPayload(data=out.data, tag=int(value))
        return out


def test_replace_data_roundtrip() -> None:
    p = DummyPayload(data=jnp.zeros((2,)), tag=0)
    q = p.replace(data=jnp.array([1.0, 2.0]))
    assert p.tag == q.tag == 0
    np.testing.assert_array_equal(np.asarray(q.data), np.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.asarray(p.data), np.zeros((2,)))


def test_replace_static_tag() -> None:
    p = DummyPayload(data=jnp.ones((1,)), tag=1)
    q = p.replace(tag=42)
    np.testing.assert_array_equal(np.asarray(q.data), np.asarray(p.data))
    assert q.tag == 42


def test_replace_multiple_fields() -> None:
    p = DummyPayload(data=jnp.arange(3.0), tag=0)
    q = p.replace(data=jnp.array([7.0, 8.0, 9.0]), tag=3)
    assert q.tag == 3
    np.testing.assert_array_equal(np.asarray(q.data), np.array([7.0, 8.0, 9.0]))


def test_replace_unknown_field_raises() -> None:
    p = DummyPayload(data=jnp.zeros((1,)), tag=0)
    with pytest.raises(TypeError, match="unknown field 'nope'"):
        p.replace(nope=jnp.ones((1,)))


def test_replace_under_jit() -> None:
    p = DummyPayload(data=jnp.array([0.0, 1.0]), tag=7)

    @jax.jit
    def bump(x: DummyPayload) -> DummyPayload:
        return x.replace(data=x.data + 1.0)

    q = bump(p)
    np.testing.assert_array_equal(np.asarray(q.data), np.array([1.0, 2.0]))
    assert q.tag == 7


def test_tree_structure_preserved_after_data_replace() -> None:
    """Dynamic-only `replace` keeps the same PyTreeDef; static metadata lives in treedef."""
    p = DummyPayload(data=jnp.array([3.0]), tag=9)
    q = p.replace(data=jnp.array([4.0]))
    leaves_p, treedef_p = jax.tree_util.tree_flatten(p)
    leaves_q, treedef_q = jax.tree_util.tree_flatten(q)
    assert treedef_p == treedef_q
    assert len(leaves_p) == len(leaves_q) == 1
    np.testing.assert_array_equal(np.asarray(leaves_p[0]), np.array([3.0]))
    np.testing.assert_array_equal(np.asarray(leaves_q[0]), np.array([4.0]))
    np.testing.assert_array_equal(np.asarray(q.data), np.array([4.0]))


def test_static_change_updates_treedef_metadata() -> None:
    """Replacing `static=True` fields rebuilds the module; treedef may differ (expected)."""
    p = DummyPayload(data=jnp.array([3.0]), tag=9)
    q = p.replace(tag=1)
    _, treedef_p = jax.tree_util.tree_flatten(p)
    _, treedef_q = jax.tree_util.tree_flatten(q)
    assert treedef_p != treedef_q
    assert q.tag == 1
