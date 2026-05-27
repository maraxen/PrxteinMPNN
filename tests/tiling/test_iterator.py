# tests/tiling/test_iterator.py
"""Tests for MapIterator and ScanIterator protocols + concrete implementations.

Covers:
- Protocol conformance via isinstance(concrete, RuntimeCheckableProtocol)
- Numerical equivalence: Vmap, SafeMap, JaxScan
- Treedef invariant (oracle REC-1): switching iterator type triggers re-JIT
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.tiling.iterator import (
    MapIterator,
    ScanIterator,
    VmapIterator,
    SafeMapIterator,
    JaxScanIterator,
)


# ============================================================================
# Protocol Conformance Tests
# ============================================================================


def test_vmap_iterator_is_map_iterator():
    """VmapIterator satisfies MapIterator protocol."""
    it = VmapIterator()
    assert isinstance(it, MapIterator)


def test_safe_map_iterator_is_map_iterator():
    """SafeMapIterator satisfies MapIterator protocol."""
    it = SafeMapIterator(tile=2)
    assert isinstance(it, MapIterator)


def test_jax_scan_iterator_is_scan_iterator():
    """JaxScanIterator satisfies ScanIterator protocol."""
    it = JaxScanIterator()
    assert isinstance(it, ScanIterator)


# ============================================================================
# Numerical Equivalence Tests — MapIterator
# ============================================================================


def test_vmap_iterator_numerical_equivalence():
    """VmapIterator correctly applies a function across elements."""
    it = VmapIterator()
    fn = lambda x: x * 2
    xs = jnp.arange(4)
    result = it(fn, xs)
    expected = jnp.array([0, 2, 4, 6])
    assert jnp.allclose(result, expected)


def test_vmap_iterator_with_in_axes():
    """VmapIterator respects in_axes argument."""
    it = VmapIterator()
    fn = lambda x, y: x + y
    xs = jnp.arange(4)
    ys = jnp.ones(4)
    # Default in_axes=0 vectorizes over first axis
    result = it(fn, xs, in_axes=(0, 0))
    expected = jnp.arange(4) + 1
    assert jnp.allclose(result, expected)


def test_safe_map_iterator_numerical_equivalence():
    """SafeMapIterator correctly applies a function across elements."""
    it = SafeMapIterator(tile=2)
    fn = lambda x: x * 2
    xs = jnp.arange(4)
    result = it(fn, xs)
    expected = jnp.array([0, 2, 4, 6])
    assert jnp.allclose(result, expected)


def test_safe_map_iterator_respects_tile():
    """SafeMapIterator uses tile parameter for chunking."""
    it = SafeMapIterator(tile=2)
    assert it.tile == 2
    # Verify it can be called with different tile sizes
    it_large = SafeMapIterator(tile=10)
    assert it_large.tile == 10


def test_safe_map_iterator_with_in_axes():
    """SafeMapIterator respects in_axes argument."""
    it = SafeMapIterator(tile=2)
    fn = lambda x, y: x + y
    xs = jnp.arange(4)
    ys = jnp.ones(4)
    result = it(fn, xs, in_axes=(0, 0))
    expected = jnp.arange(4) + 1
    assert jnp.allclose(result, expected)


# ============================================================================
# Numerical Equivalence Tests — ScanIterator
# ============================================================================


def test_jax_scan_iterator_numerical_equivalence():
    """JaxScanIterator correctly carries state through iterations."""
    it = JaxScanIterator()
    fn = lambda c, x: (c + x, c + x)  # Carry: accumulated sum, Output: running total
    init = jnp.array(0)
    xs = jnp.arange(4)
    final_carry, ys = it(fn, init, xs)
    # After processing [0,1,2,3]:
    # carry=0, x=0 -> (0, 0)
    # carry=0, x=1 -> (1, 1)
    # carry=1, x=2 -> (3, 3)
    # carry=3, x=3 -> (6, 6)
    assert final_carry == jnp.array(6)
    assert jnp.allclose(ys, jnp.array([0, 1, 3, 6]))


def test_jax_scan_iterator_with_complex_carry():
    """JaxScanIterator handles structured carry correctly."""
    it = JaxScanIterator()
    # Carry is a tuple (sum, product)
    def fn(carry, x):
        s, p = carry
        return (s + x, p * x), (s + x, p * x)

    init = (jnp.array(0), jnp.array(1))
    xs = jnp.array([1, 2, 3])
    final_carry, ys = it(fn, init, xs)

    # After processing [1, 2, 3]:
    # (0, 1), 1 -> (1, 1), (1, 1)
    # (1, 1), 2 -> (3, 2), (3, 2)
    # (3, 2), 3 -> (6, 6), (6, 6)
    expected_carry_sum = jnp.array(6)
    expected_carry_prod = jnp.array(6)
    assert final_carry[0] == expected_carry_sum
    assert final_carry[1] == expected_carry_prod


# ============================================================================
# Treedef Invariant Tests (oracle REC-1)
# ============================================================================


def test_vmap_vs_safe_map_different_treedef():
    """Different iterator types have different tree structures (triggers re-JIT).

    This is the **intended** behavior per oracle REC-1: switching iterator
    strategy changes the PyTree structure, which forces JAX to re-JIT the
    composed mode class.
    """
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class WrapperWithVmap:
        iterator: MapIterator = dataclasses.field(default_factory=VmapIterator)

    @dataclasses.dataclass(frozen=True)
    class WrapperWithSafeMap:
        iterator: MapIterator = dataclasses.field(default_factory=lambda: SafeMapIterator(tile=4))

    wrapper_vmap = WrapperWithVmap()
    wrapper_safe_map = WrapperWithSafeMap()

    vmap_struct = jax.tree.tree_structure(wrapper_vmap)
    safe_map_struct = jax.tree.tree_structure(wrapper_safe_map)

    # Structures should differ because VmapIterator() vs SafeMapIterator(tile=4)
    # have different PyTree flat_children
    assert vmap_struct != safe_map_struct, (
        "Iterator strategy change must alter tree structure to trigger re-JIT"
    )


def test_vmap_vs_jax_scan_different_treedef():
    """MapIterator vs ScanIterator have different tree structures."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class WrapperWithMap:
        iterator: MapIterator = dataclasses.field(default_factory=VmapIterator)

    @dataclasses.dataclass(frozen=True)
    class WrapperWithScan:
        iterator: ScanIterator = dataclasses.field(default_factory=JaxScanIterator)

    wrapper_map = WrapperWithMap()
    wrapper_scan = WrapperWithScan()

    map_struct = jax.tree.tree_structure(wrapper_map)
    scan_struct = jax.tree.tree_structure(wrapper_scan)

    assert map_struct != scan_struct, (
        "Map vs Scan iterator types must have different tree structures"
    )


def test_iterator_equality_by_value():
    """Iterators with same parameters are equal."""
    it1 = SafeMapIterator(tile=4)
    it2 = SafeMapIterator(tile=4)
    assert it1 == it2


def test_iterator_inequality_different_params():
    """Iterators with different parameters are not equal."""
    it1 = SafeMapIterator(tile=2)
    it2 = SafeMapIterator(tile=4)
    assert it1 != it2


def test_vmap_iterator_equality():
    """VmapIterator instances are always equal (no parameters)."""
    it1 = VmapIterator()
    it2 = VmapIterator()
    assert it1 == it2


def test_jax_scan_iterator_equality():
    """JaxScanIterator instances are always equal (no parameters)."""
    it1 = JaxScanIterator()
    it2 = JaxScanIterator()
    assert it1 == it2
