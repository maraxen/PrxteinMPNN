"""Iterator protocols and concrete implementations for axis iteration.

Three iterator strategies control how a mapped axis is iterated:
- VmapIterator: jax.vmap — fully parallel, stateless.
- SafeMapIterator: safe_map with tiling — memory-bounded, stateless.
- JaxScanIterator: jax.lax.scan — carry-bearing, sequential.

MapIterator and ScanIterator are runtime_checkable Protocols defining the
two fundamental iteration patterns: stateless (MapIterator) and carry-bearing
(ScanIterator).

Pattern 5 note: Concrete iterators are eqx.Module instances, NOT marked
@runtime_checkable. The protocols (MapIterator, ScanIterator) are the types
that are @runtime_checkable; users check isinstance(concrete, Protocol).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.lax
import jax.numpy as jnp

from prxteinmpnn.utils.safe_map import safe_map


@runtime_checkable
class MapIterator(Protocol):
    """Stateless axis iteration protocol.

    Maps a function over an axis without carrying state. Signature:
        fn: Callable — function to apply per-element
        xs: Any — pytree of arrays; first axis will be iterated
        in_axes: Any — specifies which axes to iterate over (default: 0)

    Returns: ys where tree_structure(ys) == tree_structure(xs) but with
    the mapped axis consumed.
    """

    def __call__(
        self, fn: Any, xs: Any, *, in_axes: Any = 0
    ) -> Any:
        """Apply fn over the first (or specified) axis of xs.

        Args:
            fn: Callable to apply per-element.
            xs: Input pytree; iteration happens over axis 0 (or in_axes).
            in_axes: Axis specification (default 0).

        Returns:
            Output pytree with iterated axis consumed.
        """


@runtime_checkable
class ScanIterator(Protocol):
    """Carry-bearing axis iteration protocol.

    Scans a function over an axis while maintaining and threading a carry value.
    Signature:
        fn: Callable — (carry, x) -> (new_carry, y)
        init: Any — initial carry value
        xs: Any — pytree of arrays; first axis will be scanned

    Returns: (final_carry, ys) where ys has the scanned axis consumed.
    """

    def __call__(self, fn: Any, init: Any, xs: Any) -> tuple[Any, Any]:
        """Apply fn over the first axis of xs while threading a carry.

        Args:
            fn: Callable with signature (carry, x) -> (new_carry, y).
            init: Initial carry value.
            xs: Input pytree; scan happens over axis 0.

        Returns:
            (final_carry, ys) where ys has the scanned axis consumed.
        """


class VmapIterator(eqx.Module):
    """Stateless iterator via jax.vmap.

    Maps a function over an axis by fully parallelizing it. All elements
    are materialized simultaneously. Use when memory budget allows and
    elements are independent (no cross-talk required).
    """

    def __call__(
        self, fn: Any, xs: Any, *, in_axes: Any = 0
    ) -> Any:
        """Apply fn over the specified axis via jax.vmap.

        Args:
            fn: Callable to vmap over.
            xs: Input pytree.
            in_axes: Axis specification for vmap (default 0).

        Returns:
            Output pytree with vmapped axis consumed.
        """
        return jax.vmap(fn, in_axes=in_axes)(xs)


class SafeMapIterator(eqx.Module):
    """Stateless iterator via safe_map with tiling.

    Maps a function over an axis in tiles for memory efficiency. Elements
    are processed in tiles of `tile` elements at a time. No carry state;
    elements are independent. Use for memory-constrained axes where vmap
    would OOM.

    Attributes:
        tile: Number of elements to process in each tile (static).
    """

    tile: int = eqx.field(static=True)

    def __call__(
        self, fn: Any, xs: Any, *, in_axes: Any = 0
    ) -> Any:
        """Apply fn over the specified axis via safe_map with tiling.

        Args:
            fn: Callable to map over.
            xs: Input pytree.
            in_axes: Axis specification for safe_map (default 0).

        Returns:
            Output pytree with mapped axis consumed (tiled).
        """
        return safe_map(fn, xs, in_axes=in_axes, tile=self.tile)


class JaxScanIterator(eqx.Module):
    """Carry-bearing iterator via jax.lax.scan.

    Scans a function over an axis while threading a carry value. Use when
    cross-element communication is needed or carry-state is required. The
    carry must have static shape across all iterations.
    """

    def __call__(self, fn: Any, init: Any, xs: Any) -> tuple[Any, Any]:
        """Apply fn over axis 0 of xs via jax.lax.scan with carry.

        Args:
            fn: Callable with signature (carry, x) -> (new_carry, y).
            init: Initial carry value (must have static shape).
            xs: Input pytree; scan happens over axis 0.

        Returns:
            (final_carry, ys) where ys has the scanned axis consumed.
        """
        return jax.lax.scan(fn, init, xs)


__all__ = [
    "MapIterator",
    "ScanIterator",
    "VmapIterator",
    "SafeMapIterator",
    "JaxScanIterator",
]
