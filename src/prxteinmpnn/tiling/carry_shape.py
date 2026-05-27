"""CarryShape — metadata for wave-axis carry without binding to traced values.

CarryShape is a lightweight descriptor for carry-bearing scans that holds only
shape and dtype information. Unlike CarrySpec (which holds an actual init value
that may contain traced JAX arrays), CarryShape defers materialization to the
mode class's __call__ method.

Used by AutoregressiveDecode and other mode classes that need to declare
carry metadata at class construction time without binding to a specific JAX
trace context.

Risk D-10 mitigation: Introduces a separate metadata struct, leaving CarrySpec
unchanged for planner Phase-0 consumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CarryShape:
    """Metadata for a carry-bearing scan axis (shape + dtype, no value).

    Attributes:
        name: Symbolic name of the carry (e.g., "sequence", "state").
        shape: Shape tuple for the carry array (e.g., (L,), (S, H)).
        dtype: Data type for the carry (e.g., jnp.int32, jnp.float32).
    """
    name: str
    shape: tuple[int, ...]
    dtype: Any  # jnp.dtype

    def materialize(self) -> jax.Array:
        """Materialize a zero-filled JAX array with this shape and dtype.

        Returns:
            A JAX array of zeros with the declared shape and dtype.
        """
        return jnp.zeros(self.shape, dtype=self.dtype)


__all__ = ["CarryShape"]
