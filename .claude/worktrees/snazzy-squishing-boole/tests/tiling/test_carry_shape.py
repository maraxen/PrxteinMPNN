"""Tests for CarryShape metadata struct.

CarryShape holds metadata about a carry-bearing scan without binding to a
traced JAX value. This is used by mode classes (e.g., AutoregressiveDecode)
to declare wave-axis carry shape while deferring materialization to __call__.
"""
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.tiling.carry import CarrySpec
from prxteinmpnn.tiling.carry_shape import CarryShape


class TestCarryShapeBasics:
    """Basic structure and equality tests."""

    def test_frozen_dataclass(self) -> None:
        """CarryShape is a frozen dataclass."""
        carry = CarryShape(name="sequence", shape=(20,), dtype=jnp.int32)
        with pytest.raises(AttributeError):
            carry.name = "modified"  # type: ignore

    def test_equality_by_value(self) -> None:
        """Two CarryShape instances with same values are equal."""
        c1 = CarryShape(name="seq", shape=(10,), dtype=jnp.float32)
        c2 = CarryShape(name="seq", shape=(10,), dtype=jnp.float32)
        assert c1 == c2

    def test_inequality_different_name(self) -> None:
        """Different names → inequality."""
        c1 = CarryShape(name="seq", shape=(10,), dtype=jnp.float32)
        c2 = CarryShape(name="other", shape=(10,), dtype=jnp.float32)
        assert c1 != c2

    def test_inequality_different_shape(self) -> None:
        """Different shapes → inequality."""
        c1 = CarryShape(name="seq", shape=(10,), dtype=jnp.float32)
        c2 = CarryShape(name="seq", shape=(20,), dtype=jnp.float32)
        assert c1 != c2

    def test_inequality_different_dtype(self) -> None:
        """Different dtypes → inequality."""
        c1 = CarryShape(name="seq", shape=(10,), dtype=jnp.float32)
        c2 = CarryShape(name="seq", shape=(10,), dtype=jnp.int32)
        assert c1 != c2


class TestCarryShapeMaterialize:
    """Tests for the materialize() method."""

    def test_materialize_1d_int32(self) -> None:
        """materialize() returns zeros array with correct shape and dtype."""
        carry = CarryShape(name="sequence", shape=(20,), dtype=jnp.int32)
        result = carry.materialize()
        assert isinstance(result, jax.Array)
        assert result.shape == (20,)
        assert result.dtype == jnp.int32
        assert jnp.all(result == 0)

    def test_materialize_1d_float32(self) -> None:
        """materialize() works with float32."""
        carry = CarryShape(name="state", shape=(5,), dtype=jnp.float32)
        result = carry.materialize()
        assert result.shape == (5,)
        assert result.dtype == jnp.float32
        assert jnp.allclose(result, 0.0)

    def test_materialize_2d(self) -> None:
        """materialize() handles multidimensional shapes."""
        carry = CarryShape(name="matrix", shape=(3, 4), dtype=jnp.float32)
        result = carry.materialize()
        assert result.shape == (3, 4)
        assert result.dtype == jnp.float32
        assert jnp.allclose(result, 0.0)

    def test_materialize_scalar(self) -> None:
        """materialize() can create scalar (shape=())."""
        carry = CarryShape(name="counter", shape=(), dtype=jnp.int32)
        result = carry.materialize()
        assert result.shape == ()
        assert result.dtype == jnp.int32
        assert result == 0


class TestCarryShapeDistinctness:
    """Risk D-10 mitigation: CarryShape and CarrySpec are distinct types."""

    def test_carry_shape_and_carry_spec_are_different_types(self) -> None:
        """CarryShape and CarrySpec are separate classes."""
        assert CarryShape is not CarrySpec
        assert type(CarryShape).__name__ == "type"
        assert type(CarrySpec).__name__ == "type"

    def test_carry_shape_fields_differ_from_carry_spec(self) -> None:
        """CarryShape and CarrySpec have different field sets."""
        carry_shape_fields = set(CarryShape.__dataclass_fields__.keys())
        carry_spec_fields = set(CarrySpec.__dataclass_fields__.keys())
        assert carry_shape_fields != carry_spec_fields
        # CarryShape should have: name, shape, dtype
        # CarrySpec should have: axis_name, init, transition, ordered_sinks
        assert carry_shape_fields == {"name", "shape", "dtype"}
        assert carry_spec_fields == {"axis_name", "init", "transition", "ordered_sinks"}
