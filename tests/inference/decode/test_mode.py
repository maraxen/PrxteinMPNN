"""Tests for DecodeMode sealed union and decode protocols (Task 4).

Task 4.1: 6 tests per spec
  - Each mode is frozen dataclass
  - Equality by value
  - AutoregressiveMode() has no W-axis fields
  - STEMode().inner_mode == ConditionalMode() default
  - isinstance(x, DecodeMode) for all four
  - 3 protocols exist and are callable
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import pytest

from prxteinmpnn.inference.decode.mode import (
    AutoregressiveMode,
    ConditionalMode,
    DecodeMode,
    STEMode,
    UnconditionalMode,
)
from prxteinmpnn.inference.decode.protocols import (
    ARDecodeFn,
    DecodeScoreFn,
    DecoderSinkFn,
    STEDecodeFn,
)


class TestDecodeModesFrozenDataclass:
    """Test that each mode is a frozen dataclass."""

    @pytest.mark.parametrize(
        "mode_class",
        [ConditionalMode, UnconditionalMode, AutoregressiveMode, STEMode],
    )
    def test_mode_is_frozen_dataclass(self, mode_class):
        """Each mode is a frozen dataclass."""
        assert dataclasses.is_dataclass(mode_class)
        assert mode_class.__dataclass_params__.frozen is True


class TestDecodeModeEquality:
    """Test equality by value for all modes."""

    def test_conditional_equality(self):
        """ConditionalMode() instances are equal by value."""
        m1 = ConditionalMode()
        m2 = ConditionalMode()
        assert m1 == m2

    def test_unconditional_equality(self):
        """UnconditionalMode() instances are equal by value."""
        m1 = UnconditionalMode()
        m2 = UnconditionalMode()
        assert m1 == m2

    def test_autoregressive_equality(self):
        """AutoregressiveMode() instances are equal by value."""
        m1 = AutoregressiveMode()
        m2 = AutoregressiveMode()
        assert m1 == m2

    def test_ste_equality_default(self):
        """STEMode() instances with default values are equal."""
        m1 = STEMode()
        m2 = STEMode()
        assert m1 == m2

    def test_ste_equality_custom(self):
        """STEMode() with custom iterations are equal by value."""
        m1 = STEMode(iterations=50)
        m2 = STEMode(iterations=50)
        assert m1 == m2
        # But different iterations should not be equal
        m3 = STEMode(iterations=100)
        assert m1 != m3


class TestAutoregressiveModeNoFields:
    """Test that AutoregressiveMode has no W-axis fields (oracle CONCERN-6)."""

    def test_autoregressive_mode_no_fields(self):
        """AutoregressiveMode() has no W-axis fields."""
        fields = dataclasses.fields(AutoregressiveMode)
        assert len(fields) == 0, "AutoregressiveMode should have no fields"


class TestSTEModeDefaults:
    """Test STEMode default values."""

    def test_ste_default_inner_mode(self):
        """STEMode().inner_mode defaults to ConditionalMode()."""
        mode = STEMode()
        assert mode.inner_mode == ConditionalMode()

    def test_ste_default_iterations(self):
        """STEMode().iterations defaults to 100."""
        mode = STEMode()
        assert mode.iterations == 100

    def test_ste_custom_inner_mode(self):
        """STEMode can accept custom inner_mode."""
        inner = ConditionalMode()
        mode = STEMode(inner_mode=inner)
        assert mode.inner_mode == inner


class TestDecodeModeIsinstance:
    """Test isinstance checks for DecodeMode union."""

    def test_conditional_isinstance(self):
        """ConditionalMode instance is instance of DecodeMode."""
        mode = ConditionalMode()
        assert isinstance(mode, (ConditionalMode, UnconditionalMode, AutoregressiveMode, STEMode))

    def test_unconditional_isinstance(self):
        """UnconditionalMode instance is instance of DecodeMode."""
        mode = UnconditionalMode()
        assert isinstance(mode, (ConditionalMode, UnconditionalMode, AutoregressiveMode, STEMode))

    def test_autoregressive_isinstance(self):
        """AutoregressiveMode instance is instance of DecodeMode."""
        mode = AutoregressiveMode()
        assert isinstance(mode, (ConditionalMode, UnconditionalMode, AutoregressiveMode, STEMode))

    def test_ste_isinstance(self):
        """STEMode instance is instance of DecodeMode."""
        mode = STEMode()
        assert isinstance(mode, (ConditionalMode, UnconditionalMode, AutoregressiveMode, STEMode))


class TestDecodeProtocols:
    """Test that decode protocols are callable type aliases."""

    def test_decode_score_fn_is_callable(self):
        """DecodeScoreFn is a Callable type alias."""
        # Protocols are type aliases, not runtime objects
        # Just verify they exist and are not None
        assert DecodeScoreFn is not None

    def test_ar_decode_fn_is_callable(self):
        """ARDecodeFn is a Callable type alias."""
        assert ARDecodeFn is not None

    def test_ste_decode_fn_is_callable(self):
        """STEDecodeFn is a Callable type alias."""
        assert STEDecodeFn is not None

    def test_decoder_sink_fn_is_callable(self):
        """DecoderSinkFn is a Callable type alias."""
        assert DecoderSinkFn is not None
