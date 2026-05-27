"""Composable decode axis-iteration pipeline (Sprint 6, app-side).

This package contains MPNN-specific decode mode classes and protocols.
The library-side surface (`tiling/` and `types/`) is domain-neutral and extractable.
"""

from __future__ import annotations

from prxteinmpnn.inference.decode.factory import make_decode_fn
from prxteinmpnn.inference.decode.mode import (
    AutoregressiveMode,
    ConditionalMode,
    DecodeMode,
    STEMode,
    UnconditionalMode,
)

__all__ = [
    "make_decode_fn",
    "ConditionalMode",
    "UnconditionalMode",
    "AutoregressiveMode",
    "STEMode",
    "DecodeMode",
]
