"""Composable decode axis-iteration pipeline (Sprint 6, app-side).

This package contains MPNN-specific decode mode classes and protocols.
The library-side surface (`tiling/` and `types/`) is domain-neutral and extractable.
"""

from __future__ import annotations

from aminx.inference.decode.factory import make_decode_fn
from aminx.inference.decode.mode import (
  AutoregressiveMode,
  ConditionalMode,
  DecodeMode,
  STEMode,
  UnconditionalMode,
)

__all__ = [
  "AutoregressiveMode",
  "ConditionalMode",
  "DecodeMode",
  "STEMode",
  "UnconditionalMode",
  "make_decode_fn",
]
