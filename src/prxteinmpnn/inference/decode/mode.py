"""DecodeMode sealed union (app-side, Task 4).

Four frozen dataclass variants representing decode strategies:
  - ConditionalMode: condition on structure + sequence
  - UnconditionalMode: no sequence conditioning
  - AutoregressiveMode: sequential decoding with wave iteration
  - STEMode: scoring-based sequence refinement (wraps an inner mode)

Per spec (Task 4.2), AutoregressiveMode has no W-axis fields; the wave-axis
Scan is a structural invariant on the AutoregressiveDecode class, not a knob.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalMode:
    """Decode mode: condition on structure + sequence."""
    pass


@dataclass(frozen=True)
class UnconditionalMode:
    """Decode mode: no sequence conditioning."""
    pass


@dataclass(frozen=True)
class AutoregressiveMode:
    """Decode mode: sequential decoding with wave iteration.

    Wave axis is always Scan internally — a structural invariant, not a knob.
    There is no W-axis BatchPlanner decision; the user does not pass a wave_iterator.

    Risk D-3 mitigation: no user-facing W-axis knob; unreachable by API design.
    """
    pass


@dataclass(frozen=True)
class STEMode:
    """Decode mode: scoring-based sequence refinement (STE).

    Wraps an inner decode mode (typically ConditionalMode) and applies
    iterative gradient-based refinement.

    Parameters
    ----------
    inner_mode : ConditionalMode, default=ConditionalMode()
        The inner decoding strategy used for gradient computation.
    iterations : int, default=100
        Number of STE refinement iterations.
    """
    inner_mode: ConditionalMode = ConditionalMode()
    iterations: int = 100


DecodeMode = ConditionalMode | UnconditionalMode | AutoregressiveMode | STEMode

__all__ = [
    "ConditionalMode",
    "UnconditionalMode",
    "AutoregressiveMode",
    "STEMode",
    "DecodeMode",
]
