"""DecodeMode sealed union (app-side, Task 4).

Four frozen dataclass variants representing decode strategies:
  - ConditionalMode: condition on structure + sequence
  - UnconditionalMode: no sequence conditioning
  - AutoregressiveMode: sequential decoding with wave iteration
  - STEMode: scoring-based sequence refinement (wraps an inner mode)

Per spec (Task 4.2), AutoregressiveMode has no W-axis fields; the wave-axis
iterator is a structural invariant on AutoregressiveDecode, not a user knob.
Set inference_only=True to request lax.while_loop for the wave axis —
this halves compilation pressure (WhileOp vs For/Scan in XLA) but makes
the path not reverse-mode differentiable. Always False for training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditionalMode:
  """Decode mode: condition on structure + sequence."""


@dataclass(frozen=True)
class UnconditionalMode:
  """Decode mode: no sequence conditioning."""


@dataclass(frozen=True)
class AutoregressiveMode:
  """Decode mode: sequential decoding with wave iteration.

  Parameters
  ----------
  inference_only : bool, default False
      When True, the wave axis uses ``jax.lax.while_loop`` instead of
      ``jax.lax.scan``.  ``while_loop`` lowers to a single XLA WhileOp and
      compiles significantly faster than a Scan op for large wave counts,
      but it is **not reverse-mode differentiable**.

      Set this to True for inference / benchmarking.  Leave False (default)
      for any path that requires gradients through the AR loop.

  Notes
  -----
  Risk D-3 mitigation: no user-facing W-axis iterator knob; the choice of
  scan vs. while_loop is the only user-visible W-axis decision, and it is
  gated behind an explicit ``inference_only`` flag rather than an iterator
  type argument.
  """

  inference_only: bool = False


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
  "AutoregressiveMode",
  "ConditionalMode",
  "DecodeMode",
  "STEMode",
  "UnconditionalMode",
]
