"""STEPipeline: straight-through estimator differentiable sequence design."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
from jaxtyping import Array, Float, Int


class STEInputs(eqx.Module):
  """Inputs for STEPipeline (single-state STE optimization).

  coords: (L, 4, 3) backbone coordinates.
  mask: (L,) alpha-carbon mask.
  residue_index: (L,) residue indices.
  chain_index: (L,) chain indices.
  iterations: number of optimization steps.
  learning_rate: optimizer learning rate.
  temperature: STE softmax temperature.
  """

  coords: Float[Array, ...]
  mask: Float[Array, ...]
  residue_index: Int[Array, ...]
  chain_index: Int[Array, ...]
  iterations: int
  learning_rate: float
  temperature: float


@dataclasses.dataclass(frozen=True)
class STEPipeline:
  """Wraps make_optimize_sequence_fn with PipelineFns hooks.

  Inputs:  STEInputs (single-state backbone + STE hyperparams)
  Outputs: tuple[ProteinSequence, Logits, Logits] from optimize_sequence

  Note: LogitTransformFn is not currently threaded into the STE inner loop.
  Multistate STE wiring is a follow-up task.
  """

  def __call__(
    self,
    module: Any,
    key: Any,
    inputs: STEInputs,
    *,
    fns: Any,
  ) -> Any:
    """Run STE optimization and return (sequence, decoder_logits, final_logits)."""
    from prxteinmpnn.sampling.ste_optimize import make_optimize_sequence_fn  # noqa: PLC0415

    optimize_fn = make_optimize_sequence_fn(module)
    return optimize_fn(
      key,
      inputs.coords,
      inputs.mask,
      inputs.residue_index,
      inputs.chain_index,
      inputs.iterations,
      inputs.learning_rate,
      inputs.temperature,
    )
