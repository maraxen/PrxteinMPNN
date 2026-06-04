"""Unified StageSet-driven decode driver.

This module consolidates the three inference kernels (score_conditional,
score_unconditional, sample_autoregressive) into a single unified driver
that dispatches based on stage_set topology at call time.

Topology inference:
  TOPOLOGY_AR                 — sample_step is not None (autoregressive sampling)
  TOPOLOGY_CONDITIONAL_SCORE  — decode_step is None or ConditionalDecodeStep (teacher-forced)
  TOPOLOGY_UNCONDITIONAL      — decode_step is UnconditionalDecodeStep (unconditional scoring)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from jaxtyping import PRNGKeyArray

if TYPE_CHECKING:
  from aminx.types.bundles import ConditioningBundle, WaveScheduleBundle
  from aminx.types.configs import InferenceConfig
  from aminx.types.protocols import ModelProtocol
  from aminx.types.stages import StageSet

from aminx.inference.sample_autoregressive import SampleResult
from aminx.types.arrays import Logits
from aminx.types.encodings import EncoderOutput
from aminx.types.stages import UnconditionalDecodeStep

# Topology constants (used at call time, not traced)
TOPOLOGY_AR = "ar"
TOPOLOGY_CONDITIONAL_SCORE = "conditional_score"
TOPOLOGY_UNCONDITIONAL = "unconditional"


def infer_topology(stage_set: StageSet) -> str:
  """Infer decode topology from StageSet slot occupancy.

  Examines stage_set fields to determine which decoding path to use:
  AR (sampling), unconditional (scoring without sequence), or conditional
  (teacher-forced scoring with sequence context).

  Parameters
  ----------
  stage_set : StageSet
      StageSet configuration with decode_step and sample_step.

  Returns
  -------
  str
      One of TOPOLOGY_AR, TOPOLOGY_UNCONDITIONAL, or TOPOLOGY_CONDITIONAL_SCORE.
      - TOPOLOGY_AR: sample_step is not None (autoregressive sampling)
      - TOPOLOGY_UNCONDITIONAL: decode_step is UnconditionalDecodeStep
      - TOPOLOGY_CONDITIONAL_SCORE: all else (conditional or fallback to model.decoder)

  References
  ----------
  .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
     sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
     https://doi.org/10.1126/science.add2187
  """
  if stage_set.sample_step is not None:
    return TOPOLOGY_AR
  if isinstance(stage_set.decode_step, UnconditionalDecodeStep):
    return TOPOLOGY_UNCONDITIONAL
  return TOPOLOGY_CONDITIONAL_SCORE


def decode(
  model: ModelProtocol,
  key: PRNGKeyArray,
  enc: EncoderOutput,
  cond: ConditioningBundle,
  wave: WaveScheduleBundle | None,
  config: InferenceConfig,
  stage_set: StageSet,
) -> Union[Logits, SampleResult]:
  """DEPRECATED: Unified decode driver.

  This function is deprecated as of Sprint 6. The three decode paths
  (_decode_conditional, _decode_unconditional, decode_ar) have been
  refactored into mode classes (ConditionalDecode, UnconditionalDecode,
  AutoregressiveDecode) that are resolved at InferencePlan construction time.

  Use InferencePlan.decode_fn or InferencePlan.decode() instead.

  This router function infer_topology() is preserved for backward compatibility
  with code that checks decode topology, but the actual decode dispatch is
  handled by the mode classes.

  Migration example:

    from aminx.host.plan import make_inference_plan

    # Old (deprecated):
    # result = driver.decode(model, key, enc, cond, wave, config, stage_set)

    # New (current):
    plan = make_inference_plan(model, spec)
    result = plan.decode(enc, bundle, key, config)
  """
  raise NotImplementedError(
    "driver.decode() is deprecated. Use InferencePlan.decode_fn() or "
    "InferencePlan.decode() after constructing a plan with make_inference_plan().",
  )


__all__ = [
  "TOPOLOGY_AR",
  "TOPOLOGY_CONDITIONAL_SCORE",
  "TOPOLOGY_UNCONDITIONAL",
  "infer_topology",
]
