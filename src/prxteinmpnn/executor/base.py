"""Base executor class for pipeline stage orchestration.

Executors coordinate stage dispatch and manage StageSet parameters.
This module provides the foundation; specific executors (Scorer, Sampler)
inherit from Executor and customize stage wiring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prxteinmpnn.pipeline_registry import StageSet

if TYPE_CHECKING:
  pass


class Executor:
  """Base executor for pipeline stage orchestration.

  Stores StageSet and coordinates stage dispatch across featurize, encode,
  decode, logit_transform, ar_logit_transform, and encoder_state_fn stages.
  """

  def __init__(self, stage_set: StageSet) -> None:
    """Initialize executor with stage set.

    Args:
      stage_set: StageSet containing UID strings for all pipeline stages.
    """
    self._stage_set = stage_set
