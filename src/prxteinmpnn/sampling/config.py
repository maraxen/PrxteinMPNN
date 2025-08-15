"""Configuration for sequence sampling in the PrxteinMPNN project."""

import enum

from flax.struct import dataclass

from prxteinmpnn.utils.types import Logits


class SamplingEnum(enum.Enum):
  """Enum for different sampling strategies."""

  TEMPERATURE = "temperature"
  STRAIGHT_THROUGH = "straight_through"


@dataclass(frozen=True)
class SamplingConfig:
  """Configuration for sequence sampling."""

  # Static parameters that control the computation graph
  sampling_strategy: SamplingEnum

  iterations: int = 1
  temperature: float = 1.0
  target_logits: Logits | None = None
  learning_rate: float = 0.1
