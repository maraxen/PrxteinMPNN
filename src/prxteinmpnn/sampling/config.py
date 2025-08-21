"""Configuration for sequence sampling in the PrxteinMPNN project."""

from typing import Literal

from flax.struct import dataclass

from prxteinmpnn.utils.types import Logits

SamplingStrategy = Literal["temperature", "straight_through"]


@dataclass(frozen=True)
class SamplingConfig:
  """Configuration for sequence sampling."""

  sampling_strategy: SamplingStrategy
  iterations: int = 1
  temperature: float = 1.0
  target_logits: Logits | None = None
  learning_rate: float = 0.1
