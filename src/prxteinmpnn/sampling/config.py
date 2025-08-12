"""Configuration for sequence sampling in the PrxteinMPNN project."""

import enum

import optax
from flax.struct import dataclass, field

from prxteinmpnn.utils.types import Logits


class SamplingEnum(enum.Enum):
  """Enum for different sampling strategies."""

  GREEDY = "greedy"
  TOP_K = "top_k"
  TOP_P = "top_p"
  TEMPERATURE = "temperature"
  BEAM_SEARCH = "beam_search"
  STRAIGHT_THROUGH = "straight_through"


@dataclass(frozen=True)
class SamplingConfig:
  """Configuration for sequence sampling."""

  # Static parameters that control the computation graph
  iterations: int
  sampling_strategy: SamplingEnum
  optimizer: optax.GradientTransformation | None = field(pytree_node=False, default=None)

  # Dynamic parameters (data) used within the computation
  temperature: float = 1.0
  target_logits: Logits | None = None
