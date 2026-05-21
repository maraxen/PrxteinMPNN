"""Static configuration dataclasses for PrxteinMPNN inference."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import equinox as eqx

if TYPE_CHECKING:
  from prxteinmpnn.types.protocols import DecodingOrderFn


class InferenceConfig(eqx.Module):
    """JIT-traced configuration for PrxteinMPNN inference."""
    mode: str = eqx.field(static=True, default="score_conditional")
    backbone_noise_mode: str = eqx.field(static=True, default="direct") # "direct" | "thermal"
    use_rolling_state: bool = eqx.field(static=True, default=False) # scan vs vmap over steps
    inference: bool = eqx.field(static=True, default=True)


@dataclasses.dataclass
class InferenceOptions:
    """Non-traced host-side options for inference dispatch."""
    decoding_order_fn: DecodingOrderFn | None = None
    batch_size: int = 1
    num_samples: int = 1
    iterations: int | None = None       # STE-specific
    learning_rate: float | None = None   # STE-specific
