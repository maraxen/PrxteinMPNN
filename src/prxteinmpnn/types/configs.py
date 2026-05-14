"""Static configuration dataclasses for PrxteinMPNN inference."""

from __future__ import annotations

import dataclasses
from typing import Literal


import equinox as eqx
from typing import Literal


class InferenceConfig(eqx.Module):
    """Configuration for PrxteinMPNN inference."""
    mode: str = eqx.static_field(default="score_conditional") 
    temperature: float = 1.0
    logit_combine_strategy: int = eqx.static_field(default=0)  # 0=arithmetic, 1=geometric, 2=product
    use_rolling_state: bool = eqx.static_field(default=False)  # scan vs vmap over states
    inference: bool = eqx.static_field(default=True)
