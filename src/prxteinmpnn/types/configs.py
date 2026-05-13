"""Static configuration dataclasses for PrxteinMPNN inference."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class InferenceConfig:
    """Compile-time constants — static_argnames to JIT."""
    mode: Literal["score_unconditional", "score_conditional", "sample_ar"]
    temperature: float = 1.0
    logit_combine_strategy: int = 0  # 0=arithmetic, 1=geometric, 2=product
    use_rolling_state: bool = False  # scan vs vmap over states
    inference: bool = True
