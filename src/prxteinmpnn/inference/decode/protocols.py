"""Decode protocols and type aliases (app-side, Task 4).

Three callable type aliases distinguish decode paths by return type:
  - DecodeScoreFn: conditional/unconditional decoding → Logits
  - ARDecodeFn: autoregressive decoding → SampleResult (or similar)
  - STEDecodeFn: scoring-based refinement → (sequence, logits_a, logits_b)
  - DecoderSinkFn: io_callback hook (mirrors EncoderSinkFn)

Naming note (oracle round 2): ScoreFn is already a top-level type alias
used by InferencePlan.score (Sprint 2 invariant, CLAUDE.md). The decode-specific
Conditional/Unconditional scoring protocol is therefore named DecodeScoreFn
to avoid shadowing.

Per Pattern 5 (skill caution): these are static-only type aliases, NOT @runtime_checkable.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = [
    "DecodeScoreFn",
    "ARDecodeFn",
    "STEDecodeFn",
    "DecoderSinkFn",
]

# Callable type aliases for decode protocols
# (Use Any for return types; specific imports verified at Task 4.3)

DecodeScoreFn = Callable[..., Any]
"""Conditional/unconditional decode protocol.

Signature: (key, enc, bundle, config, stage_set) -> Logits
"""

ARDecodeFn = Callable[..., Any]
"""Autoregressive decode protocol.

Signature: (key, enc, bundle, config, stage_set) -> SampleResult (or similar)
"""

STEDecodeFn = Callable[..., tuple[Any, ...]]
"""STE decode protocol.

Signature: (key, enc, bundle, config, stage_set) -> (sequence, logits_a, logits_b)
"""

DecoderSinkFn = Callable[..., None]
"""Decoder sink protocol (io_callback hook, mirrors EncoderSinkFn)."""
