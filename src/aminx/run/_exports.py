"""Spec-driven exports for aminx.run.

This module re-exports `sample` and `score` from `host.runner` (spec-driven,
returns dict) as the primary public API via `aminx.run`.

The legacy tensor APIs (`aminx.sampling.sample`, `aminx.scoring.score.score`)
remain importable.
"""

from __future__ import annotations

# Import spec-driven functions from host.runner
from aminx.host.runner import sample, score

__all__ = ["sample", "score"]
