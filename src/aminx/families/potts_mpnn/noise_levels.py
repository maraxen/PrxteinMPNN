"""Every backbone noise level a spec asks for (spec 261011 N-D5)."""

from __future__ import annotations

from typing import Any


def potts_backbone_noise_levels(spec: Any) -> tuple[float, ...]:  # noqa: ANN401
  """All levels of the first enabled backbone noise bundle, or ``(0.0,)``.

  Levels share ONE noise draw scaled by the level (the level index is never
  folded into the key), so a single-level run is bit-identical to a run that
  only ever read ``noise_levels[0]``. The shape of the output depends only on
  ``len(levels)``: a noise axis exists iff there are two or more.
  """
  for bundle in getattr(spec, "noise", ()) or ():
    if bundle.feature_type == "backbone" and bundle.enabled and bundle.noise_levels:
      return tuple(float(level) for level in bundle.noise_levels)
  return (0.0,)
