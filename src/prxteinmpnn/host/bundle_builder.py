"""Deprecated: use prxteinmpnn.inference.bundle_builder instead."""

import warnings

warnings.warn(
  "prxteinmpnn.host.bundle_builder is deprecated; use prxteinmpnn.inference.bundle_builder",
  DeprecationWarning,
  stacklevel=2,
)
from prxteinmpnn.inference.bundle_builder import *  # noqa: F403
