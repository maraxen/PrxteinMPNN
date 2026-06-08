"""Deprecated: use aminx.inference.bundle_builder instead."""

import warnings

warnings.warn(
  "aminx.host.bundle_builder is deprecated; use aminx.inference.bundle_builder",
  DeprecationWarning,
  stacklevel=2,
)
from aminx.inference.bundle_builder import *  # noqa: F403
