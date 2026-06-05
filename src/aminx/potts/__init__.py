"""Potts model and related inference utilities.

Parallel model family for sequence design via Potts energy:
  - model: Bare Potts energy function
  - sampling: MCMC sampling utilities
  - calibration: Post-hoc marginal calibration
  - designer: Multi-chain sequence design protocol
"""

from aminx.potts.calibration import (
    CalibrationModule,
    IdentityCalibration,
    LearnedCalibration,
    load_calibration,
)

__all__ = [
    "CalibrationModule",
    "IdentityCalibration",
    "LearnedCalibration",
    "load_calibration",
]
