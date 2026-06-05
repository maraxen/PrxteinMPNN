"""Potts model and related inference utilities.

Parallel model family for sequence design via Potts energy:
  - model: PottsModel(eqx.Module) with TRW inference
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
from aminx.potts.model import (
  POTTS_ALPHABET,
  POTTS_TO_MPNN_ALPHABET_MAP,
  PottsModel,
  PottsParams,
)

__all__ = [
  "CalibrationModule",
  "IdentityCalibration",
  "LearnedCalibration",
  "POTTS_ALPHABET",
  "POTTS_TO_MPNN_ALPHABET_MAP",
  "PottsModel",
  "PottsParams",
  "load_calibration",
]
