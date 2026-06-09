"""Potts model and related inference utilities.

Parallel model family for sequence design via Potts energy:
  - model: PottsModel(eqx.Module) with TRW inference
  - poe: PoeModel(eqx.Module) Product-of-Experts N-backbone ensemble
  - spec: PottsRunSpec frozen dataclass for run configuration
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
from aminx.potts.spec import PottsRunSpec
from aminx.potts._trw_spec import PottsTRWRunSpec

try:
  from aminx.potts.poe import PoeModel, PoeParams
except ImportError:
  # poe may not be available if prxteinmpnn or other deps are not installed
  pass

__all__ = [
  "POTTS_ALPHABET",
  "POTTS_TO_MPNN_ALPHABET_MAP",
  "CalibrationModule",
  "IdentityCalibration",
  "LearnedCalibration",
  "PoeModel",
  "PoeParams",
  "PottsModel",
  "PottsParams",
  "PottsRunSpec",
  "PottsTRWRunSpec",
  "load_calibration",
]
