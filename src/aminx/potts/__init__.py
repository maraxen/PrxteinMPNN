"""Potts model and related inference utilities.

Parallel model family for sequence design via Potts energy:
  - model: PottsModel(eqx.Module) with TRW inference
  - poe: PoeModel(eqx.Module) Product-of-Experts N-backbone ensemble
  - spec: PottsRunSpec frozen dataclass for run configuration
  - sampling: MCMC sampling utilities
  - calibration: Post-hoc marginal calibration
  - designer: Multi-chain sequence design protocol
"""

from aminx.potts._trw_spec import PottsTRWRunSpec
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
from aminx.potts.sampling import gibbs_sweep, log_energy, parallel_tempering
from aminx.potts.spec import PottsRunSpec

try:
  from aminx.potts.poe import PoeModel, PoeOutput, PoeParams
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
  "PoeOutput",
  "PoeParams",
  "PottsModel",
  "PottsParams",
  "PottsRunSpec",
  "PottsTRWRunSpec",
  "gibbs_sweep",
  "load_calibration",
  "log_energy",
  "parallel_tempering",
]
