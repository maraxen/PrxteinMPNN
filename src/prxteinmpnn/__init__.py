"""PrxteinMPNN: A functional interface for ProteinMPNN."""

import multiprocessing as mp

mp.set_start_method("spawn", force=True)
from .run import (
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
  categorical_jacobian,
  sample,
  score,
)

__version__ = "0.1.0"
__author__ = "Marielle Russo"
__description__ = "PrxteinMPNN: A functional interface for ProteinMPNN"
__license__ = "MIT"
__url__ = "https://github.com/maraxen/prxteinmpnn"
__all__ = [
  "JacobianSpecification",
  "RunSpecification",
  "SamplingSpecification",
  "ScoringSpecification",
  "categorical_jacobian",
  "sample",
  "score",
]
