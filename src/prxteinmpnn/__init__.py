"""PrxteinMPNN: A functional interface for ProteinMPNN."""

import multiprocessing as mp

from .run import (
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
  sample,
  score,
)

mp.set_start_method("spawn", force=True)

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
  "sample",
  "score",
]
