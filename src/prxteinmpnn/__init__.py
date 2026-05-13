"""PrxteinMPNN: a functional interface for ProteinMPNN.

Importing this package does not configure the multiprocessing start method. Call
``configure_multiprocessing()`` once at process start in notebooks, standalone scripts,
or any entrypoint that uses ``multiprocessing`` worker pools (the campaign CLI already
does this); see ``prxteinmpnn.runtime``.
"""

from .host.specs import (
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)
from .host.runner import sample
from .host.scoring import score

from .runtime import configure_multiprocessing

__version__ = "0.1.0"
__author__ = "Marielle Russo"
__description__ = "PrxteinMPNN: A functional interface for ProteinMPNN"
__license__ = "MIT"
__url__ = "https://github.com/maraxen/prxteinmpnn"

# TODO(tech-debt): `.agents/TECHNICAL_DEBT.md` §13 — repository hygiene (stale docs, dead code, CI warning budget).

__all__ = [
  "JacobianSpecification",
  "RunSpecification",
  "SamplingSpecification",
  "ScoringSpecification",
  "configure_multiprocessing",
  "sample",
  "score",
]
