"""Aminx: a functional interface for ProteinMPNN.

Importing this package does not configure the multiprocessing start method. Call
``configure_multiprocessing()`` once at process start in notebooks, standalone scripts,
or any entrypoint that uses ``multiprocessing`` worker pools (the campaign CLI already
does this); see ``aminx.runtime``.

Submodules:
  - ``aminx.potts``: Potts model family with TRW inference and Gibbs sampling
    (see ``aminx.potts.__init__.py`` for PottsModel, PoeModel, etc.)
"""

from .run.specs import (
  JacobianSpecification,
  RunSpecification,
  SamplingSpecification,
  ScoringSpecification,
)
from .runtime import configure_multiprocessing
from .sampling import sample
from .scoring import score

__version__ = "0.1.0"
__author__ = "Marielle Russo"
__description__ = "Aminx: A functional interface for ProteinMPNN"
__license__ = "MIT"
__url__ = "https://github.com/maraxen/aminx"

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
