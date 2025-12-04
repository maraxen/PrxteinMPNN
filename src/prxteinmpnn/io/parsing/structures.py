"""Structure data classes for parsing."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import numpy as np
  from biotite.structure import AtomArray, AtomArrayStack


@dataclasses.dataclass
class ProcessedStructure:
  """A structure that has been parsed and processed."""

  atom_array: AtomArray | AtomArrayStack
  r_indices: np.ndarray
  chain_ids: np.ndarray
  charges: np.ndarray | None = None
  radii: np.ndarray | None = None
  sigmas: np.ndarray | None = None
  epsilons: np.ndarray | None = None
