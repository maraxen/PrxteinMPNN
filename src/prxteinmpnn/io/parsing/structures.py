from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack


@dataclass
class ProcessedStructure:
  """Intermediate structure representation wrapping a Biotite AtomArray.

  Attributes:
    atom_array: The Biotite AtomArray containing all atoms (including hydrogens).
    r_indices: Residue indices (N,).
    chain_ids: Chain IDs (N,).
    charges: Optional array of atomic charges.
    radii: Optional array of atomic radii.
    epsilons: Optional array of atomic epsilons.
    sigmas: Optional array of atomic sigmas.

  """

  atom_array: AtomArray | AtomArrayStack
  r_indices: np.ndarray
  chain_ids: np.ndarray
  charges: np.ndarray | None = None
  radii: np.ndarray | None = None
  epsilons: np.ndarray | None = None
  sigmas: np.ndarray | None = None
