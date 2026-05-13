"""Static capability flags for MPNN models."""

from __future__ import annotations

import equinox as eqx

class ModelCapabilities(eqx.Module):
  """Declared model surface capabilities."""
  is_ligand_model: bool = eqx.field(static=True)

PRXTEIN_MPNN_CAPABILITIES: ModelCapabilities = ModelCapabilities(
  is_ligand_model=False,
)

PRXTEIN_LIGAND_MPNN_CAPABILITIES: ModelCapabilities = ModelCapabilities(
  is_ligand_model=True,
)
