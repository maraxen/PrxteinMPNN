"""Public API for prxteinmpnn.model.

Public contract (everything in __all__):
- Model classes: PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN, Packer
- Encoder/decoder layers: Encoder, EncoderLayer, Decoder, DecoderLayer
- Feature extraction: ProteinFeatures
- Capability introspection: ModelCapabilities, PRXTEIN_MPNN_CAPABILITIES, PRXTEIN_LIGAND_MPNN_CAPABILITIES
- Multistate helpers: gather_flat_to_stack, scatter_stack_to_flat

Internal: files in model/_inference/ are implementation details.
Importing from prxteinmpnn.model._inference.* outside of model/ is unsupported.
"""

from __future__ import annotations

from .capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from .decoder import Decoder, DecoderLayer
from .diffusion_mpnn import DiffusionPrxteinMPNN
from .encoder import Encoder, EncoderLayer
from .features import ProteinFeatures
from .ligand_mpnn import PrxteinLigandMPNN
from .mpnn import PrxteinMPNN
from .multistate_stack import gather_flat_to_stack, scatter_stack_to_flat
from .packer import Packer

__all__ = [
  "PrxteinMPNN",
  "PrxteinLigandMPNN",
  "DiffusionPrxteinMPNN",
  "Packer",
  "Encoder",
  "EncoderLayer",
  "Decoder",
  "DecoderLayer",
  "ProteinFeatures",
  "ModelCapabilities",
  "PRXTEIN_MPNN_CAPABILITIES",
  "PRXTEIN_LIGAND_MPNN_CAPABILITIES",
  "gather_flat_to_stack",
  "scatter_stack_to_flat",
]
