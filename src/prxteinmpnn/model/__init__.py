"""Model module for PrxteinMPNN.

This module contains the core Equinox-based neural network components for ProteinMPNN.
"""

# TODO(tech-debt): `.agents/TECHNICAL_DEBT.md` §11 — StableHLO / WASM export constraints for published model entrypoints.

from __future__ import annotations

from .capabilities import (
  PRXTEIN_LIGAND_MPNN_CAPABILITIES,
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from .decoder import Decoder, DecoderLayer
from .encoder import Encoder, EncoderLayer
from .features import ProteinFeatures
from .mpnn import PrxteinLigandMPNN, PrxteinMPNN
from .multistate_stack import gather_flat_to_stack, scatter_stack_to_flat

__all__ = [
  "PRXTEIN_LIGAND_MPNN_CAPABILITIES",
  "PRXTEIN_MPNN_CAPABILITIES",
  "Decoder",
  "DecoderLayer",
  "Encoder",
  "EncoderLayer",
  "ModelCapabilities",
  "ProteinFeatures",
  "PrxteinLigandMPNN",
  "PrxteinMPNN",
  "gather_flat_to_stack",
  "scatter_stack_to_flat",
]
