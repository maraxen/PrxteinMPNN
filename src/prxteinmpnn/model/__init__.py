"""Neural network architectures for PrxteinMPNN.

Public inference functions are organized along two conceptual axes:

    Operation × Strategy

    Operation: sample  — autoregressive sequence generation
               score   — log-probability evaluation of a given sequence

    Strategy:  exact   — full multistate attention over all conformational
                         states simultaneously (implemented via vmap over states)
               scan    — sequential per-position decode using lax.scan
                         (lower peak memory; single-state only)

    Modality:  implicit in the file name suffix:
               (no suffix) — protein-only model (PrxteinMPNN)
               _ligand     — ligand-aware model (PrxteinLigandMPNN)

File-to-concept mapping:
    model/ar_scan.py              sample × scan  × protein
    model/ar_exact.py             sample × exact × protein
    model/ar_exact_ligand.py      sample × exact × ligand
    model/score_exact_ligand.py   score  × exact × ligand

The verb ``vmap`` does not appear in any public name; it is an implementation
detail documented in the module-level docstrings of the individual files.
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
from .ligand_mpnn import PrxteinLigandMPNN
from .mpnn import PrxteinMPNN
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
