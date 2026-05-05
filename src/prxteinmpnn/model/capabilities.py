"""Static capability flags for MPNN models (replaces ``inspect.signature`` branching)."""

from __future__ import annotations

import equinox as eqx


class ModelCapabilities(eqx.Module):
  """Declared model surface for sampling, scoring, and encoding helpers."""

  #: ``True`` when the model is ligand-aware (ligand geometry / ``Y`` path).
  is_ligand_model: bool = eqx.field(static=True)
  #: ``model.__call__`` accepts ``multi_state_temperature`` (flat conditional / autoreg paths).
  accepts_multi_state_temperature: bool = eqx.field(static=True)
  #: ``model.__call__`` accepts ``state_weights`` / ``state_mapping`` style weighting kwargs.
  accepts_state_weights: bool = eqx.field(static=True)
  #: ``model.__call__`` accepts ``fixed_mask`` and ``fixed_tokens``.
  accepts_fixed_mask_and_tokens: bool = eqx.field(static=True)
  #: Encoder split used by averaging / Jacobian accepts ``structure_mapping`` (e.g. ``make_encoding_sampling_split_fn``).
  encode_fn_supports_structure_mapping: bool = eqx.field(static=True)


PRXTEIN_MPNN_CAPABILITIES: ModelCapabilities = ModelCapabilities(
  is_ligand_model=False,
  accepts_multi_state_temperature=True,
  accepts_state_weights=True,
  accepts_fixed_mask_and_tokens=True,
  encode_fn_supports_structure_mapping=True,
)

PRXTEIN_LIGAND_MPNN_CAPABILITIES: ModelCapabilities = ModelCapabilities(
  is_ligand_model=True,
  accepts_multi_state_temperature=True,
  accepts_state_weights=True,
  accepts_fixed_mask_and_tokens=True,
  encode_fn_supports_structure_mapping=True,
)
