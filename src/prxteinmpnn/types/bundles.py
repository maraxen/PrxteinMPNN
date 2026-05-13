"""Canonical PyTree bundles for PrxteinMPNN inference.

These bundles form the strict boundary between host-side preparation and
accelerator-side JIT kernels. All Optional fields are resolved to concrete
zero-filled arrays by the host before entering JIT.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int


class GeometryBundle(eqx.Module):
    """Stacked backbone geometry. S=1 for single-state."""
    coords: Float[Array, "S L 4 3"]
    mask: Float[Array, "S L"]
    residue_index: Int[Array, "S L"]
    chain_index: Int[Array, "S L"]
    state_flat_rows: Int[Array, "S L"]  # maps stack → flat supersystem
    n_states: int = eqx.field(static=True)
    n_canonical: int = eqx.field(static=True)
    n_flat: int = eqx.field(static=True)


class ConditioningBundle(eqx.Module):
    """Sequence conditioning — fully resolved, no Optional."""
    fixed_mask: Float[Array, "L"]
    fixed_tokens: Int[Array, "L"]
    bias: Float[Array, "L V"]
    tie_group_map: Int[Array, "S L"]
    state_weights: Float[Array, "S"]
    sequence_oh: Float[Array, "L V"]  # zeros for unconditional/AR
    ar_mask: Float[Array, "S L L"]  # full 1s for purely conditional


class LigandBundle(eqx.Module):
    """Ligand context. All-zeros when no ligand."""
    y: Float[Array, "S L_lig A 3"]
    y_t: Int[Array, "S L_lig A"]
    y_m: Float[Array, "S L_lig A"]


class WaveScheduleBundle(eqx.Module):
    """Wave-parallel AR schedule."""
    group_ids: Int[Array, "W L"]
    group_positions: Int[Array, "W L G"]
    group_valid: Bool[Array, "W L"]
    position_valid: Bool[Array, "W L G"]


class InferenceBundle(eqx.Module):
    """Top-level JIT input."""
    geometry: GeometryBundle
    conditioning: ConditioningBundle
    ligand: LigandBundle
    wave: WaveScheduleBundle
