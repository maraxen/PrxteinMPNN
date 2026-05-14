"""Canonical PyTree bundles for PrxteinMPNN inference.

These bundles form the strict boundary between host-side preparation and
accelerator-side JIT kernels. All Optional fields are resolved to concrete
zero-filled arrays by the host before entering JIT.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
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
    structure_mapping: Int[Array, "S L"] | None = None


class ConditioningBundle(eqx.Module):
    """Sequence conditioning — fully resolved, no Optional."""
    fixed_mask: Float[Array, "L"]
    fixed_tokens: Int[Array, "L"]
    bias: Float[Array, "L V"]
    tie_group_map: Int[Array, "S L"]
    state_weights: Float[Array, "S"]
    sequence_oh: Float[Array, "L V"]  # zeros for unconditional/AR
    ar_mask: Float[Array, "S L L"]  # full 1s for purely conditional
    temperature: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(1.0))


class LigandBundle(eqx.Module):
    """Ligand context. All-zeros when no ligand."""
    y: Float[Array, "S L_lig A 3"]
    y_t: Int[Array, "S L_lig A"]
    y_m: Float[Array, "S L_lig A"]


class WaveScheduleBundle(eqx.Module):
    group_ids: Int[Array, "W G"]
    group_positions: Int[Array, "W G P"]
    group_valid: Bool[Array, "W G"]
    position_valid: Bool[Array, "W G P"]

    @staticmethod
    def from_tie_groups(
        tie_group_map: Int[Array, "L"],
        decoding_order: Int[Array, "L"]
    ) -> WaveScheduleBundle:
        """Create a schedule where tied positions are in the same wave step."""
        L = tie_group_map.shape[0]
        # Map each position to its decoding step
        # (Assuming decoding_order respects ties: positions in same tie group
        # must appear consecutively or be handled as a block)
        # For now, let's group by tie_group_map.
        
        # Unique tie groups in order of first appearance in decoding_order
        present_groups = []
        seen_groups = set()
        for i in decoding_order.tolist():
            g = int(tie_group_map[i])
            if g not in seen_groups:
                present_groups.append(g)
                seen_groups.add(g)
        
        W = len(present_groups)
        # Maximum positions in a group
        counts = jnp.bincount(tie_group_map)
        P = int(jnp.max(counts))
        G = 1 # One tie-group per wave step for simplicity
        
        group_ids = jnp.array(present_groups)[:, None] # (W, 1)
        
        # group_positions: (W, 1, P)
        # This is tricky to do in JAX without loops if we want it general.
        # But since this is host-side factory, we can use loops.
        pos_list = []
        for g in present_groups:
            indices = jnp.where(tie_group_map == g)[0]
            # Pad to P
            padded = jnp.pad(indices, (0, P - len(indices)), constant_values=-1)
            pos_list.append(padded)
        
        group_positions = jnp.array(pos_list)[:, None, :]
        group_valid = jnp.ones((W, 1), dtype=jnp.bool_)
        position_valid = group_positions != -1
        
        # Replace -1 with 0 to avoid index errors (masked by position_valid)
        group_positions = jnp.where(position_valid, group_positions, 0)
        
        return WaveScheduleBundle(
            group_ids=group_ids,
            group_positions=group_positions,
            group_valid=group_valid,
            position_valid=position_valid
        )

    @staticmethod
    def empty(seq_len: int) -> WaveScheduleBundle:
        """Sequential single-position-at-a-time schedule."""
        # W = L, G = 1
        group_ids = jnp.arange(seq_len)[:, None]
        group_positions = jnp.arange(seq_len)[:, None, None]
        group_valid = jnp.ones((seq_len, 1), dtype=jnp.bool_)
        position_valid = jnp.ones((seq_len, 1, 1), dtype=jnp.bool_)
        return WaveScheduleBundle(
            group_ids=group_ids,
            group_positions=group_positions,
            group_valid=group_valid,
            position_valid=position_valid
        )


class InferenceBundle(eqx.Module):
    """Top-level JIT input."""
    geometry: GeometryBundle
    conditioning: ConditioningBundle
    ligand: LigandBundle
    wave: WaveScheduleBundle
    backbone_noise: Float[Array, ""] = eqx.field(default_factory=lambda: jnp.array(0.0))


class EncodedFeatures(eqx.Module):
    """Encoder outputs carried across encode / decode boundaries (single or batched)."""
    node_features: Float[Array, "L D"]
    edge_features: Float[Array, "L K D"]
    neighbor_indices: Int[Array, "L K"]


class EncoderOutput(eqx.Module):
    """Multi-state encoder output (S states)."""
    node_features: Float[Array, "S L D"]
    edge_features: Float[Array, "S L K D"]
    neighbor_indices: Int[Array, "S L K"]
    mask: Float[Array, "S L"]


class PackerResult(eqx.Module):
    """Mixture parameters for side-chain torsions."""
    mean: Float[Array, "L 4 3"]
    concentration: Float[Array, "L 4 3"]
    mix_logits: Float[Array, "L 4 3"]


class PackerBundle(eqx.Module):
    """Input features for side-chain packing."""
    s: Int[Array, "L"]
    x: Float[Array, "L 14 3"]
    x_m: Float[Array, "L 14"]
    y: Float[Array, "L M 3"]
    y_m: Float[Array, "L M"]
    y_t: Float[Array, "L M"]
    mask: Float[Array, "L"]
    residue_index: Int[Array, "L"]
    chain_labels: Int[Array, "L"]
    backbone_noise: Float[Array, ""] = 0.0
