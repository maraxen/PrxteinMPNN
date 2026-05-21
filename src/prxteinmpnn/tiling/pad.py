"""InferenceBundle padding routines.

Host-side logic to pad PyTree bundles up to fixed compile-time bucket sizes.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from prxteinmpnn.types.bundles import InferenceBundle


def pad_bundle(bundle: InferenceBundle, target_length: int) -> InferenceBundle:
    """Pad the bundle sequence length dimension (L) up to target_length.
    
    This function zeroes out the padded positions and sets masks to False.
    """
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand

    S, L = geo.mask.shape
    if target_length <= L:
        return bundle

    pad_amt = target_length - L

    # Pad Geometry
    coords = jnp.pad(geo.coords, ((0, 0), (0, pad_amt), (0, 0), (0, 0)))
    mask = jnp.pad(geo.mask, ((0, 0), (0, pad_amt)))
    residue_index = jnp.pad(geo.residue_index, ((0, 0), (0, pad_amt)))
    chain_index = jnp.pad(geo.chain_index, ((0, 0), (0, pad_amt)))
    state_flat_rows = jnp.pad(geo.state_flat_rows, ((0, 0), (0, pad_amt)), constant_values=-1)

    new_geo = eqx.tree_at(
        lambda g: (g.coords, g.mask, g.residue_index, g.chain_index, g.state_flat_rows),
        geo,
        (coords, mask, residue_index, chain_index, state_flat_rows),
    )

    # Pad Conditioning
    fixed_mask = jnp.pad(cond.fixed_mask, (0, pad_amt))
    fixed_tokens = jnp.pad(cond.fixed_tokens, (0, pad_amt))
    bias = jnp.pad(cond.bias, ((0, pad_amt), (0, 0)))
    tie_group_map = jnp.pad(cond.tie_group_map, ((0, 0), (0, pad_amt)))
    sequence_oh = jnp.pad(cond.sequence_oh, ((0, pad_amt), (0, 0)))

    # For AR mask, pad the 2D grid
    ar_mask = jnp.pad(cond.ar_mask, ((0, 0), (0, pad_amt), (0, pad_amt)))

    new_cond = eqx.tree_at(
        lambda c: (c.fixed_mask, c.fixed_tokens, c.bias, c.tie_group_map, c.sequence_oh, c.ar_mask),
        cond,
        (fixed_mask, fixed_tokens, bias, tie_group_map, sequence_oh, ar_mask),
    )

    # Ligand is sparse/scattered or padded similarly. Ligand length is independent of protein length L.
    # It has shape (S, L_lig, A, 3) etc.
    # Usually we don't pad ligand here unless target_length is meant for ligand too.
    # Assuming ligand has its own padding upstream.

    # Wave schedule
    # If wave schedule is used, wave length must also match target_length
    wave = bundle.wave
    W, L_wave = wave.group_ids.shape
    if L_wave < target_length:
        pad_wave = target_length - L_wave
        group_ids = jnp.pad(wave.group_ids, ((0, 0), (0, pad_wave)))
        group_positions = jnp.pad(wave.group_positions, ((0, 0), (0, pad_wave), (0, 0)))
        group_valid = jnp.pad(wave.group_valid, ((0, 0), (0, pad_wave)))
        position_valid = jnp.pad(wave.position_valid, ((0, 0), (0, pad_wave), (0, 0)))

        new_wave = eqx.tree_at(
            lambda w: (w.group_ids, w.group_positions, w.group_valid, w.position_valid),
            wave,
            (group_ids, group_positions, group_valid, position_valid),
        )
    else:
        new_wave = wave

    return InferenceBundle(
        geometry=new_geo,
        conditioning=new_cond,
        ligand=lig,
        wave=new_wave,
    )
