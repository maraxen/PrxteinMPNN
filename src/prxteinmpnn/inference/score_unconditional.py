"""Unconditional scoring kernel using InferenceBundle."""

from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig
from prxteinmpnn.types.protocols import ModelProtocol
from prxteinmpnn.types.arrays import Logits
from prxteinmpnn.types.encodings import EncoderOutput



from prxteinmpnn.types.stages import StageSet

def kernel(
    model: ModelProtocol,
    prng_key: PRNGKeyArray,
    bundle: InferenceBundle,
    config: InferenceConfig,
    stage_set: StageSet,
) -> Logits:
    """Compute unconditional logits."""
    k_enc, k_dec = jax.random.split(prng_key)
    
    geo = bundle.geometry
    cond = bundle.conditioning
    lig = bundle.ligand

    S = geo.n_states
    
    # We vmap over the S dimension
    def encode_one(
        coords: jax.Array, 
        mask: jax.Array, 
        residue_index: jax.Array, 
        chain_index: jax.Array, 
        ligand_y: jax.Array, 
        ligand_y_t: jax.Array, 
        ligand_y_m: jax.Array,
        structure_mapping: jax.Array | None,
        backbone_noise: jax.Array
    ):
        return model(
            key=k_enc,
            coords=coords,
            mask=mask,
            residue_index=residue_index,
            chain_index=chain_index,
            y=ligand_y,
            y_t=ligand_y_t,
            y_m=ligand_y_m,
            structure_mapping=structure_mapping,
            backbone_noise=backbone_noise,
            inference=config.inference,
        )

    noise_stack = jnp.broadcast_to(geo.backbone_noise, (S,))

    # Encode
    if config.use_rolling_state:
        # scan over states
        def scan_body(carry: Any, per_state: tuple[jax.Array, ...]):
            coords, mask, residue_index, chain_index, ligand_y, ligand_y_t, ligand_y_m, sm, noise = per_state
            h_V, h_E, E_idx = encode_one(coords, mask, residue_index, chain_index, ligand_y, ligand_y_t, ligand_y_m, sm, noise)
            # Accumulate running stats if needed; for now, simple stack
            return carry, EncoderOutput(node_features=h_V, edge_features=h_E, neighbor_indices=E_idx)

        _, enc = jax.lax.scan(
            scan_body,
            None,
            (geo.coords, geo.mask, geo.residue_index, geo.chain_index, lig.y, lig.y_t, lig.y_m, geo.structure_mapping, noise_stack)
        )
    else:
        node_f, edge_f, nei_f = jax.vmap(encode_one, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0))(
            geo.coords, geo.mask, geo.residue_index, geo.chain_index,
            lig.y, lig.y_t, lig.y_m, geo.structure_mapping, noise_stack
        )
        enc = EncoderOutput(node_features=node_f, edge_features=edge_f, neighbor_indices=nei_f)

    # Decode unconditional
    def decode_one(nb: jax.Array, eb: jax.Array, nei: jax.Array, mk: jax.Array):
        # For unconditional, we don't pass sequence to the decoder
        return model.decoder(nb, eb, nei, mk, key=k_dec, inference=config.inference)

    decoded = jax.vmap(decode_one, in_axes=(0, 0, 0, 0))(enc.node_features, enc.edge_features, enc.neighbor_indices, geo.mask)
    logits_stack = jax.vmap(jax.vmap(model.w_out, in_axes=0), in_axes=0)(decoded)

    # Add bias
    logits_stack = logits_stack + cond.bias[None, ...]

    return stage_set.logit_transform(logits_stack)
