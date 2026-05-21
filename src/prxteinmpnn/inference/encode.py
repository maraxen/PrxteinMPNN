"""Single encode site: make_encode_fn returns an EncodeFn over InferenceBundle.

This module extracts the shared encode-once logic used across kernels:
- sample_autoregressive: vmap over S states
- score_conditional/unconditional: scan or vmap over S states (controlled by use_rolling_state)
- conditional_logits: stateless encode (single state)

The make_encode_fn factory controls the strategy (scan vs vmap over S).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from prxteinmpnn.types.encodings import EncoderOutput

if TYPE_CHECKING:
    from prxteinmpnn.types.bundles import InferenceBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.protocols import ModelProtocol


# Type alias for the encode function returned by make_encode_fn
# Signature: (bundle, prng_key, config) -> EncoderOutput
if TYPE_CHECKING:
    EncodeFn = Callable[["InferenceBundle", Any, "InferenceConfig"], EncoderOutput]
else:
    EncodeFn = Callable


def make_encode_fn(model: ModelProtocol, *, use_rolling_state: bool = False) -> EncodeFn:
    """Factory: returns an encode function using the specified strategy.

    The returned function encapsulates the choice between scan (sequential) and vmap
    (parallel) over conformational states S. Both strategies consume the same inputs
    and produce the same EncoderOutput.

    Args:
        model: PrxteinMPNN model with .features() and .encoder() methods.
        use_rolling_state: If True, use jax.lax.scan over S states (sequential,
            lower memory, allows state accumulation).
            If False, use jax.vmap over S states (parallel, higher throughput).
            Default: False (parallel vmap).

    Returns:
        EncodeFn: callable(bundle, prng_key, config) -> EncoderOutput
            Encodes structure and features, returning node_features, edge_features,
            and neighbor_indices stacked over S conformational states.

    Example:
        >>> encode_fn = make_encode_fn(model, use_rolling_state=False)
        >>> enc = encode_fn(bundle, key, config)
        >>> print(enc.node_features.shape)  # (S, L, H)
    """

    def encode_fn(
        bundle: InferenceBundle,
        prng_key: Any,
        config: InferenceConfig,
    ) -> EncoderOutput:
        """Encode structure and features.

        Args:
            bundle: InferenceBundle with geometry, ligand, and structure info.
            prng_key: JAX random key for stochastic operations.
            config: InferenceConfig controlling backbone noise mode.

        Returns:
            EncoderOutput with node_features, edge_features, neighbor_indices.
        """
        k_enc = prng_key
        geo = bundle.geometry
        lig = bundle.ligand
        S = geo.n_states

        # Broadcast backbone noise across S states
        noise_stack = jnp.broadcast_to(bundle.backbone_noise, (S,))

        phys = geo.physics_features  # None for soluble/ligand, (S, L, P) for membrane

        def encode_one(
            coords: jax.Array,
            mask: jax.Array,
            residue_index: jax.Array,
            chain_index: jax.Array,
            ligand_coords: jax.Array,
            ligand_atom_types: jax.Array,
            ligand_mask: jax.Array,
            structure_mapping: jax.Array | None,
            noise: jax.Array,
            pf: jax.Array | None,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            """Encode a single state: features + encoder."""
            kwargs: dict[str, Any] = dict(
                prng_key=jax.random.fold_in(k_enc, 0),
                backbone_noise=noise,
                backbone_noise_mode=config.backbone_noise_mode,
                structure_mapping=structure_mapping,
                y=ligand_coords,
                y_t=ligand_atom_types,
                y_m=ligand_mask,
                inference=config.inference,
            )
            if pf is not None:
                kwargs["initial_node_features"] = pf
            node_f, edge_f, edge_i = model(coords, mask, residue_index, chain_index, **kwargs)
            return node_f, edge_f, edge_i

        if use_rolling_state:
            if phys is not None:
                def scan_body(carry: Any, per_state: Any) -> tuple[Any, EncoderOutput]:
                    c, m, ri, ci, l_coords, l_types, l_mask, sm, n, pf = per_state
                    node_f, edge_f, edge_i = encode_one(c, m, ri, ci, l_coords, l_types, l_mask, sm, n, pf)
                    return carry, EncoderOutput(node_features=node_f, edge_features=edge_f, neighbor_indices=edge_i)
                scan_xs: Any = (
                    geo.coords, geo.mask, geo.residue_index, geo.chain_index,
                    lig.ligand_coords, lig.ligand_atom_types, lig.ligand_mask, geo.structure_mapping, noise_stack, phys,
                )
            else:
                def scan_body(carry: Any, per_state: Any) -> tuple[Any, EncoderOutput]:  # type: ignore[misc]
                    c, m, ri, ci, l_coords, l_types, l_mask, sm, n = per_state
                    node_f, edge_f, edge_i = encode_one(c, m, ri, ci, l_coords, l_types, l_mask, sm, n, None)
                    return carry, EncoderOutput(node_features=node_f, edge_features=edge_f, neighbor_indices=edge_i)
                scan_xs = (
                    geo.coords, geo.mask, geo.residue_index, geo.chain_index,
                    lig.ligand_coords, lig.ligand_atom_types, lig.ligand_mask, geo.structure_mapping, noise_stack,
                )

            _, enc = jax.lax.scan(scan_body, None, scan_xs)
            enc = EncoderOutput(
                node_features=enc.node_features,
                edge_features=enc.edge_features,
                neighbor_indices=enc.neighbor_indices,
                mask=geo.mask,
            )
        else:
            # Parallel vmap over S states; physics axis is 0 when present, None (broadcast) when absent.
            in_axes = (0, 0, 0, 0, 0, 0, 0, 0, 0, None if phys is None else 0)
            node_f, edge_f, nei_f = jax.vmap(encode_one, in_axes=in_axes)(
                geo.coords,
                geo.mask,
                geo.residue_index,
                geo.chain_index,
                lig.ligand_coords,
                lig.ligand_atom_types,
                lig.ligand_mask,
                geo.structure_mapping,
                noise_stack,
                phys,
            )
            enc = EncoderOutput(
                node_features=node_f,
                edge_features=edge_f,
                neighbor_indices=nei_f,
                mask=geo.mask,
            )

        return enc

    return encode_fn


__all__ = ["EncodeFn", "make_encode_fn"]
