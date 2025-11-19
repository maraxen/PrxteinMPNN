"""Diffusion-aware ProteinMPNN subclass."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model.mpnn import PrxteinMPNN

if TYPE_CHECKING:
    from prxteinmpnn.utils.types import (
        AlphaCarbonMask,
        ChainIndex,
        Logits,
        OneHotProteinSequence,
        PRNGKeyArray,
        ResidueIndex,
        StructureAtomicCoordinates,
    )


class SinusoidalEmbedding(eqx.Module):
    """Sinusoidal positional embeddings for time."""
    
    embedding_dim: int
    max_period: float = 10000.0
    
    def __call__(self, timesteps: jax.Array, *, key: PRNGKeyArray | None = None) -> jax.Array:
        """Compute sinusoidal embeddings.
        
        Args:
            timesteps: [B] array of timesteps
            
        Returns:
            [B, embedding_dim] embeddings
        """
        half_dim = self.embedding_dim // 2
        freqs = jnp.exp(
            -jnp.log(self.max_period) * jnp.arange(0, half_dim, dtype=jnp.float32) / half_dim
        )
        if timesteps.ndim == 0:
            args = timesteps * freqs
            embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        else:
            args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
            embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        
        if self.embedding_dim % 2 == 1:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
            
        return embedding


class SwiGLU(eqx.Module):
    """SwiGLU activation layer."""
    
    w_gate: eqx.nn.Linear
    w_val: eqx.nn.Linear
    w_out: eqx.nn.Linear
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, key: PRNGKeyArray):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w_gate = eqx.nn.Linear(in_dim, hidden_dim, key=k1)
        self.w_val = eqx.nn.Linear(in_dim, hidden_dim, key=k2)
        self.w_out = eqx.nn.Linear(hidden_dim, out_dim, key=k3)
        
    def __call__(self, x: jax.Array, *, key: PRNGKeyArray | None = None) -> jax.Array:
        gate = jax.nn.silu(self.w_gate(x))
        val = self.w_val(x)
        return self.w_out(gate * val)


class DiffusionPrxteinMPNN(PrxteinMPNN):
    """ProteinMPNN extended for diffusion training."""
    
    w_t_embed: eqx.Module  # Sinusoidal + MLP
    
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        k_neighbors: int,
        physics_feature_dim: int | None = None,
        num_amino_acids: int = 21,
        vocab_size: int = 21,
        *,
        key: PRNGKeyArray,
    ) -> None:
        key, t_key = jax.random.split(key)
        
        super().__init__(
            node_features,
            edge_features,
            hidden_features,
            num_encoder_layers,
            num_decoder_layers,
            k_neighbors,
            physics_feature_dim,
            num_amino_acids,
            vocab_size,
            key=key,
        )
        
        # Timestep embedding: Sinusoidal -> SwiGLU MLP -> Node Features
        # We project to node_features_dim so we can add it to node features
        self.w_t_embed = eqx.nn.Sequential([
            SinusoidalEmbedding(node_features),
            SwiGLU(node_features, node_features * 4, node_features, key=t_key),
        ])

    def __call__(
        self,
        structure_coordinates: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_index: ResidueIndex,
        chain_index: ChainIndex,
        decoding_approach: Literal["unconditional", "conditional", "autoregressive", "diffusion"],
        *,
        prng_key: PRNGKeyArray | None = None,
        ar_mask: jax.Array | None = None,
        one_hot_sequence: OneHotProteinSequence | None = None,
        temperature: jax.Array | None = None,
        bias: Logits | None = None,
        backbone_noise: jax.Array | None = None,
        tie_group_map: jnp.ndarray | None = None,
        multi_state_strategy: Literal["mean", "min", "product", "max_min"] = "mean",
        multi_state_alpha: float = 0.5,
        structure_mapping: jnp.ndarray | None = None,
        initial_node_features: jnp.ndarray | None = None,
        # Diffusion specific args
        timestep: jax.Array | None = None, # [B]
        noisy_sequence: OneHotProteinSequence | None = None, # [B, N, 21]
        physics_features: jax.Array | None = None,
        physics_noise_scale: float | jax.Array = 0.0,
    ) -> tuple[OneHotProteinSequence, Logits]:
        
        if decoding_approach != "diffusion":
            return super().__call__(
                structure_coordinates,
                mask,
                residue_index,
                chain_index,
                decoding_approach, # type: ignore
                prng_key=prng_key,
                ar_mask=ar_mask,
                one_hot_sequence=one_hot_sequence,
                temperature=temperature,
                bias=bias,
                backbone_noise=backbone_noise,
                tie_group_map=tie_group_map,
                multi_state_strategy=multi_state_strategy,
                multi_state_alpha=multi_state_alpha,
                structure_mapping=structure_mapping,
                initial_node_features=initial_node_features,
                physics_features=physics_features,
            )
            
        # --- Diffusion Logic ---
        if prng_key is None:
            prng_key = jax.random.PRNGKey(0)
        prng_key, feat_key = jax.random.split(prng_key)

        if backbone_noise is None:
            backbone_noise = jnp.array(0.0, dtype=jnp.float32)
            
        # Apply noise to physics features if provided
        if physics_features is not None and physics_noise_scale > 0.0:
             # Use feat_key for noise generation
             phys_noise = jax.random.normal(feat_key, physics_features.shape)
             physics_features = physics_features + phys_noise * physics_noise_scale
            
        # 1. Extract Features & Encode (Standard MPNN)
        edge_features, neighbor_indices, node_features, _ = self.features(
            feat_key,
            structure_coordinates,
            mask,
            residue_index,
            chain_index,
            backbone_noise,
            structure_mapping=structure_mapping,
            initial_node_features=initial_node_features,
        )

        node_features, edge_features = self.encoder(
            edge_features,
            neighbor_indices,
            mask,
            node_features,
        )
        
        # 2. Inject Timestep Embedding
        if timestep is None:
             raise ValueError("timestep is required for diffusion mode")
             
        t_embed = self.w_t_embed(timestep) # [B, C]
        
        # Broadcast t_embed to [B, N, C] (batched) or [1, N, C] (unbatched/vmapped)
        if timestep.ndim == 1:
             t_embed = t_embed[:, None, :] # [B, 1, C]
        else:
             t_embed = t_embed[None, :] # [1, C]
             
        node_features = node_features + t_embed
        
        # 3. Decode (Conditional on Noisy Sequence)
        if noisy_sequence is None:
            raise ValueError("noisy_sequence is required for diffusion mode")
            
        # Use full visibility mask for parallel decoding if not provided
        if ar_mask is None:
            n = mask.shape[0]
            ar_mask = jnp.ones((n, n), dtype=jnp.int32)
            
        # Call internal conditional method bypassing super().__call__ dispatch
        
        return self._call_conditional(
            node_features,
            edge_features,
            neighbor_indices,
            mask,
            ar_mask,
            noisy_sequence,
            prng_key, # Unused
            jnp.array(1.0), # Temp unused
            jnp.zeros((mask.shape[0], 21)), # Bias unused
            None, # tie_group_map unused
            0, # strategy unused
            0.5, # alpha unused
            initial_node_features,
        )
