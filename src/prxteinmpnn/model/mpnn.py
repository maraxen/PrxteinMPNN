"""Main ProteinMPNN model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.model.capabilities import (
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from prxteinmpnn.model.decoder import Decoder
from prxteinmpnn.model.encoder import (
  Encoder,
  PhysicsEncoder,
)
from prxteinmpnn.model.features import ProteinFeatures

from prxteinmpnn.types.bundles import InferenceBundle
from prxteinmpnn.types.configs import InferenceConfig

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import PRNGKeyArray

class PrxteinMPNN(eqx.Module):
  """The complete end-to-end ProteinMPNN model."""

  features: ProteinFeatures
  encoder: Encoder | PhysicsEncoder
  decoder: Decoder

  w_s_embed: eqx.nn.Embedding
  w_out: eqx.nn.Linear

  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)
  num_decoder_layers: int = eqx.field(static=True)
  capabilities: ModelCapabilities = eqx.field(static=True, default=PRXTEIN_MPNN_CAPABILITIES)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_encoder_layers: int,
    num_decoder_layers: int,
    k_neighbors: int,
    num_positional_embeddings: int = 16,
    physics_feature_dim: int | None = None,
    num_amino_acids: int = 21,
    vocab_size: int = 21,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.num_decoder_layers = num_decoder_layers

    keys = jax.random.split(key, 5)

    self.features = ProteinFeatures(
      node_features,
      edge_features,
      k_neighbors,
      num_positional_embeddings=num_positional_embeddings,
      key=keys[0],
    )
    self.encoder = (
      Encoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        dropout_rate=dropout_rate,
        key=keys[1],
      )
      if physics_feature_dim is None
      else PhysicsEncoder(
        node_features,
        edge_features,
        hidden_features,
        num_encoder_layers,
        dropout_rate,
        physics_feature_dim,
        key=keys[1],
      )
    )
    self.decoder = Decoder(
      node_features,
      edge_features,
      hidden_features,
      num_decoder_layers,
      dropout_rate=dropout_rate,
      key=keys[2],
    )
    self.w_s_embed = eqx.nn.Embedding(
      num_embeddings=vocab_size,
      embedding_size=node_features,
      key=keys[3],
    )
    self.w_out = eqx.nn.Linear(node_features, num_amino_acids, key=keys[4])

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Unified entry point: supports both raw features and InferenceBundle."""
    # 1. Normalize inputs
    bundle = kwargs.get("bundle")
    config = kwargs.get("config")
    key = kwargs.get("prng_key", kwargs.get("key"))
    
    coords = kwargs.get("coords", kwargs.get("structure_coordinates"))
    mask = kwargs.get("mask")
    residue_index = kwargs.get("residue_index")
    chain_index = kwargs.get("chain_index")
    decoding_approach = kwargs.get("decoding_approach")
    backbone_noise = kwargs.get("backbone_noise")

    # Handle positional args
    if args:
        first = args[0]
        if isinstance(first, InferenceBundle):
            bundle = first
            if len(args) > 1 and isinstance(args[1], InferenceConfig):
                config = args[1]
        elif (hasattr(first, "ndim") and first.ndim >= 2):
            # Legacy signature: (coords, mask, res, chain, approach, ...)
            coords = first
            if len(args) > 1: mask = args[1]
            if len(args) > 2: residue_index = args[2]
            if len(args) > 3: chain_index = args[3]
            if len(args) > 4: decoding_approach = args[4]
            # Handle additional legacy positional args (key, noise, mapping)
            if len(args) > 5 and key is None: key = args[5]
            if len(args) > 6 and backbone_noise is None: backbone_noise = args[6]
        else:
            # New protocol signature: (key, bundle, config) or (key, coords, ...)
            key = first
            if len(args) > 1:
                second = args[1]
                if isinstance(second, InferenceBundle):
                    bundle = second
                    if len(args) > 2 and isinstance(args[2], InferenceConfig):
                        config = args[2]
                else:
                    coords = second
                    if len(args) > 2: mask = args[2]
                    if len(args) > 3: residue_index = args[3]
                    if len(args) > 4: chain_index = args[4]

    # Defaults
    if key is None: key = jax.random.key(0)
    if backbone_noise is None: backbone_noise = jnp.array(0.0)

    # 2. Dispatch to Bundle path
    if bundle is not None:
        geo = bundle.geometry
        cond = bundle.conditioning
        lig = bundle.ligand
        
        def encode_one(c, m, ri, ci, y, yt, ym, sm):
            return self.features(key, c, m, ri, ci, backbone_noise, structure_mapping=sm)

        edge_f, edge_i, node_f, _ = jax.vmap(encode_one)(
            geo.coords, geo.mask, geo.residue_index, geo.chain_index,
            lig.y, lig.y_t, lig.y_m, geo.structure_mapping
        )
        
        node_f, edge_f = jax.vmap(self.encoder)(
            edge_f, edge_i, geo.mask, initial_node_features=node_f, key=jax.random.split(key, geo.n_states)
        )
        
        if config is not None:
            seq_oh = cond.sequence_oh
            decoded = jax.vmap(lambda n, e, i, m, a: 
                self.decoder.call_conditional(
                    n, e, i, m, a, seq_oh, self.w_s_embed.weight, 
                    key=key, inference=config.inference
                )
            )(node_f, edge_f, edge_i, geo.mask, cond.ar_mask)
            
            logits = jax.vmap(jax.vmap(self.w_out))(decoded)
            return logits 
            
        return node_f, edge_f, edge_i

    # 3. Step 3: Legacy/Training path
    if coords is None or mask is None or residue_index is None or chain_index is None:
        raise ValueError(f"Missing required inputs for legacy path: coords={coords is None}, mask={mask is None}, res={residue_index is None}, chain={chain_index is None}, args_len={len(args)}, kwargs_keys={list(kwargs.keys())}")
    
    if decoding_approach == "autoregressive":
        # Bridge legacy AR call to new kernel
        from prxteinmpnn.inference.sample_autoregressive import kernel as sample_ar
        from prxteinmpnn.inference._combine import ArithmeticMeanLogits
        from prxteinmpnn.types.bundles import (
            GeometryBundle, ConditioningBundle, 
            LigandBundle as InferenceLigandBundle, WaveScheduleBundle
        )
        from prxteinmpnn.types.stages import StageSet
        
        L = coords.shape[0]
        # Legacy coordinate inputs are (L, 4, 3). Modern bundles are (S, L, 4, 3).
        geo = GeometryBundle(
            coords=coords[None, ...], mask=mask[None, ...], 
            residue_index=residue_index[None, ...], chain_index=chain_index[None, ...],
            state_flat_rows=jnp.zeros((1, L), dtype=jnp.int32),
            n_states=1, n_canonical=L, n_flat=L,
            structure_mapping=kwargs.get("structure_mapping")[None, ...] if kwargs.get("structure_mapping") is not None else None
        )
        cond = ConditioningBundle(
            fixed_mask=kwargs.get("fixed_mask", jnp.zeros(L)),
            fixed_tokens=kwargs.get("fixed_tokens", jnp.zeros(L, dtype=jnp.int32)),
            bias=kwargs.get("bias", jnp.zeros((L, 21))),
            tie_group_map=kwargs.get("tie_group_map", jnp.arange(L)[None, :]),
            state_weights=jnp.array([1.0]),
            sequence_oh=kwargs.get("one_hot_sequence", jnp.zeros((L, 21))),
            ar_mask=kwargs.get("ar_mask", jnp.zeros((1, L, L)))
        )
        temp_bundle = InferenceBundle(
            geometry=geo, conditioning=cond, 
            ligand=InferenceLigandBundle(jnp.zeros((1, 0, 4, 3)), jnp.zeros((1, 0, 4), dtype=jnp.int32), jnp.zeros((1, 0, 4))),
            wave=WaveScheduleBundle.empty(L)
        )
        temp_config = InferenceConfig(mode="sample_autoregressive", inference=True)
        stage_set = StageSet(logit_transform=ArithmeticMeanLogits(jnp.array([1.0])))
        
        result_bundle = sample_ar(self, key, temp_bundle, temp_config, stage_set)
        
        # Final scoring for the sampled sequence to satisfy legacy API return expectations
        final_seq = result_bundle.conditioning.sequence_oh
        e_f, e_i, n_f, _ = self.features(
            key, coords, mask, residue_index, chain_index, backbone_noise,
            structure_mapping=kwargs.get("structure_mapping")
        )
        n_f, e_f = self.encoder(e_f, e_i, mask, initial_node_features=n_f, key=key)
        decoded = self.decoder.call_conditional(
            n_f, e_f, e_i, mask, result_bundle.conditioning.ar_mask[0], final_seq,
            self.w_s_embed.weight, key=key, inference=True
        )
        logits = jax.vmap(self.w_out)(decoded)
        return final_seq, logits

    # Standard encode/decode
    edge_features, edge_indices, node_features, _ = self.features(
      key, coords, mask, residue_index, chain_index, backbone_noise,
      structure_mapping=kwargs.get("structure_mapping")
    )
    node_features, edge_features = self.encoder(
        edge_features, edge_indices, mask, initial_node_features=node_features, key=key
    )
    
    if decoding_approach in ("conditional", "unconditional"):
        seq_oh = kwargs.get("one_hot_sequence")
        if seq_oh is None:
            seq_oh = jnp.zeros((node_features.shape[0], 21))
        
        ar_mask = kwargs.get("ar_mask")
        if ar_mask is None:
            L = node_features.shape[0]
            if decoding_approach == "conditional":
                # Default to causal mask for conditional scoring if none provided
                ar_mask = jnp.tril(jnp.ones((L, L)))
            else:
                ar_mask = jnp.zeros((L, L)) # Not used for unconditional but keeps shapes happy if passed
        
        if decoding_approach == "unconditional":
            decoded = self.decoder(
                node_features, edge_features, edge_indices, mask, key=key
            )
        else:
            decoded = self.decoder.call_conditional(
                node_features, edge_features, edge_indices, mask, ar_mask, seq_oh,
                self.w_s_embed.weight, key=key, inference=True
            )
        logits = jax.vmap(self.w_out)(decoded)
        return seq_oh, logits

    return node_features, edge_features, edge_indices
