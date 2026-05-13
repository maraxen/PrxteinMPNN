# TODO: Explore internal state-batching (K, N, ...) instead of super-sequence concatenation to optimize attention complexity.
"""Main ProteinMPNN model implementation.

:class:`~prxteinmpnn.model.mpnn.PrxteinMPNN` lives here; ligand-conditioned
:class:`~prxteinmpnn.model.ligand_mpnn.PrxteinLigandMPNN` is defined in
:mod:`~prxteinmpnn.model.ligand_mpnn` and re-exported below for stable import paths.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.model._shared import (
  apply_multistate_to_all_logits,
  combine_logits_multistate,
  combine_logits_multistate_idx,
)
from prxteinmpnn.model.capabilities import (
  PRXTEIN_MPNN_CAPABILITIES,
  ModelCapabilities,
)
from prxteinmpnn.model.decoder import Decoder
from prxteinmpnn.model.encoder import (
  Encoder,
  PhysicsEncoder,
  encoder_forward_with_int_neighbors,
)
from prxteinmpnn.model.features import ProteinFeatures
from prxteinmpnn.model._inference.ar_scan import run_sample_ar_scan
from prxteinmpnn.model._inference.ar_exact import run_sample_ar_exact
from prxteinmpnn.model.mpnn_core import (
  autoregressive_decoding_context,
  edge_sequence_features_autoregressive,
)
from prxteinmpnn.model_inputs import (
  AutoregressiveInputs,
  ConditionalInputs,
  ModelInputs,
  UnconditionalInputs,
)
from prxteinmpnn.bundles import ProteinBundle
from prxteinmpnn.registry import combine_strategy_to_index, multistate_mode_descriptor

if TYPE_CHECKING:
  from prxteinmpnn.model_inputs import (
    ARLogitTransformFn,
    BackboneGeometry,
    ConditionalDecodeFn,
    FeaturizeFn,
    LogitTransformFn,
    ProteinEncodeFn,
    UnconditionalDecodeFn,
  )
  from prxteinmpnn.protocols import EncoderStateFn
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneNoise,
    ChainIndex,
    EdgeFeatures,
    Float,
    GroupMask,
    Int,
    LinkMask,
    Logits,
    NeighborIndices,
    NodeEdgeFeatures,
    NodeFeatures,
    OneHotProteinSequence,
    PRNGKeyArray,
    ResidueIndex,
    StructureAtomicCoordinates,
    TieGroupMap,
  )

DecodingApproach = Literal["unconditional", "conditional", "autoregressive"]


class PrxteinMPNN(eqx.Module):
  """The complete end-to-end ProteinMPNN model."""

  features: ProteinFeatures
  encoder: Encoder | PhysicsEncoder
  decoder: Decoder

  w_s_embed: eqx.nn.Embedding  # For sequence

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
    vocab_size: int = 21,  # for w_s
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the complete model.

    Args:
      node_features: Dimension of node features (e.g., 128).
      edge_features: Dimension of edge features (e.g., 128).
      hidden_features: Dimension of hidden layer in encoder/decoder.
      num_encoder_layers: Number of encoder layers.
      num_decoder_layers: Number of decoder layers.
      k_neighbors: Number of nearest neighbors for graph construction.
      physics_feature_dim: Dimension of physical features (if any).
      num_amino_acids: Number of amino acid types (default: 21).
      vocab_size: Size of sequence vocabulary (default: 21).
      dropout_rate: Dropout rate (default: 0.1).
      key: PRNG key for initialization.

    Returns:
      None

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)

    """
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features
    self.num_decoder_layers = num_decoder_layers

    keys = jax.random.split(key, 5)  # 1 for features, 4 for main model

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

  @classmethod
  def stage_schema(cls) -> dict[str, type | None]:
    """Returns {stage_name: type_alias} for this MPNN model variant."""
    from prxteinmpnn.model_inputs import (
      ARLogitTransformFn,
      ConditionalDecodeFn,
      FeaturizeFn,
      LogitTransformFn,
      ProteinEncodeFn,
      UnconditionalDecodeFn,
    )

    return {
        "featurize": FeaturizeFn,
        "encode": ProteinEncodeFn,
        "decode": ConditionalDecodeFn | UnconditionalDecodeFn,
        "logit_transform": LogitTransformFn,
        "ar_logit_transform": ARLogitTransformFn,
        "encoder_state_fn": None,
    }

