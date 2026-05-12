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
from prxteinmpnn.model.mpnn_autoregressive_scan import run_autoregressive_scan
from prxteinmpnn.model.mpnn_autoregressive_state_vmap_exact import (
  run_sample_autoregressive_state_vmap_exact,
)
from prxteinmpnn.model.mpnn_core import (
  autoregressive_decoding_context,
  edge_sequence_features_autoregressive,
)
from prxteinmpnn.model.multistate_stack import gather_flat_to_stack, scatter_stack_to_flat
from prxteinmpnn.payloads import MultistateStackPayload
from prxteinmpnn.registry import combine_strategy_to_index, multistate_mode_descriptor

if TYPE_CHECKING:
  from prxteinmpnn.model_inputs import ARLogitTransformFn, BackboneGeometry, LogitTransformFn
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

  def _call_unconditional(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    _ar_mask: AutoRegressiveMask,
    _one_hot_sequence: OneHotProteinSequence,
    _prng_key: PRNGKeyArray,
    _temperature: Float,
    _bias: Logits,
    _tie_group_map: TieGroupMap | None,
    _multi_state_strategy_idx: Int,
    _initial_node_features: NodeFeatures | None,
    _state_weights: jnp.ndarray | None,
    _state_mapping: jnp.ndarray | None,
    _fixed_mask: jnp.ndarray | None,
    _fixed_tokens: jnp.ndarray | None,
    _group_indices_table: jnp.ndarray | None,
    _group_valid_table: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the unconditional (scoring) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      _ar_mask: Unused, required for jax.lax.switch signature.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      _prng_key: Unused, required for jax.lax.switch signature.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      _tie_group_map: Unused, required for jax.lax.switch signature.
      _multi_state_strategy_idx: Unused, required for jax.lax.switch signature.
      _initial_node_features: Unused.
      _state_weights: Weights for each structural state.
      _state_mapping: Mapping of each residue to its state index.

    Returns:
      Tuple of (dummy sequence, logits).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> seq, logits = model._call_unconditional(edge_feats, neighbor_idx, mask)

    """
    decoded_node_features = self.decoder(
      node_features,
      edge_features,
      neighbor_indices,  # Pass neighbor indices for correct context
      mask,
      key=_prng_key,
    )

    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Multi-state logit combining for unconditional mode (consensus likelihood)
    if _tie_group_map is not None:
      logits = self._apply_multistate_to_all_logits(
        logits,
        _tie_group_map,
        _multi_state_strategy_idx,
        jnp.asarray(1.0, jnp.float32),
        _state_weights,
        _state_mapping,
      )

    # Return dummy sequence to match PyTree shape
    dummy_seq = jnp.zeros(
      (logits.shape[0], self.w_s_embed.num_embeddings),
      dtype=logits.dtype,
    )
    return dummy_seq, logits

  def _call_conditional(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    prng_key: PRNGKeyArray,
    _temperature: Float,
    _bias: Logits,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    _initial_node_features: NodeFeatures | None,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    _fixed_mask: jnp.ndarray | None,
    _fixed_tokens: jnp.ndarray | None,
    _group_indices_table: jnp.ndarray | None,
    _group_valid_table: jnp.ndarray | None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the conditional (scoring) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      ar_mask: Autoregressive mask for conditional decoding.
      one_hot_sequence: One-hot encoded protein sequence.
      prng_key: PRNG Key.
      _temperature: Unused, required for jax.lax.switch signature.
      _bias: Unused, required for jax.lax.switch signature.
      tie_group_map: Group mapping for multi-state scoring (consensus).
      multi_state_strategy_idx: Strategy index for combining logits.
      _initial_node_features: Unused.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      Tuple of (input sequence, logits).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> ar_mask = jnp.ones((10, 10))
      >>> seq = jax.nn.one_hot(jnp.arange(10), 21)
      >>> out_seq, logits = model._call_conditional(
      ...     edge_feats, neighbor_idx, mask, ar_mask, seq
      ... )

    """
    decoded_node_features = self.decoder.call_conditional(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      self.w_s_embed.weight,
      key=prng_key,
    )
    logits = jax.vmap(self.w_out)(decoded_node_features)

    # Multi-state logit combining for consensus likelihood
    if tie_group_map is not None:
      logits = self._apply_multistate_to_all_logits(
        logits,
        tie_group_map,
        multi_state_strategy_idx,
        jnp.asarray(1.0, jnp.float32),
        state_weights,
        state_mapping,
      )

    # Return input sequence to match PyTree shape
    return one_hot_sequence.astype(logits.dtype), logits

  def _call_autoregressive(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    _one_hot_sequence: OneHotProteinSequence,
    prng_key: PRNGKeyArray,
    temperature: Float,
    bias: Logits,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    _initial_node_features: NodeFeatures | None,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    fixed_mask: jnp.ndarray | None,
    fixed_tokens: jnp.ndarray | None,
    group_indices_table: jnp.ndarray | None,
    group_valid_table: jnp.ndarray | None,
    *,
    num_groups: int | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run the autoregressive (sampling) path.

    Args:
      node_features: Node features from encoding.
      edge_features: Edge features from encoding.
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask.
      ar_mask: Autoregressive mask for sampling.
      _one_hot_sequence: Unused, required for jax.lax.switch signature.
      prng_key: PRNG key for sampling.
      temperature: Temperature for Gumbel-max sampling.
      bias: Bias to add to logits before sampling (N, 21).
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group sample identical amino acids.
      multi_state_strategy_idx: Integer index for strategy
          (0=arithmetic_mean, 1=geometric_mean, 2=product).
      _initial_node_features: Unused.
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.
      group_indices_table: Pre-calculated table of group indices.
      group_valid_table: Pre-calculated boolean valid mask for group indices.

    Returns:
      Tuple of (sampled sequence, logits).

    Raises:
      None

    Warning:
      This path is NOT differentiable. Sequence sampling uses Gumbel-max noise +
      straight-through estimator (STE), creating a hard gradient barrier. Do not
      wrap calls with jax.grad / jax.value_and_grad / eqx.filter_grad — gradients
      will be zero or incorrect. For differentiable sequence optimization, use
      decoding_approach="conditional" (teacher-forced parallel decode).

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> ar_mask = jnp.ones((10, 10))
      >>> temp = jnp.array(1.0)
      >>> bias = jnp.zeros((10, 21))
      >>> seq, logits = model._call_autoregressive(
      ...     edge_feats, neighbor_idx, mask, ar_mask, None, key, temp, bias, None
      ... )

    """
    seq, logits = self._run_autoregressive_scan(
      prng_key,
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      temperature,
      bias,
      tie_group_map,
      multi_state_strategy_idx,
      state_weights,
      state_mapping,
      fixed_mask,
      fixed_tokens,
      group_indices_table,
      group_valid_table,
      num_groups=num_groups,
    )
    return seq, logits

  @staticmethod
  def _combine_logits_multistate(
    logits: Logits,
    group_mask: GroupMask,
    strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean",
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Combine logits across tied positions (:mod:`prxteinmpnn.model._shared`)."""
    return combine_logits_multistate(
      logits,
      group_mask,
      strategy,
      temperature,
      state_weights,
      state_mapping,
    )

  @staticmethod
  def _apply_multistate_to_all_logits(
    logits: Logits,
    tie_group_map: TieGroupMap,
    strategy_idx: Int,
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """Apply multi-state strategies in parallel (:mod:`prxteinmpnn.model._shared`)."""
    return apply_multistate_to_all_logits(
      logits,
      tie_group_map,
      strategy_idx,
      temperature,
      state_weights,
      state_mapping,
    )

  @staticmethod
  def _combine_logits_multistate_idx(
    logits: Logits,
    group_mask: GroupMask,
    strategy_idx: Int,
    temperature: float = 1.0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
  ) -> Logits:
    """JAX-traceable ``lax.switch`` combination (:mod:`prxteinmpnn.model._shared`)."""
    return combine_logits_multistate_idx(
      logits,
      group_mask,
      strategy_idx,
      temperature,
      state_weights,
      state_mapping,
    )

  def _process_group_positions(
    self,
    group_indices: jnp.ndarray,
    valid_mask: jnp.ndarray,
    all_layers_h: NodeFeatures,
    s_embed: NodeFeatures,
    encoder_context: NodeEdgeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_bw: LinkMask,
  ) -> tuple[NodeFeatures, Logits]:
    """Process only positions belonging to the current group through decoder.

    Args:
      group_indices: (S,) array of indices in the current tie group.
      valid_mask: (S,) boolean mask indicating valid indices.
      all_layers_h: Hidden states (num_layers+1, N, C).
      s_embed: Sequence embeddings (N, C).
      encoder_context: Precomputed encoder context (N, K, features).
      edge_features: Edge features (N, K, C).
      neighbor_indices: Neighbor indices (N, K).
      mask: Alpha carbon mask (N,).
      mask_bw: Backward mask (N, K).

    Returns:
      Tuple of (updated all_layers_h, computed logits (N, 21)).

    """
    num_residues = all_layers_h.shape[1]
    max_group_size = group_indices.shape[0]
    computed_logits = jnp.zeros((num_residues, 21), dtype=self.w_out.weight.dtype)

    def _decode_position(
      position_all_layers_h: jax.Array,
      idx: Int,
    ) -> tuple[jax.Array, jax.Array]:
      """Decode one position and return updated hidden state + logits."""
      encoder_context_pos = encoder_context[idx]
      neighbor_indices_pos = neighbor_indices[idx]
      mask_pos = mask[idx]
      mask_bw_pos = mask_bw[idx]

      edge_sequence_features = edge_sequence_features_autoregressive(
        s_embed,
        edge_features,
        neighbor_indices,
        idx,
      )

      for layer_idx, layer in enumerate(self.decoder.layers):
        h_in_pos = position_all_layers_h[layer_idx, idx]

        decoding_context = autoregressive_decoding_context(
          position_all_layers_h[layer_idx],
          edge_sequence_features,
          neighbor_indices_pos,
          encoder_context_pos,
          mask_bw_pos,
        )

        h_in_expanded = jnp.expand_dims(h_in_pos, axis=0)
        decoding_context_expanded = jnp.expand_dims(decoding_context, axis=0)

        h_out_pos = layer(
          h_in_expanded,
          decoding_context_expanded,
          mask=mask_pos,
          key=None,  # Dropout will use inference mode set on model
        )

        position_all_layers_h = position_all_layers_h.at[layer_idx + 1, idx].set(
          jnp.squeeze(h_out_pos),
        )

      final_h_pos = position_all_layers_h[-1, idx]
      logits_pos = self.w_out(final_h_pos)
      return position_all_layers_h, logits_pos

    def process_one_member(i: Int, state: tuple) -> tuple:
      """Process i-th member of the group if valid."""
      position_all_layers_h, position_logits = state
      idx = group_indices[i]
      is_valid = valid_mask[i]

      def _process(_: None) -> tuple:
        updated_h, logits_pos = _decode_position(position_all_layers_h, idx)
        return updated_h, position_logits.at[idx].set(logits_pos)

      return jax.lax.cond(
        is_valid,
        _process,
        lambda _: (position_all_layers_h, position_logits),
        operand=None,
      )

    return jax.lax.fori_loop(
      0,
      max_group_size,
      process_one_member,
      (all_layers_h, computed_logits),
    )

  def _run_autoregressive_scan(
    self,
    prng_key: PRNGKeyArray,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    autoregressive_mask: AutoRegressiveMask,
    temperature: Float,
    bias: Logits,
    tie_group_map: TieGroupMap | None = None,
    multi_state_strategy_idx: Int = 0,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    num_groups: int | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Run JAX scan loop for autoregressive sampling; see :mod:`prxteinmpnn.model.mpnn_autoregressive_scan`."""
    return run_autoregressive_scan(
      self,
      prng_key,
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      temperature,
      bias,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      fixed_mask=fixed_mask,
      fixed_tokens=fixed_tokens,
      group_indices_table=group_indices_table,
      group_valid_table=group_valid_table,
      num_groups=num_groups,
    )

  def __call__(
    self,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    decoding_approach: DecodingApproach,
    *,
    prng_key: PRNGKeyArray | None = None,
    ar_mask: AutoRegressiveMask | None = None,
    one_hot_sequence: OneHotProteinSequence | None = None,
    temperature: Float | None = None,
    bias: Logits | None = None,
    fixed_mask: jnp.ndarray | None = None,
    fixed_tokens: jnp.ndarray | None = None,
    backbone_noise: BackboneNoise | None = None,
    tie_group_map: jnp.ndarray | None = None,
    group_indices_table: jnp.ndarray | None = None,
    group_valid_table: jnp.ndarray | None = None,
    multi_state_strategy: Literal[
      "arithmetic_mean",
      "geometric_mean",
      "product",
    ] = "arithmetic_mean",
    multistate_mode: Literal["flat", "state_vmap", "state_vmap_exact"] = "flat",
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
    rbf_features: jnp.ndarray | None = None,
    neighbor_indices: jnp.ndarray | None = None,
    precomputed_node_features: jnp.ndarray | None = None,
    precomputed_edge_features: jnp.ndarray | None = None,
    precomputed_neighbor_indices: jnp.ndarray | None = None,
    membrane_per_residue_labels: jnp.ndarray | None = None,
    state_weights: jnp.ndarray | None = None,
    state_mapping: jnp.ndarray | None = None,
    num_groups: int | None = None,
    wave_group_ids: jnp.ndarray | None = None,  # noqa: ARG002
    wave_group_positions: jnp.ndarray | None = None,  # noqa: ARG002
    wave_group_valid: jnp.ndarray | None = None,  # noqa: ARG002
    wave_position_valid: jnp.ndarray | None = None,  # noqa: ARG002
    inference: bool = True,
    coords_stack: jnp.ndarray | None = None,
    mask_stack: jnp.ndarray | None = None,
    residue_index_stack: jnp.ndarray | None = None,
    chain_index_stack: jnp.ndarray | None = None,
    state_flat_rows: jnp.ndarray | None = None,
    n_flat: int | None = None,
    ar_mask_stack: jnp.ndarray | None = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Forward pass for the complete model.

    Dispatches to one of three modes:
    1. "unconditional": Scores all positions in parallel.
    2. "conditional": Scores a given sequence.
    3. "autoregressive": Samples a new sequence.

    When ``multistate_mode='state_vmap_exact'`` and stacked tensors are provided
    (``coords_stack``, ``mask_stack``, ``residue_index_stack``, ``chain_index_stack``,
    ``state_flat_rows``, ``n_flat``), unconditional and conditional scoring run
    :meth:`score_unconditional` / :meth:`score_conditional`.
    Autoregressive requests must use :meth:`sample_autoregressive_state_vmap_exact` or the sampler façade.

    Args:
      structure_coordinates: Raw atomic coordinates of protein structure.
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices for each position.
      chain_index: Chain indices for each position.
      decoding_approach: One of "unconditional", "conditional", or "autoregressive".
      prng_key: PRNG key for random operations (optional).
      ar_mask: Autoregressive mask for decoding order (optional).
      one_hot_sequence: One-hot encoded sequence for conditional mode (optional).
      temperature: Temperature for autoregressive sampling (optional).
      bias: Optional bias to add to logits before sampling (N, 21) (optional).
      backbone_noise: Noise level for backbone coordinates (optional).
      tie_group_map: Optional (N,) array mapping each position to a group ID.
          When provided, positions in the same group sample identical amino acids
          using logit combining. Only used in "autoregressive" mode (optional).
      multi_state_strategy: Strategy for combining logits across tied positions.
          Options: "arithmetic_mean" (default, log-sum-exp average),
          "geometric_mean" (geometric mean with temperature scaling),
          "product" (multiply probabilities).
          Only used in "autoregressive" mode with tied positions (optional).
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.
      initial_node_features: Optional (n_residues, feature_dim) physics features
                to use as initial node representations. If provided and the encoder
                is a PhysicsEncoder with use_initial_features=True, these will be
                used instead of zeros. Typically contains electrostatic features
                with shape (n_residues, 5).
      state_weights: Weights for each structural state.
      state_mapping: Mapping of each residue to its state index.

    Returns:
      A tuple of (OneHotProteinSequence, Logits).
        - For "unconditional", the sequence is a zero-tensor.
        - For "conditional", the sequence is the input sequence.
        - For "autoregressive", the sequence is the newly sampled one.

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> model = PrxteinMPNN(128, 128, 128, 3, 3, 30, key=key)
      >>> coords = jnp.ones((10, 4, 3))
      >>> mask = jnp.ones((10,))
      >>> residue_idx = jnp.arange(10)
      >>> chain_idx = jnp.zeros((10,), dtype=jnp.int32)
      >>> seq, logits = model(
      ...     coords, mask, residue_idx, chain_idx, "unconditional", prng_key=key
      ... )

    """
    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    ms_route = multistate_mode_descriptor(multistate_mode)

    if ms_route.uses_stacked_exact_model_call:
      if decoding_approach == "autoregressive":
        msg = (
          "PrxteinMPNN.__call__ does not run state_vmap_exact autoregressive decoding; "
          "use sample_autoregressive_state_vmap_exact or prxteinmpnn.sampling.sample.make_sample_sequences."
        )
        raise ValueError(msg)
      if membrane_per_residue_labels is not None:
        msg = (
          "membrane_per_residue_labels is not supported with multistate_mode='state_vmap_exact'."
        )
        raise ValueError(msg)
      if (
        precomputed_node_features is not None
        or precomputed_edge_features is not None
        or precomputed_neighbor_indices is not None
      ):
        msg = (
          "Encoder hoist kwargs (precomputed_node_features / edge / neighbor_indices) are not "
          "supported with multistate_mode='state_vmap_exact'; pass stacked geometry only."
        )
        raise ValueError(msg)
      need = {
        "coords_stack": coords_stack,
        "mask_stack": mask_stack,
        "residue_index_stack": residue_index_stack,
        "chain_index_stack": chain_index_stack,
        "state_flat_rows": state_flat_rows,
      }
      missing = [k for k, v in need.items() if v is None]
      if missing:
        msg = (
          "multistate_mode='state_vmap_exact' requires coords_stack, mask_stack, "
          "residue_index_stack, chain_index_stack, state_flat_rows, and n_flat; "
          f"missing: {', '.join(missing)}"
        )
        raise ValueError(msg)
      if n_flat is None:
        msg = "multistate_mode='state_vmap_exact' requires n_flat (flat logits length)."
        raise ValueError(msg)

      multi_state_strategy_idx_sv = jnp.array(
        combine_strategy_to_index(multi_state_strategy),
        dtype=jnp.int32,
      )

      if decoding_approach == "unconditional":
        logits_sv = self.score_unconditional(
          prng_key,
          coords_stack,
          mask_stack,
          residue_index_stack,
          chain_index_stack,
          state_flat_rows,
          n_flat,
          tie_group_map=tie_group_map,
          multi_state_strategy_idx=multi_state_strategy_idx_sv,
          state_weights=state_weights,
          state_mapping=state_mapping,
          inference=inference,
        )
        log_dtype = logits_sv.dtype
        zseq = jnp.zeros((n_flat, self.w_s_embed.num_embeddings), dtype=log_dtype)
        return zseq, logits_sv

      if decoding_approach == "conditional":
        if one_hot_sequence is None:
          msg = (
            "decoding_approach='conditional' with multistate_mode='state_vmap_exact' requires "
            "one_hot_sequence as (n_flat, num_embeddings) one-hot or (n_flat,) aa indices."
          )
          raise ValueError(msg)
        s_dim, p_dim = mask_stack.shape[0], mask_stack.shape[1]
        arm_sv = (
          jnp.zeros((s_dim, p_dim, p_dim), dtype=jnp.int32)
          if ar_mask_stack is None
          else ar_mask_stack
        )
        if one_hot_sequence.ndim == 1:
          oh_in = jax.nn.one_hot(
            one_hot_sequence.astype(jnp.int32),
            self.w_s_embed.num_embeddings,
          )
        else:
          oh_in = one_hot_sequence
        seq_stack_sv = gather_flat_to_stack(oh_in, state_flat_rows)
        logits_sv = self.score_conditional(
          prng_key,
          coords_stack,
          mask_stack,
          residue_index_stack,
          chain_index_stack,
          seq_stack_sv,
          arm_sv,
          state_flat_rows,
          n_flat,
          tie_group_map=tie_group_map,
          multi_state_strategy_idx=multi_state_strategy_idx_sv,
          state_weights=state_weights,
          state_mapping=state_mapping,
          bias_flat=bias,
          inference=inference,
        )
        return oh_in, logits_sv

      raise ValueError(
        f"Unsupported decoding_approach for PrxteinMPNN stacked path: {decoding_approach!r}",
      )

    # Automatically handle membrane labels if provided
    if membrane_per_residue_labels is not None:
      # Assume 3 classes for standard membrane MPNN (0: unknown/rest, 1: interface, 2: buried)
      initial_node_features = jax.nn.one_hot(membrane_per_residue_labels, 3)

    prng_key, feat_key = jax.random.split(prng_key)

    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    edge_features, new_neighbor_indices, node_features, _ = self.features(
      feat_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      structure_mapping=structure_mapping,
      initial_node_features=initial_node_features,
      rbf_features=rbf_features,
      neighbor_indices=neighbor_indices,
    )
    neighbor_indices = cast("jax.Array", new_neighbor_indices)

    node_features, edge_features = self.encoder(
      edge_features,
      neighbor_indices,
      mask,
      initial_node_features=node_features,
      inference=inference,
      key=prng_key,
    )

    branch_indices = {
      "unconditional": 0,
      "conditional": 1,
      "autoregressive": 2,
    }
    branch_index = branch_indices[decoding_approach]

    if ar_mask is None:
      ar_mask = jnp.zeros((mask.shape[0], mask.shape[0]), dtype=jnp.int32)

    if one_hot_sequence is None:
      one_hot_sequence = jnp.zeros(
        (mask.shape[0], self.w_s_embed.num_embeddings),
      )

    if temperature is None:
      temperature = jnp.array(1.0)

    if bias is None:
      bias = jnp.zeros((mask.shape[0], 21), dtype=jnp.float32)

    multi_state_strategy_idx = jnp.array(
      combine_strategy_to_index(multi_state_strategy),
      dtype=jnp.int32,
    )

    ar_fn = (
      partial(self._call_autoregressive, num_groups=num_groups)
      if num_groups is not None
      else self._call_autoregressive
    )
    branches = [
      self._call_unconditional,
      self._call_conditional,
      ar_fn,
    ]

    operands = (
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      prng_key,
      temperature,
      bias,
      tie_group_map,
      multi_state_strategy_idx,
      initial_node_features,
      state_weights,
      state_mapping,
      fixed_mask,
      fixed_tokens,
      group_indices_table,
      group_valid_table,
    )
    return jax.lax.switch(branch_index, branches, *operands)

  def sample_autoregressive_state_vmap_exact(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    autoregressive_mask_stack: jax.Array,
    tie_group_map_stack: jax.Array | None,
    bias_stack: jax.Array,
    temperature: Float | float,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    fixed_mask_stack: jax.Array,
    fixed_tokens_stack: jax.Array,
    wave_group_ids_local: jax.Array,
    wave_group_positions_local: jax.Array,
    wave_group_valid_local: jax.Array,
    wave_position_valid_local: jax.Array,
    ar_logit_transform_fn: "ARLogitTransformFn | None" = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Stacked-graph wave-parallel sampler for ProteinMPNN (``state_vmap_exact``).

    Outputs one-hot sequences and logits with shape ``(n_states, n_pad, ·)``.
    """
    return run_sample_autoregressive_state_vmap_exact(
      self,
      prng_key,
      coords_stack,
      mask_stack,
      residue_index_stack,
      chain_index_stack,
      autoregressive_mask_stack,
      tie_group_map_stack,
      bias_stack,
      temperature,
      multi_state_strategy_idx,
      state_weights,
      fixed_mask_stack,
      fixed_tokens_stack,
      wave_group_ids_local,
      wave_group_positions_local,
      wave_group_valid_local,
      wave_position_valid_local,
      ar_logit_transform_fn=ar_logit_transform_fn,
    )

  def sample_autoregressive_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    autoregressive_mask_stack: jax.Array,
    bias_stack: jax.Array,
    temperature: Float | float,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    wave_group_ids_local: jax.Array,
    wave_group_positions_local: jax.Array,
    wave_group_valid_local: jax.Array,
    wave_position_valid_local: jax.Array,
    ar_logit_transform_fn: "ARLogitTransformFn | None" = None,
  ) -> tuple[OneHotProteinSequence, Logits]:
    """Same as :meth:`sample_autoregressive_state_vmap_exact` with geometry read from ``stack``."""
    warnings.warn(
        "sample_autoregressive_state_vmap_exact_from_payload is deprecated; "
        "use sample_autoregressive_from_payload",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.sample_autoregressive_state_vmap_exact(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      autoregressive_mask_stack,
      stack.tie_group_map_stack,
      bias_stack,
      temperature,
      multi_state_strategy_idx,
      state_weights,
      stack.fixed_mask_stack,
      stack.fixed_tokens_stack,
      wave_group_ids_local,
      wave_group_positions_local,
      wave_group_valid_local,
      wave_position_valid_local,
      ar_logit_transform_fn=ar_logit_transform_fn,
    )

  def sample_autoregressive_from_payload(
    self,
    *args: Any,
    **kwargs: Any,
  ) -> Any:
    """Autoregressive sequence sampling from MultistateStackPayload.

    Clean alias for sample_autoregressive_state_vmap_exact_from_payload.
    """
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
      warnings.simplefilter("ignore", DeprecationWarning)
      return self.sample_autoregressive_state_vmap_exact_from_payload(*args, **kwargs)

  def score_unconditional(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
  ) -> Logits:
    """Compute unconditional logits per stacked state then scatter+fuse.

    The ``tie_group_map is None`` encode/decode stack matches the explicit ``jax.vmap`` reference
    in ``tests/sampling/spikes/test_state_vmap_exact_spike.py`` (roadmap §13.2 Phase 0a **GO**).
    When ``tie_group_map`` is set, logits are post-processed via
    :meth:`_apply_multistate_to_all_logits`.

    When ``logit_transform_fn`` is provided, it supersedes ``tie_group_map`` fusion —
    the hook receives raw per-state ``(S, L, V)`` logits and is responsible for
    combining them; ``apply_multistate_to_all_logits`` is not called.

    When ``encoder_state_fn`` is provided, the encoder runs via ``jax.lax.scan`` with
    carry accumulation instead of ``jax.vmap``.
    """
    k_enc, k_feat = jax.random.split(prng_key)

    def encode_one(coords, ma, ri, ci):
      ef, nei, nf, _ = self.features(
        k_feat,
        coords,
        ma,
        ri,
        ci,
        jnp.asarray(0.0, jnp.float32),
        structure_mapping=None,
        initial_node_features=None,
        rbf_features=None,
        neighbor_indices=None,
      )
      return encoder_forward_with_int_neighbors(
        self.encoder,
        ef,
        nei,
        ma,
        nf,
        inference=inference,
        key=k_enc,
      )

    if encoder_state_fn is not None:
      def scan_body(carry, per_state):
        coords_s, mask_s, ri_s, ci_s, idx_s = per_state
        from prxteinmpnn.model_inputs import BackboneGeometry  # noqa: PLC0415
        backbone_s = BackboneGeometry(
          coords=coords_s, mask=mask_s,
          residue_index=ri_s, chain_index=ci_s,
        )
        new_carry, enc_out = encoder_state_fn(carry, idx_s, backbone_s)
        return new_carry, enc_out

      _, enc_stacked = jax.lax.scan(
        scan_body,
        encoder_state_fn.init_carry(),
        (coords_stack, mask_stack, residue_index_stack, chain_index_stack,
         jnp.arange(coords_stack.shape[0], dtype=jnp.int32)),
      )
      node_b = enc_stacked.node_features
      edge_b = enc_stacked.edge_features
      nei_b = enc_stacked.neighbor_indices
    else:
      node_b, edge_b, nei_b = jax.vmap(encode_one)(
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
      )

    def decode_one(nb, eb, nei, mk):
      return self.decoder(nb, eb, nei, mk, key=k_enc)

    decoded = jax.vmap(decode_one)(node_b, edge_b, nei_b, mask_stack)
    logits_s = jax.vmap(jax.vmap(self.w_out))(decoded)

    if logit_transform_fn is not None:
      _sw = (
        jnp.ones(logits_s.shape[0], dtype=jnp.float32) / logits_s.shape[0]
        if state_weights is None
        else state_weights
      )
      _si = jnp.arange(logits_s.shape[0], dtype=jnp.int32)
      merged = logit_transform_fn(logits_s, _si, _sw)  # (L, V)
      return merged

    logits_flat = scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)

    if tie_group_map is not None:
      logits_flat = apply_multistate_to_all_logits(
        logits_flat,
        tie_group_map,
        multi_state_strategy_idx,
        jnp.asarray(1.0, jnp.float32),
        state_weights,
        state_mapping,
      )
    return logits_flat

  # Deprecated alias for backward compatibility
  score_unconditional_state_vmap_exact = score_unconditional

  def score_unconditional_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
  ) -> Logits:
    """Deprecated: same as score_unconditional_from_payload."""
    warnings.warn(
        "score_unconditional_state_vmap_exact_from_payload is deprecated; "
        "use score_unconditional_from_payload",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.score_unconditional(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      inference=inference,
      logit_transform_fn=logit_transform_fn,
      encoder_state_fn=encoder_state_fn,
    )

  def score_unconditional_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
  ) -> Logits:
    """Unconditional scoring from MultistateStackPayload."""
    return self.score_unconditional(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      inference=inference,
      logit_transform_fn=logit_transform_fn,
      encoder_state_fn=encoder_state_fn,
    )

  def score_conditional(
    self,
    prng_key: PRNGKeyArray,
    coords_stack: jax.Array,
    mask_stack: jax.Array,
    residue_index_stack: jax.Array,
    chain_index_stack: jax.Array,
    seq_oh_stack: jax.Array,
    ar_mask_stack: jax.Array,
    state_flat_rows: jax.Array,
    n_flat: int,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    bias_flat: jax.Array | None = None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
  ) -> Logits:
    """Teacher-forced conditional logits stacked then scatter+fused.

    When ``logit_transform_fn`` is provided, it supersedes ``tie_group_map`` fusion —
    the hook receives raw per-state ``(S, L, V)`` logits and is responsible for
    combining them; ``apply_multistate_to_all_logits`` is not called.

    When ``encoder_state_fn`` is provided, the encoder runs via ``jax.lax.scan`` with
    carry accumulation instead of ``jax.vmap``.
    """
    k_enc, k_feat = jax.random.split(prng_key)

    def encode_one(coords, ma, ri, ci):
      ef, nei, nf, _ = self.features(
        k_feat,
        coords,
        ma,
        ri,
        ci,
        jnp.asarray(0.0, jnp.float32),
        structure_mapping=None,
        initial_node_features=None,
        rbf_features=None,
        neighbor_indices=None,
      )
      return encoder_forward_with_int_neighbors(
        self.encoder,
        ef,
        nei,
        ma,
        nf,
        inference=inference,
        key=k_enc,
      )

    if encoder_state_fn is not None:
      def scan_body(carry, per_state):
        coords_s, mask_s, ri_s, ci_s, idx_s = per_state
        from prxteinmpnn.model_inputs import BackboneGeometry  # noqa: PLC0415
        backbone_s = BackboneGeometry(
          coords=coords_s, mask=mask_s,
          residue_index=ri_s, chain_index=ci_s,
        )
        new_carry, enc_out = encoder_state_fn(carry, idx_s, backbone_s)
        return new_carry, enc_out

      _, enc_stacked = jax.lax.scan(
        scan_body,
        encoder_state_fn.init_carry(),
        (coords_stack, mask_stack, residue_index_stack, chain_index_stack,
         jnp.arange(coords_stack.shape[0], dtype=jnp.int32)),
      )
      node_b = enc_stacked.node_features
      edge_b = enc_stacked.edge_features
      nei_b = enc_stacked.neighbor_indices
    else:
      node_b, edge_b, nei_b = jax.vmap(encode_one)(
        coords_stack,
        mask_stack,
        residue_index_stack,
        chain_index_stack,
      )

    def dec_one(nb, eb, nei, mk, arm, oh):
      return self.decoder.call_conditional(
        nb,
        eb,
        nei,
        mk,
        arm,
        oh,
        self.w_s_embed.weight,
        inference=inference,
        key=k_enc,
      )

    decoded = jax.vmap(dec_one)(
      node_b,
      edge_b,
      nei_b,
      mask_stack,
      ar_mask_stack,
      seq_oh_stack,
    )

    logits_s = jax.vmap(jax.vmap(self.w_out))(decoded)
    if bias_flat is not None:
      logits_s = logits_s + gather_flat_to_stack(bias_flat, state_flat_rows)

    if logit_transform_fn is not None:
      _sw = (
        jnp.ones(logits_s.shape[0], dtype=jnp.float32) / logits_s.shape[0]
        if state_weights is None
        else state_weights
      )
      _si = jnp.arange(logits_s.shape[0], dtype=jnp.int32)
      merged = logit_transform_fn(logits_s, _si, _sw)  # (L, V)
      return merged

    logits_flat = scatter_stack_to_flat(logits_s, state_flat_rows, n_flat)
    if tie_group_map is not None:
      logits_flat = apply_multistate_to_all_logits(
        logits_flat,
        tie_group_map,
        multi_state_strategy_idx,
        jnp.asarray(1.0, jnp.float32),
        state_weights,
        state_mapping,
      )
    return logits_flat

  # Deprecated alias for backward compatibility
  score_conditional_state_vmap_exact = score_conditional

  def score_conditional_state_vmap_exact_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    seq_oh_stack: jax.Array,
    ar_mask_stack: jax.Array,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    bias_flat: jax.Array | None = None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
  ) -> Logits:
    """Deprecated: same as score_conditional_from_payload."""
    warnings.warn(
        "score_conditional_state_vmap_exact_from_payload is deprecated; "
        "use score_conditional_from_payload",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.score_conditional(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      seq_oh_stack,
      ar_mask_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      bias_flat=bias_flat,
      inference=inference,
      logit_transform_fn=logit_transform_fn,
      encoder_state_fn=encoder_state_fn,
    )

  def score_conditional_from_payload(
    self,
    prng_key: PRNGKeyArray,
    stack: MultistateStackPayload,
    seq_oh_stack: jax.Array,
    ar_mask_stack: jax.Array,
    *,
    tie_group_map: TieGroupMap | None,
    multi_state_strategy_idx: Int,
    state_weights: jnp.ndarray | None,
    state_mapping: jnp.ndarray | None,
    bias_flat: jax.Array | None = None,
    inference: bool = True,
    logit_transform_fn: LogitTransformFn | None = None,
    encoder_state_fn: "EncoderStateFn | None" = None,
  ) -> Logits:
    """Conditional scoring from MultistateStackPayload."""
    return self.score_conditional(
      prng_key,
      stack.coords_stack,
      stack.mask_stack,
      stack.residue_index_stack,
      stack.chain_index_stack,
      seq_oh_stack,
      ar_mask_stack,
      stack.state_flat_rows,
      stack.n_flat,
      tie_group_map=tie_group_map,
      multi_state_strategy_idx=multi_state_strategy_idx,
      state_weights=state_weights,
      state_mapping=state_mapping,
      bias_flat=bias_flat,
      inference=inference,
      logit_transform_fn=logit_transform_fn,
      encoder_state_fn=encoder_state_fn,
    )
