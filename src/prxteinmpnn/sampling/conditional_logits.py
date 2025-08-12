"""Sample the logits conditioned on sequence from ProteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import DecodingEnum, RunConditionalDecoderFn, make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.sampling.initialize import sampling_encode
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AtomMask,
  ChainIndex,
  EdgeFeatures,
  InputBias,
  Logits,
  ModelParameters,
  NodeFeatures,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

ConditionalLogitsFn = Callable[
  [
    PRNGKeyArray,
    StructureAtomicCoordinates,
    ProteinSequence,
    AtomMask,
    ResidueIndex,
    ChainIndex,
    InputBias | None,
    int,
    float,
  ],
  tuple[Logits, NodeFeatures, EdgeFeatures],
]


def make_conditional_logits_fn(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> ConditionalLogitsFn:
  """Perform one conditional pass on the model to get the logits for a given sequence.

  This function sets up the ProteinMPNN model, runs a single encoder pass to extract features,
  and then runs a single conditional decoder pass to compute the final logits for a given
  input sequence and structure.

  Args:
    model_parameters: A dictionary of the pre-trained ProteinMPNN model parameters.
    decoding_order_fn: A function that generates the decoding order.
    structure_coordinates: A JAX array of shape `(L, 3)` containing the atomic coordinates
      of the protein structure.
    sequence: A JAX array of shape `(L,)` representing the one-hot encoded protein sequence.
    mask: A JAX array of shape `(L,)` with a boolean mask for valid atoms.
    residue_index: A JAX array of shape `(L,)` with the residue indices.
    chain_index: A JAX array of shape `(L,)` with the chain indices.
    prng_key: JAX pseudo-random number generator key for feature augmentation.
    bias: An optional JAX array of shape `(L, 21)` to add a bias to the final logits. Defaults to
      None.
    k_neighbors: The number of neighbors to consider for each residue. Defaults to 48.
    augment_eps: The epsilon value for data augmentation. Defaults to 0.0.
    num_encoder_layers: The number of encoder layers to use. Defaults to 3.
    num_decoder_layers: The number of decoder layers to use. Defaults to 3.

  Returns:
    A JAX array of shape `(L, 21)` representing the conditional logits for the input sequence.

  Example:
    >>> import jax.random as jr
    >>> prng_key = jr.PRNGKey(0)
    >>> # Assume `model_parameters`, `decoding_order_fn`, `structure_coordinates`, `sequence`,
    >>> # `mask`, `residue_index`, and `chain_index` are already defined.
    >>> # ...
    >>> logits = get_conditional_logits(
    >>>   model_parameters=model_parameters,
    >>>   decoding_order_fn=decoding_order_fn,
    >>>   structure_coordinates=structure_coordinates,
    >>>   sequence=sequence,
    >>>   mask=mask,
    >>>   residue_index=residue_index,
    >>>   chain_index=chain_index,
    >>>   prng_key=prng_key
    >>> )
    >>> print(logits.shape)
    (100, 21)

  """
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CROSS,
    num_encoder_layers=num_encoder_layers,
  )

  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CONDITIONAL,
    decoding_enum=DecodingEnum.CONDITIONAL,
    num_decoder_layers=num_decoder_layers,
  )
  decoder = cast("RunConditionalDecoderFn", decoder)

  sample_model_pass = sampling_encode(
    encoder=encoder,
    decoding_order_fn=decoding_order_fn,
  )

  @partial(jax.jit, static_argnames=("k_neighbors", "augment_eps"))
  def condition_logits(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    sequence: ProteinSequence,
    mask: AtomMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    bias: InputBias | None = None,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
  ) -> tuple[Logits, NodeFeatures, EdgeFeatures]:
    if bias is None:
      bias = jnp.zeros((structure_coordinates.shape[0], 21), dtype=jnp.float32)

    (
      node_features,
      edge_features,
      neighbor_indices,
      _,
      autoregressive_mask,
      _,  # next_rng_key (not needed for this function)
    ) = sample_model_pass(
      prng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      k_neighbors,
      augment_eps,
    )

    one_hot_sequence = jax.nn.one_hot(sequence, 21, dtype=jnp.float32)

    decoded_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      one_hot_sequence,
    )

    logits = final_projection(model_parameters, decoded_node_features)

    return logits + bias, decoded_node_features, edge_features

  return condition_logits
