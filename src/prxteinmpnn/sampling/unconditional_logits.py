"""Sample the logits conditioned on sequence from ProteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.sampling.initialize import sampling_encode
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  BackboneNoise,
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

if TYPE_CHECKING:
  from prxteinmpnn.model.decoding_signatures import RunDecoderFn

UnconditionalLogitsFn = Callable[
  [
    PRNGKeyArray,
    StructureAtomicCoordinates,
    ProteinSequence,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    InputBias | None,
    int,
    BackboneNoise | None,
  ],
  tuple[Logits, NodeFeatures, EdgeFeatures],
]


def make_unconditional_logits_fn(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn | None = None,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> UnconditionalLogitsFn:
  """Perform one conditional pass on the model to get the logits for a given sequence.

  This function sets up the ProteinMPNN model, runs a single encoder pass to extract features,
  and then runs a single conditional decoder pass to compute the final logits for a given
  input sequence and structure.

  Args:
    model_parameters: A dictionary of the pre-trained ProteinMPNN model parameters.
    decoding_order_fn: A function that generates the decoding order.
    k_neighbors: The number of neighbors to consider for each residue. Defaults to 48.
    backbone_noise: The epsilon value for data augmentation. Defaults to 0.0.
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
    attention_mask_type="cross",
    num_encoder_layers=num_encoder_layers,
  )

  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_type=None,
    decoding_approach="unconditional",
    num_decoder_layers=num_decoder_layers,
  )
  decoder = cast("RunDecoderFn", decoder)

  sample_model_pass = sampling_encode(
    encoder=encoder,
    decoding_order_fn=decoding_order_fn,
  )

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def unconditioned_logits(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    bias: InputBias | None = None,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple[Logits, NodeFeatures, EdgeFeatures]:
    if bias is None:
      bias = jnp.zeros((structure_coordinates.shape[0], 21), dtype=jnp.float32)

    (
      node_features,
      edge_features,
      _,
      _,
      _,
      _,  # next_rng_key (not needed for this function)
    ) = sample_model_pass(
      prng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      dihedrals=None,
      autoregressive_mask=None,
      k_neighbors=k_neighbors,
      backbone_noise=backbone_noise,
    )

    decoded_node_features = decoder(
      node_features,
      edge_features,
      mask,
    )

    logits = final_projection(model_parameters, decoded_node_features)

    return logits + bias, decoded_node_features, edge_features

  return unconditioned_logits
