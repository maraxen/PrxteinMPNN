"""Decoder function signatures."""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import PRNGKeyArray

from prxteinmpnn.utils.types import (
  AtomMask,
  AttentionMask,
  AutoRegressiveMask,
  EdgeFeatures,
  Logits,
  Message,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  OneHotProteinSequence,
  ProteinSequence,
)

DecodeMessageInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  ModelParameters,
]
DecodeMessageFn = Callable[[*DecodeMessageInputs], Message]


DecoderNormalizeInputs = tuple[
  Message,
  NodeFeatures,
  AtomMask,
  ModelParameters,
  float,
]
DecoderNormalizeFn = Callable[[*DecoderNormalizeInputs], NodeFeatures]


MaskedAttentionDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AtomMask,
  AttentionMask,
  ModelParameters,
  float,
]
MaskedAttentionDecoderFn = Callable[[*MaskedAttentionDecoderInputs], NodeFeatures]
DecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AtomMask,
  ModelParameters,
  float,
]
DecoderFn = Callable[[*DecoderInputs], NodeFeatures]


RunDecoderInputs = tuple[NodeFeatures, EdgeFeatures, AtomMask]
RunDecoderFn = Callable[[*RunDecoderInputs], NodeFeatures]
RunMaskedAttentionDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AtomMask,
  AttentionMask,
]
RunMaskedAttentionDecoderFn = Callable[[*RunMaskedAttentionDecoderInputs], NodeFeatures]
RunConditionalDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AtomMask,
  AutoRegressiveMask,
  ProteinSequence,
]
RunConditionalDecoderFn = Callable[[*RunConditionalDecoderInputs], NodeFeatures]
RunAutoregressiveDecoderInputs = tuple[
  PRNGKeyArray,
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AtomMask,
  AutoRegressiveMask,
  float,
]
RunAutoregressiveDecoderFn = Callable[
  [*RunAutoregressiveDecoderInputs],
  tuple[OneHotProteinSequence, Logits],
]

RunSTEAutoregressiveDecoderInputs = tuple[
  PRNGKeyArray,
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AtomMask,
  AutoRegressiveMask,
]
RunSTEAutoregressiveDecoderFn = Callable[
  [*RunSTEAutoregressiveDecoderInputs],
  tuple[OneHotProteinSequence, Logits],
]
