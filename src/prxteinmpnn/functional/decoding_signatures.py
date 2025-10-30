"""Decoder function signatures."""

from __future__ import annotations

from collections.abc import Callable

from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
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
  AlphaCarbonMask,
  ModelParameters,
  float,
]
DecoderNormalizeFn = Callable[[*DecoderNormalizeInputs], NodeFeatures]


MaskedAttentionDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AlphaCarbonMask,
  AttentionMask,
  ModelParameters,
  float,
]
MaskedAttentionDecoderFn = Callable[[*MaskedAttentionDecoderInputs], NodeFeatures]
DecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AlphaCarbonMask,
  ModelParameters,
  float,
]
DecoderFn = Callable[[*DecoderInputs], NodeFeatures]


RunDecoderInputs = tuple[NodeFeatures, EdgeFeatures, AlphaCarbonMask]
RunDecoderFn = Callable[[*RunDecoderInputs], NodeFeatures]
RunMaskedAttentionDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AlphaCarbonMask,
  AttentionMask,
]
RunMaskedAttentionDecoderFn = Callable[[*RunMaskedAttentionDecoderInputs], NodeFeatures]
RunConditionalDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AlphaCarbonMask,
  AutoRegressiveMask,
  ProteinSequence,
]
RunConditionalDecoderFn = Callable[[*RunConditionalDecoderInputs], NodeFeatures]
RunAutoregressiveDecoderInputs = tuple[
  PRNGKeyArray,
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AlphaCarbonMask,
  AutoRegressiveMask,
  Float | None,
  Logits | None,
]
RunAutoregressiveDecoderFn = Callable[
  [*RunAutoregressiveDecoderInputs],
  tuple[OneHotProteinSequence, Logits],
]
