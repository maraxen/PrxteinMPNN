"""Structural protocols and type aliases for prxteinmpnn.

ModelProtocol — structural seam over model implementations, unifying protein and ligand variants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import equinox as eqx
import jax
from jaxtyping import Float, PRNGKeyArray

if TYPE_CHECKING:
    from prxteinmpnn.model.capabilities import ModelCapabilities
    from prxteinmpnn.model.decoder import Decoder
    from prxteinmpnn.model.encoder import Encoder, PhysicsEncoder
    from prxteinmpnn.model.features import ProteinFeatures
    from prxteinmpnn.model.ligand_features import ProteinFeaturesLigand
    from prxteinmpnn.types.arrays import (
        AlphaCarbonMask,
        AutoRegressiveMask,
        BackboneNoise,
        ChainIndex,
        DecodingOrder,
        Logits,
        ProteinSequence,
        ResidueIndex,
        StructureAtomicCoordinates,
    )
    from prxteinmpnn.types.bundles import InferenceBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.stages import StageSet


@runtime_checkable
class ConditionalLogitsFn(Protocol):
    def __call__(
        self,
        prng_key: PRNGKeyArray,
        structure_coordinates: StructureAtomicCoordinates,
        mask: AlphaCarbonMask,
        residue_index: ResidueIndex,
        chain_index: ChainIndex,
        sequence: ProteinSequence,
        ar_mask: AutoRegressiveMask | None = None,
        backbone_noise: BackboneNoise | None = None,
        structure_mapping: jax.Array | None = None,
    ) -> Logits:
        """Conditional logits (single graph); optional ``structure_mapping`` for multistate encode."""


@runtime_checkable
class UnconditionalLogitsFn(Protocol):
    def __call__(
        self,
        prng_key: jax.Array,
        structure_coordinates: jax.Array,
        mask: jax.Array,
        residue_index: jax.Array,
        chain_index: jax.Array,
        ar_mask: jax.Array | None = None,
        backbone_noise: jax.Array | None = None,
    ) -> jax.Array:
        """Dense single-graph unconditional logits."""




@runtime_checkable
class SamplerFn(Protocol):
    """Unified sequence sampling protocol."""

    def __call__(
        self,
        prng_key: PRNGKeyArray,
        bundle: InferenceBundle,
        config: InferenceConfig,
        stage_set: StageSet,
    ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
        """Sample sequences and return (sequence, logits, decoding_order)."""
        ...


@runtime_checkable
class ScoreFn(Protocol):
    """Unified sequence scoring protocol."""

    def __call__(
        self,
        prng_key: PRNGKeyArray,
        bundle: InferenceBundle,
        config: InferenceConfig,
        stage_set: StageSet,
    ) -> tuple[Float, Logits, DecodingOrder]:
        """Score sequence and return (score, logits, decoding_order)."""
        ...


@runtime_checkable
class DesignSink(Protocol):
    """Host-side consumer for design tensor payloads emitted via ``io_callback``."""

    def on_sampling_sequences_logits(
        self,
        batch_idx: object,
        batch_count: object,
        chunk_start: object,
        chunk_count: object,
        sequences_host: object,
        logits_host: object,
    ) -> None:
        """Record per-batch sequences/logits host tensors."""

    def on_scoring_scores_logits(
        self,
        batch_idx: object,
        batch_count: object,
        scores_host: object,
        logits_host: object,
    ) -> None:
        """Record per-batch scores/logits host tensors."""


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural protocol over prxteinmpnn model implementations.

    Satisfied by PrxteinMPNN, PrxteinLigandMPNN, DiffusionPrxteinMPNN.
    """

    features: ProteinFeatures | ProteinFeaturesLigand
    encoder: Encoder | PhysicsEncoder
    decoder: Decoder
    w_out: eqx.nn.Linear
    w_s_embed: eqx.nn.Embedding
    capabilities: ModelCapabilities

    def __call__(self, key: PRNGKeyArray, **kwargs: jax.Array | str | float | bool | None) -> tuple[jax.Array, jax.Array, jax.Array]:
        ...

    @classmethod
    def stage_schema(cls) -> dict[str, type | None]:
        ...


@runtime_checkable
class Pipeline(Protocol):
    """Callable protocol for model pipeline implementations."""

    def __call__(
        self,
        module: ModelProtocol,
        key: PRNGKeyArray,
        inputs: Any,
        *,
        stage_set: StageSet,
    ) -> Any:
        ...


__all__ = [
    "ConditionalLogitsFn",
    "DesignSink",
    "ModelProtocol",
    "Pipeline",
    "SamplerFn",
    "ScoreFn",
    "UnconditionalLogitsFn",
]
