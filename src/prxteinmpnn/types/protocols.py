"""Core protocols defining the interfaces for models, samplers, and sinks."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
import jax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.capabilities import ModelCapabilities

@runtime_checkable
class ModelProtocol(Protocol):
    """Structural protocol over prxteinmpnn model implementations.
    
    Models are strictly parameter containers with a single forward pass.
    """
    features: Any
    encoder: Any
    decoder: Any
    w_out: Any
    w_s_embed: Any
    capabilities: ModelCapabilities

    def __call__(
        self,
        key: PRNGKeyArray,
        coords: jax.Array,
        mask: jax.Array,
        residue_index: jax.Array,
        chain_index: jax.Array,
        **kwargs: Any
    ) -> Any:
        """Pure forward: features → encode → return encoded representation."""
        ...


@runtime_checkable
class SamplerFn(Protocol):
    """Unified sequence sampling protocol."""
    def __call__(self, prng_key: PRNGKeyArray, inputs: Any) -> Any: ...


@runtime_checkable
class ScoreFn(Protocol):
    """Unified sequence scoring protocol."""
    def __call__(self, prng_key: PRNGKeyArray, inputs: Any) -> Any: ...


@runtime_checkable
class DesignSink(Protocol):
    """Host-side consumer for design tensor payloads emitted via io_callback."""
    def on_sampling_sequences_logits(
        self,
        batch_idx: object,
        batch_count: object,
        chunk_start: object,
        chunk_count: object,
        sequences_host: object,
        logits_host: object,
    ) -> None: ...

    def on_scoring_scores_logits(
        self,
        batch_idx: object,
        batch_count: object,
        scores_host: object,
        logits_host: object,
    ) -> None: ...
