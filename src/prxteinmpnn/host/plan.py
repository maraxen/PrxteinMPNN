"""Batch planning and scheduling logic for sampling operations."""

from __future__ import annotations

import dataclasses
import logging
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from prxteinmpnn.tiling.planner import BatchPlan, BatchPlanner, estimate_memory_theoretical
from prxteinmpnn.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES

if TYPE_CHECKING:
    from collections.abc import Sequence
    from jaxtyping import PRNGKeyArray
    from prxteinmpnn.run.specs import SamplingSpecification
    from prxteinmpnn.types.arrays import ProteinSequence, Logits

logger = logging.getLogger(__name__)
_batch_logger = logging.getLogger(__name__ + ".batch_plan")


# Axis name constants for batch planning
class AxisNames:
    """Named access to batch axes to avoid hardcoded strings."""
    N_STRUCTURES = "n_structures"
    N_SAMPLES = "n_samples"
    N_TEMPERATURES = "n_temperatures"
    N_NOISES = "n_noises"


def make_sampling_planner(
    spec: SamplingSpecification,
    param_bytes: float = 0.0,
    headroom: float = 0.80,
    activation_multiplier: float = 2.5,
) -> BatchPlan:
    """Create a BatchPlan for _sample_batch dispatch with advisory logging.

    Args:
        spec: The sampling specification containing batch size and temperature/noise parameters.
        param_bytes: Estimated model parameter size in bytes.
        headroom: Fraction of device memory to use (0.80 = 80% headroom).
        activation_multiplier: Multiplier for activation memory estimation.

    Returns:
        A BatchPlan describing the batch size decisions for each axis.
    """
    try:
        limit = jax.devices()[0].memory_stats()["bytes_limit"]
    except Exception:
        limit = 4 * 1024**3
    budget = limit * headroom - param_bytes
    axes = [
        dataclasses.replace(N_STRUCTURES, cardinality=max(1, getattr(spec, "batch_size", 1) or 1)),
        dataclasses.replace(N_SAMPLES, cardinality=max(1, getattr(spec, "samples_batch_size", 128) or 128)),
        dataclasses.replace(N_TEMPERATURES, cardinality=max(1, len(getattr(spec, "temperature", [1.0])))),
        dataclasses.replace(N_NOISES, cardinality=max(1, len(getattr(spec, "backbone_noise", [0.0])))),
    ]
    return BatchPlanner(
        axes=axes,
        budget_bytes=budget,
        estimate_memory=lambda ds: estimate_memory_theoretical(ds, 1.0, activation_multiplier),
    ).plan()


def extract_batch_sizes(plan: BatchPlan) -> tuple[int, int, int, int]:
    """Extract batch sizes for all sampling axes from a BatchPlan.

    Args:
        plan: The BatchPlan from make_sampling_planner.

    Returns:
        A tuple of (structures_bs, samples_bs, temps_bs, noises_bs).
    """
    structures_bs = plan.decision_for(AxisNames.N_STRUCTURES).batch_size
    samples_bs = plan.decision_for(AxisNames.N_SAMPLES).batch_size
    temps_bs = plan.decision_for(AxisNames.N_TEMPERATURES).batch_size
    noises_bs = plan.decision_for(AxisNames.N_NOISES).batch_size
    return structures_bs, samples_bs, temps_bs, noises_bs


def compute_sample_keys(
    base_key: PRNGKeyArray,
    target_num_samples: int,
    chunk_sample_start: int | None = None,
    grid_lineage_sample_start: int | None = None,
) -> jax.Array:
    """Compute deterministic PRNG keys for all samples based on indexing strategy.

    Args:
        base_key: The base PRNG key for folding.
        target_num_samples: Number of samples to generate keys for.
        chunk_sample_start: Optional explicit chunk start index.
        grid_lineage_sample_start: Optional grid lineage sample start index.

    Returns:
        JAX array of shape (target_num_samples,) containing folded keys.
    """
    sample_indices = np.arange(target_num_samples, dtype=np.int32)
    if chunk_sample_start is not None:
        sample_indices += int(chunk_sample_start)
    elif grid_lineage_sample_start is not None:
        sample_indices += int(grid_lineage_sample_start)

    # Generate keys for each sample via fold_in
    return jax.vmap(lambda idx: jax.random.fold_in(base_key, idx))(sample_indices)


def resolve_target_samples(
    spec: SamplingSpecification,
    chunk_sample_count: int | None = None,
    grid_lineage: dict[str, int | str] | None = None,
) -> int:
    """Resolve the target number of samples for this batch.

    Prioritizes explicit chunk_sample_count, then grid lineage sample_count,
    then spec.num_samples as fallback.

    Args:
        spec: The sampling specification.
        chunk_sample_count: Optional explicit sample count for this chunk.
        grid_lineage: Optional grid lineage dict with 'sample_count' key.

    Returns:
        The resolved target sample count as a positive integer.

    Raises:
        ValueError: If resolved sample count is not positive.
    """
    if chunk_sample_count is not None:
        target = int(chunk_sample_count)
    elif grid_lineage is not None:
        target = int(grid_lineage["sample_count"])
    else:
        target = int(spec.num_samples)

    if target <= 0:
        msg = "num_samples must be positive."
        raise ValueError(msg)
    return target


def resolve_chunk_size(
    spec: SamplingSpecification,
    total_num_samples: int,
    grid_lineage: dict[str, int | str] | None = None,
) -> int:
    """Resolve the chunk size for streaming sample output.

    Uses spec.samples_chunk_size if set, otherwise uses grid_lineage['sample_count']
    if grid_lineage exists, otherwise uses total_num_samples.

    Args:
        spec: The sampling specification.
        total_num_samples: Total number of samples to be generated.
        grid_lineage: Optional grid lineage dict.

    Returns:
        The chunk size as a positive integer.
    """
    if hasattr(spec, "samples_chunk_size") and spec.samples_chunk_size:
        return int(spec.samples_chunk_size)
    elif grid_lineage is not None:
        return int(grid_lineage["sample_count"])
    else:
        return total_num_samples


def resolve_sample_start(
    grid_lineage: dict[str, int | str] | None = None,
) -> int:
    """Resolve the sample start index from grid lineage or default to 0.

    Args:
        grid_lineage: Optional grid lineage dict with 'sample_start' key.

    Returns:
        The sample start index (0-based).
    """
    return int(grid_lineage["sample_start"]) if grid_lineage is not None else 0


