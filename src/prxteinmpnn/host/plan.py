"""Batch planning and scheduling logic for sampling operations.

Also includes InferencePlan and related components for unified inference dispatch.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.tiling.axes import N_NOISES, N_SAMPLES, N_STRUCTURES, N_TEMPERATURES
from prxteinmpnn.tiling.planner import BatchPlan, BatchPlanner, estimate_memory_theoretical

if TYPE_CHECKING:
    from jaxtyping import PRNGKeyArray

    from prxteinmpnn.run.specs import SamplingSpecification
    from prxteinmpnn.types.arrays import Logits
    from prxteinmpnn.types.bundles import InferenceBundle
    from prxteinmpnn.types.configs import InferenceConfig
    from prxteinmpnn.types.protocols import ModelProtocol

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

    Parameters
    ----------
    spec : SamplingSpecification
        Sampling specification containing batch size and temperature/noise parameters.
    param_bytes : float, optional
        Estimated model parameter size in bytes. Default 0.0.
    headroom : float, optional
        Fraction of device memory to use. Default 0.80 (80% headroom).
    activation_multiplier : float, optional
        Multiplier for activation memory estimation. Default 2.5.

    Returns
    -------
    BatchPlan
        Batch size decisions for each sampling axis (structures, samples, temps, noises).
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

    Parameters
    ----------
    plan : BatchPlan
        Batch plan from make_sampling_planner.

    Returns
    -------
    tuple[int, int, int, int]
        Tuple of (structures_bs, samples_bs, temps_bs, noises_bs).
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
    """Compute deterministic PRNG keys for all samples via fold_in.

    Parameters
    ----------
    base_key : PRNGKeyArray
        Base PRNG key for folding.
    target_num_samples : int
        Number of samples to generate keys for.
    chunk_sample_start : int | None, optional
        Explicit chunk start index. Default None.
    grid_lineage_sample_start : int | None, optional
        Grid lineage sample start index. Default None.

    Returns
    -------
    jax.Array
        Array of shape ``(target_num_samples,)`` containing folded keys.
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

    Parameters
    ----------
    spec : SamplingSpecification
        Sampling specification.
    chunk_sample_count : int | None, optional
        Explicit sample count for this chunk. Default None.
    grid_lineage : dict[str, int | str] | None, optional
        Grid lineage dict with 'sample_count' key. Default None.

    Returns
    -------
    int
        Resolved target sample count (positive).

    Raises
    ------
    ValueError
        If resolved sample count is not positive.
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

    Uses spec.samples_chunk_size if set, otherwise grid_lineage['sample_count']
    if grid_lineage exists, otherwise total_num_samples.

    Parameters
    ----------
    spec : SamplingSpecification
        Sampling specification.
    total_num_samples : int
        Total number of samples to be generated.
    grid_lineage : dict[str, int | str] | None, optional
        Grid lineage dict. Default None.

    Returns
    -------
    int
        Chunk size (positive).
    """
    if hasattr(spec, "samples_chunk_size") and spec.samples_chunk_size:
        return int(spec.samples_chunk_size)
    if grid_lineage is not None:
        return int(grid_lineage["sample_count"])
    return total_num_samples


def resolve_sample_start(
    grid_lineage: dict[str, int | str] | None = None,
) -> int:
    """Resolve the sample start index from grid lineage or default to 0.

    Parameters
    ----------
    grid_lineage : dict[str, int | str] | None, optional
        Grid lineage dict with 'sample_start' key. Default None.

    Returns
    -------
    int
        Sample start index (0-based).
    """
    return int(grid_lineage["sample_start"]) if grid_lineage is not None else 0


# ---------------------------------------------------------------------------
# COMP-8: InferencePlan and related components for unified inference dispatch
# ---------------------------------------------------------------------------


class InferenceComponents(NamedTuple):
    """Resolved inference components for encode-once/decode-many pipeline.

    Parameters
    ----------
    encode_fn : Callable
        Encoder forward pass. Signature:
        ``(bundle: InferenceBundle, key: PRNGKeyArray, config: InferenceConfig) → EncoderOutput``
    driver : Callable
        Decode driver. Signature:
        ``(model, key, enc, conditioning, wave, config, stage_set) → result``
        Routes to sample_autoregressive or score_conditional based on stage_set.
    stage_set : Any
        StageSet instance with all slots wired (logit_transform, ar_logit_transform,
        decode_step, sample_step, tie_group_fuse). Ready for JIT.
    """
    encode_fn: Callable
    driver: Callable
    stage_set: Any  # StageSet


@dataclass
class InferencePlan:
    """Resolved inference plan with encode-once/decode-many pattern.

    Encodes geometry and ligand context once, then reuses encoder output for
    multiple decode passes (sampling or scoring) with different stage_set instances.

    Parameters
    ----------
    model : Any
        Parameterized protein/ligand model (carries JAX arrays).
    components : InferenceComponents
        Resolved components: encode_fn, driver, stage_set.

    Notes
    -----
    `.sample()` and `.score()` invoke the same encode → decode pipeline but with
    different stage_set configurations. The decode_step and sample_step fields in
    stage_set determine the output (sampled sequence or logits).
    """

    model: Any
    components: InferenceComponents

    @property
    def stage_set(self) -> Any:
        """Access the wired StageSet directly."""
        return self.components.stage_set

    def sample(self, bundle: InferenceBundle, key: PRNGKeyArray, config: InferenceConfig) -> Any:
        """Encode and sample from the pipeline.

        Parameters
        ----------
        bundle : InferenceBundle
            Input geometry, conditioning, ligand, and wave schedule.
        key : PRNGKeyArray
            PRNG key for sampling.
        config : InferenceConfig
            Inference configuration (batch, device, etc.).

        Returns
        -------
        Any
            Sampled sequence (or auxiliary output from driver).
        """
        enc = self.components.encode_fn(bundle, key, config)
        return self.components.driver(
            self.model, key, enc, bundle.conditioning, bundle.wave, config, self.components.stage_set,
        )

    def score(self, bundle: InferenceBundle, key: PRNGKeyArray, config: InferenceConfig) -> Logits:
        """Encode and score the pipeline.

        Parameters
        ----------
        bundle : InferenceBundle
            Input geometry, conditioning, ligand, and wave schedule.
        key : PRNGKeyArray
            PRNG key for any stochastic operations.
        config : InferenceConfig
            Inference configuration (batch, device, etc.).

        Returns
        -------
        Logits
            Logit scores per position per amino acid.
        """
        enc = self.components.encode_fn(bundle, key, config)
        return self.components.driver(
            self.model, key, enc, bundle.conditioning, bundle.wave, config, self.components.stage_set,
        )


def make_inference_plan(model: ModelProtocol, spec: Any) -> InferencePlan:
    """Factory: resolve and create an InferencePlan from model and spec.

    Assembles the inference pipeline by resolving encode_fn (from use_rolling_state),
    wiring logit_transform (from multi_state_strategy), and instantiating stage_set
    with ARLogitFuse and TieGroupProductOfExperts.

    Parameters
    ----------
    model : ModelProtocol
        Parameterized model with decoder, encoder, and embeddings.
    spec : Any
        Specification with attributes: use_rolling_state, multi_state_strategy,
        multi_state_temperature, state_weights.

    Returns
    -------
    InferencePlan
        Ready-to-use inference plan for sampling/scoring.

    References
    ----------
    .. [ProteinMPNN] Dauparas, J., et al. "Robust deep learning-based protein
       sequence design using ProteinMPNN." *Science* 378(6615):49-56 (2022).
       https://doi.org/10.1126/science.add2187

    .. [LigandMPNN] Dauparas, J., et al. "Atomic context-conditioned protein
       sequence design using LigandMPNN." *Nature Methods* 22(4):717-723 (2025).
       https://doi.org/10.1038/s41592-025-02626-1
    """
    from prxteinmpnn.inference import driver as driver_module
    from prxteinmpnn.inference.encode import make_encode_fn
    from prxteinmpnn.inference.logits import LOGIT_STRATEGIES, ARLogitFuse, TieGroupProductOfExperts
    from prxteinmpnn.types.stages import StageSet

    use_rolling_state = getattr(spec, "use_rolling_state", False)
    encode_fn = make_encode_fn(model, use_rolling_state=use_rolling_state)

    strategy_name = getattr(spec, "multi_state_strategy", None) or "arithmetic_mean"
    strategy_temp = getattr(spec, "multi_state_temperature", 1.0) or 1.0
    state_weights = getattr(spec, "state_weights", None)

    strategy_cls = LOGIT_STRATEGIES.get(strategy_name)
    if strategy_cls is None:
        msg = f"Logit strategy '{strategy_name}' not found in registry"
        raise ValueError(msg)

    weights = jnp.asarray(state_weights, dtype=jnp.float32) if state_weights is not None else jnp.ones(1, dtype=jnp.float32)

    try:
        logit_transform = strategy_cls(weights, temperature=strategy_temp)
    except TypeError:
        logit_transform = strategy_cls(weights)

    stage_set = StageSet(
        logit_transform=logit_transform,
        ar_logit_transform=ARLogitFuse(),
        decode_step=None,
        sample_step=None,
        tie_group_fuse=TieGroupProductOfExperts(),
    )

    components = InferenceComponents(
        encode_fn=encode_fn,
        driver=driver_module.decode,
        stage_set=stage_set,
    )

    return InferencePlan(model=model, components=components)

