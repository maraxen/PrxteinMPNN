"""Core user interface for the PrxteinMPNN package."""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from prxteinmpnn.io.weights import MODEL_VERSION, MODEL_WEIGHTS

mp.set_start_method("spawn", force=True)


if TYPE_CHECKING:
  from io import StringIO

  from jaxtyping import ArrayLike

  from prxteinmpnn.ensemble.dbscan import ConformationalStates
  from prxteinmpnn.utils.catjac import CombineCatJacPairFn
  from prxteinmpnn.utils.decoding_order import DecodingOrderFn
  from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase

# Type aliases for convenience
ModelWeights = MODEL_WEIGHTS
ModelVersion = MODEL_VERSION


AlignmentStrategy = Literal["sequence", "structure"]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


def _loader_inputs(inputs: Sequence[str | StringIO] | str | StringIO) -> Sequence[str | StringIO]:
  return (inputs,) if not isinstance(inputs, Sequence) else inputs  # type: ignore[invalid-return-type]


@dataclass
class RunSpecification:
  """Configuration for running the model.

  Attributes:
      inputs: A sequence of input file paths or StringIO objects, or a single input.
      model_weights: The model weights to use (default is "original").
      model_version: The model version to use (default is "v_48_020.pkl").
      batch_size: The batch size to use (default is 32).
      backbone_noise: The backbone noise levels to use (default is (0.0,)).
                      Can be a single float or a sequence of floats.
      foldcomp_database: An optional path to a FoldComp database (default is None).
      ar_mask: An optional array-like mask for autoregressive positions (default is None).
      random_seed: The random seed to use (default is 42).
      chain_id: An optional chain ID to use (default is None).
      model: An optional model ID to use (default is None).
      altloc: The alternate location to use (default is "first").
      decoding_order_fn: An optional function to generate the decoding order (default is None).
      conformational_states: ConformationalStates to use for coarse graining the inference.
      max_length: Maximum sequence length for padding/truncation (default is 512).
                  Set to None to disable padding. Controls memory usage - smaller values
                  reduce memory but may cause recompilation for different sequence lengths.
      truncation_strategy: Strategy for handling sequences longer than max_length.
                          Options: "none" (default, no truncation), "random_crop", "center_crop".

  """

  inputs: Sequence[str | StringIO] | str | StringIO
  topology: str | Path | None = None
  model_weights: ModelWeights = "original"
  model_version: ModelVersion = "v_48_020"
  batch_size: int = 32
  backbone_noise: Sequence[float] | float = (0.0,)
  backbone_noise_mode: str = "direct"
  """Mode for backbone noise: 'direct' (Gaussian) or 'md' (Molecular Dynamics)."""
  
  md_min_steps: int = 500
  """Number of minimization steps for MD mode."""
  
  md_therm_steps: int = 1000
  """Number of thermalization steps for MD mode."""
  
  md_temperature: float = 300.0
  """Temperature (Kelvin) for MD thermalization."""
  estat_noise: Sequence[float] | float | None = None
  estat_noise_mode: Literal["direct", "thermal"] = "direct"
  vdw_noise: Sequence[float] | float | None = None
  vdw_noise_mode: Literal["direct", "thermal"] = "direct"
  use_electrostatics: bool = False
  use_vdw: bool = False
  foldcomp_database: FoldCompDatabase | None = None
  ar_mask: None | ArrayLike = None
  random_seed: int = 42
  chain_id: Sequence[str] | str | None = None
  model: int | None = None
  altloc: Literal["first", "all"] = "first"
  decoding_order_fn: DecodingOrderFn | None = None
  conformational_states: ConformationalStates | None = None
  cache_path: str | Path | None = None
  overwrite_cache: bool = False
  output_path: str | Path | None = None
  max_length: int | None = 512
  truncation_strategy: Literal["none", "random_crop", "center_crop"] = "none"

  # Data/Sharding
  use_preprocessed: bool = False
  preprocessed_index_path: str | Path | None = None
  split: str = "inference"

  # Tied-position logit averaging fields
  tied_positions: Sequence[tuple[int, int]] | Literal["auto", "direct"] | None = None
  pass_mode: Literal["inter", "intra"] = "intra"  # noqa: S105

  def __post_init__(self) -> None:
    """Post-initialization processing and validation for tied-position logit averaging."""
    if isinstance(self.backbone_noise, float):
      object.__setattr__(self, "backbone_noise", (self.backbone_noise,))
    if isinstance(self.estat_noise, float):
      object.__setattr__(self, "estat_noise", (self.estat_noise,))
    if self.estat_noise is not None:
      object.__setattr__(self, "use_electrostatics", True)

    if isinstance(self.vdw_noise, float):
      object.__setattr__(self, "vdw_noise", (self.vdw_noise,))
    if self.vdw_noise is not None:
      object.__setattr__(self, "use_vdw", True)
    if self.cache_path and isinstance(self.cache_path, str):
      object.__setattr__(self, "cache_path", Path(self.cache_path))
    # Validation for tied_positions and pass_mode
    if self.tied_positions in ("auto", "direct") and self.pass_mode != "inter":  # noqa: S105
      msg = (
        f"If tied_positions is '{self.tied_positions}', pass_mode must be 'inter'. "
        f"Got pass_mode='{self.pass_mode}'."
      )
      raise ValueError(msg)


@dataclass
class ScoringSpecification(RunSpecification):
  """Configuration for scoring sequences.

  Attributes:
      sequences_to_score: A sequence of amino acid sequences to score.
      temperature: The temperature for scoring (default is 1.0).
      return_logits: Whether to return the raw logits (default is False).
      return_decoding_orders: Whether to return decoding orders (default is False).
      return_all_scores: Whether to return scores for all sequences (default is False).
      score_batch_size: The batch size for scoring sequences (default is 16).
      output_h5_path: Optional path to an HDF5 file for streaming output.
      average_node_features: Whether to average node features (default is False).
      average_encoding_mode: Mode for averaging encodings (default is "inputs_and_noise").
      noise_batch_size: The batch size for noise levels (default is 4).

  """

  sequences_to_score: Sequence[str] = ()
  temperature: float = 1.0
  return_logits: bool = False
  return_decoding_orders: bool = False
  return_all_scores: bool = False
  score_batch_size: int = 16
  output_h5_path: str | Path | None = None
  average_node_features: bool = False
  average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"] = "inputs_and_noise"
  noise_batch_size: int = 4

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if not self.sequences_to_score:
      msg = (
        "No sequences provided for scoring."
        "`sequences_to_score` must be a non-empty list of strings."
      )
      raise ValueError(msg)
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))


@dataclass
class SamplingSpecification(RunSpecification):
  """Configuration for sampling sequences."""

  num_samples: int = 1
  sampling_strategy: Literal["temperature", "straight_through"] = "temperature"
  temperature: Sequence[float] | float = 0.1
  bias: ArrayLike | None = None
  fixed_positions: ArrayLike | None = None
  iterations: int | None = None
  learning_rate: float | None = None
  output_h5_path: str | Path | None = None
  samples_batch_size: int = 16
  noise_batch_size: int = 1
  temperature_batch_size: int = 1
  average_node_features: bool = False
  average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"] = "inputs_and_noise"
  average_logits: None | Literal["structures", "noise", "both"] = None
  multi_state_strategy: Literal["arithmetic_mean", "geometric_mean", "product"] = "arithmetic_mean"
  compute_pseudo_perplexity: bool = False

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if isinstance(self.temperature, float):
      object.__setattr__(self, "temperature", (self.temperature,))
    if self.sampling_strategy == "straight_through" and (
      self.iterations is None or self.learning_rate is None
    ):
      msg = "For 'straight_through' sampling, 'iterations' and 'learning_rate' must be provided."
      raise ValueError(msg)
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))


@dataclass
class JacobianSpecification(RunSpecification):
  """Configuration for computing categorical Jacobians."""

  noise_batch_size: int = 1
  jacobian_batch_size: int = 16
  average_encodings: bool = True
  average_encoding_mode: Literal["inputs", "noise_levels", "inputs_and_noise"] = "inputs_and_noise"
  average_logits: None | Literal["structures", "noise", "both"] = None
  combine: bool = False
  combine_batch_size: int = 8
  combine_noise_batch_size: int = 1
  combine_weights: ArrayLike | None = None
  combine_fn: CombineCatJacPairFn | None = None
  combine_fn_kwargs: dict[str, Any] | None = None
  output_h5_path: str | Path | None = None
  compute_apc: bool = True
  apc_batch_size: int = 8
  apc_residue_batch_size: int = 1000

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))


@dataclass
class ConformationalInferenceSpecification(RunSpecification):
  """Configuration for deriving states from a protein ensemble.

  Attributes:
      output_h5_path: Optional path to an HDF5 file for streaming output.
      batch_size: The batch size for processing proteins (default is 8).

  """

  output_h5_path: str | Path | None = None
  batch_size: int = 8
  inference_strategy: Literal["unconditional", "conditional", "vmm"] = "unconditional"
  inference_features: Sequence[Literal["logits", "node_features", "edge_features"]] = ("logits",)
  mode: Literal["global", "per"] = "global"
  covariance_type: Literal["full", "diag"] = "diag"
  gmm_n_components: int = 100
  eps_std_scale: float = 1.0
  min_cluster_weight: float = 0.01
  preprocessing_mode: Literal["pca"] | None = None
  gmm_init: Literal["kmeans", "random"] = "kmeans"
  gmm_max_iters: int = 100
  kmeans_max_iters: int = 200
  pca_n_components: int = 20
  pca_solver: Literal["full", "randomized"] = "full"
  pca_rng_seed: int = 0
  gmm_min_iters: int = 10
  covariance_regularization: float = 1e-3

  reference_sequence: str | None = None

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))


MIN_PAIR = 2


@dataclass
class InspectionSpecification(RunSpecification):
  """Configuration for inspecting model encodings and features."""

  output_h5_path: str | Path | None = None
  inspection_features: Sequence[
    Literal[
      "unconditional_logits",
      "encoded_node_features",
      "edge_features",
      "decoded_node_features",
      "conditional_logits",
    ]
  ] = ("unconditional_logits",)
  distance_matrix: bool = False
  distance_matrix_method: Literal["ca", "cb", "backbone_average", "closest_atom"] = "ca"
  cross_input_similarity: bool = False
  similarity_metric: Literal[
    "rmsd",
    "tm-score",
    "gdt_ts",
    "gdt_ha",
    "cosine",
  ] = "rmsd"

  def __post_init__(self) -> None:
    """Post-initialization processing."""
    super().__post_init__()
    if self.output_h5_path and isinstance(self.output_h5_path, str):
      object.__setattr__(self, "output_h5_path", Path(self.output_h5_path))
    if self.cross_input_similarity and len(_loader_inputs(self.inputs)) < MIN_PAIR:
      msg = f"Cross-input similarity requires at least {MIN_PAIR} input structures."
      raise ValueError(msg)


Specs = (
  RunSpecification
  | ScoringSpecification
  | SamplingSpecification
  | JacobianSpecification
  | InspectionSpecification
)
