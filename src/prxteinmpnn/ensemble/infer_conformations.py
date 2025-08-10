"""Conformational-inference from ProteinMPNN logits."""

import enum

import jax.numpy as jnp
from gmmx import GaussianMixtureModelJax
from jax.scipy.special import entr
from jaxtyping import PRNGKeyArray

from prxteinmpnn.ensemble.dbscan import (
  ResidueConformationalStates,
  compute_component_distances,
  dbscan_cluster,
)
from prxteinmpnn.ensemble.gmm import make_fit_gmm
from prxteinmpnn.ensemble.residue_states import residue_states_from_ensemble
from prxteinmpnn.utils.data_structures import ProteinEnsemble
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.entropy import posterior_mean_std
from prxteinmpnn.utils.types import EdgeFeatures, InputBias, Logits, ModelParameters, NodeFeatures


class ConformationalInferenceStrategy(enum.Enum):
  """Determines what features to use for conformational inference.

  Options are LOGITS (output conditional logits), NODE_FEATURES (decoded node features),
  and EDGE_FEATURES (edge features from the encoder).
  """

  LOGITS = "logits"
  NODE_FEATURES = "node_features"
  EDGE_FEATURES = "edge_features"


def _infer_residue_states(
  gmm: GaussianMixtureModelJax,
  features: Logits | NodeFeatures | EdgeFeatures,
  eps_std_scale: float = 1.0,
  min_cluster_weight: float = 0.01,
) -> ResidueConformationalStates:
  """Infer residue states by clustering a GMM fit on input features.

  Args:
    gmm: Fitted GaussianMixtureModelJax object.
    features: Input features (logits or message), shape compatible with gmm.predict_proba.
    eps_std_scale: Scaling factor for DBSCAN epsilon.
    min_cluster_weight: Minimum cluster weight threshold.

  Returns:
    ResidueConformationalStates containing clustering results and statistics.

  """
  distance_matrix = compute_component_distances(gmm.means)
  component_weights = gmm.weights
  responsibility_matrix = jnp.squeeze(gmm.predict_proba(features), axis=(2, 3))
  triu_indices = jnp.triu_indices_from(distance_matrix, k=1)
  eps = 1.0 - eps_std_scale * jnp.std(distance_matrix[triu_indices])

  cluster_result = dbscan_cluster(
    distance_matrix,
    component_weights,
    responsibility_matrix,
    eps,
    min_cluster_weight,
  )

  state_responsibilities = responsibility_matrix @ cluster_result.coarse_graining_matrix.T
  state_trajectory = jnp.argmax(state_responsibilities, axis=1)

  states, counts = jnp.unique(
    state_trajectory,
    size=gmm.n_components,
    fill_value=-1,
    return_counts=True,
  )
  n_states = jnp.sum(states != -1)
  mle_entropy = entr(counts[counts > 0] / counts.sum()).sum()
  # Convert counts to float to avoid digamma dtype error
  _, mle_entropy_se = posterior_mean_std(counts[counts > 0].astype(jnp.float32))

  return ResidueConformationalStates(
    n_states=n_states,
    mle_entropy=mle_entropy,
    mle_entropy_se=mle_entropy_se,
    state_trajectory=state_trajectory,
    state_counts=counts,
    cluster_entropy=cluster_result.plug_in_entropy,
    cluster_probabilities=cluster_result.state_probabilities,
    dbscan_eps=eps,
    min_cluster_weight=min_cluster_weight,
    coarse_graining_matrix=cluster_result.coarse_graining_matrix,
  )


def infer_residue_states(
  gmm: GaussianMixtureModelJax,
  features: Logits | NodeFeatures | EdgeFeatures,
  eps_std_scale: float = 1.0,
  min_cluster_weight: float = 0.01,
) -> ResidueConformationalStates:
  """Infer residue states from features using a fitted GMM.

  Args:
    gmm: Fitted GaussianMixtureModelJax object.
    features: Input features, shape (n_timesteps, n_residues, n_features).
    eps_std_scale: Scaling factor for DBSCAN epsilon.
    min_cluster_weight: Minimum cluster weight threshold.

  Returns:
    ResidueConformationalStates containing clustering results and statistics.

  """
  return _infer_residue_states(
    gmm=gmm,
    features=features,
    eps_std_scale=eps_std_scale,
    min_cluster_weight=min_cluster_weight,
  )


def infer_conformations(
  prng_key: PRNGKeyArray,
  model_parameters: ModelParameters,
  inference_strategy: ConformationalInferenceStrategy,
  decoding_order_fn: DecodingOrderFn,
  ensemble: ProteinEnsemble,
  bias: InputBias | None = None,
  gmm_n_components: int = 100,
  eps_std_scale: float = 1.0,
  min_cluster_weight: float = 0.01,
) -> ResidueConformationalStates:
  """Infer conformations from a trajectory.

  Args:
    prng_key: JAX pseudo-random number generator key. Used for stochastic processes during sampling.
    model_parameters: A ModelParameters object containing the parameters for the protein model.
    inference_strategy: Conformational inference strategy enum to determine what features
      are used for conformational inference.
    decoding_order_fn: A function that defines the order in which residues are sampled.
    ensemble: A ProteinEnsemble object representing the initial protein structure.
    bias: An optional InputBias object to apply biases during sampling, such as a residue-specific
      bias. Defaults to None.
    gmm_n_components: The number of components to use when fitting the Gaussian Mixture Model.
      Defaults to 100.
    eps_std_scale: Scaling factor for DBSCAN epsilon.
    min_cluster_weight: Minimum cluster weight threshold.

  Returns:
    A ResidueConformationalStates object containing the inferred conformational states from the
    generated trajectory.

  Raises:
    ValueError: Invalid inference strategy used or empty input array.

  Example:
    >>> import jax.random as jr
    >>> prng_key = jr.PRNGKey(0)
    >>> # Assume `model_parameters`, `decoding_order_fn`, `ensemble`, and `sampling_strategy` are
    defined
    >>> # ... (e.g., by loading a pre-trained model and creating a protein ensemble)
    >>> conformations = infer_conformations(
    >>>   prng_key=prng_key,
    >>>   model_parameters=model_parameters,
    >>>   decoding_order_fn=decoding_order_fn,
    >>>   ensemble=ensemble,
    >>>   sampling_strategy=sampling_strategy,
    >>>   iterations=100
    >>> )
    >>> print(conformations.n_states)
    5

  """
  if not isinstance(inference_strategy, ConformationalInferenceStrategy):
    msg = "Invalid inference strategy used."
    raise TypeError(msg)

  residue_states_generator = residue_states_from_ensemble(
    prng_key=prng_key,
    model_parameters=model_parameters,
    decoding_order_fn=decoding_order_fn,
    ensemble=ensemble,
    bias=bias,
  )

  match inference_strategy:
    case ConformationalInferenceStrategy.LOGITS:
      states = (states[0] for _, states, _ in residue_states_generator)
    case ConformationalInferenceStrategy.NODE_FEATURES:
      states = (states[1] for _, states, _ in residue_states_generator)
    case ConformationalInferenceStrategy.EDGE_FEATURES:
      states = (states[2] for _, states, _ in residue_states_generator)
    case _:
      msg = "Invalid inference strategy used."
      raise ValueError(msg)

  all_states = jnp.array(tuple(states))

  if all_states.size == 0:
    msg = "Input array for GMM fitting cannot be empty."
    raise ValueError(msg)

  gmm_fitter = make_fit_gmm(n_components=gmm_n_components, n_features=all_states.shape[-1])
  gmm = gmm_fitter(all_states)

  return infer_residue_states(
    gmm=gmm,
    features=all_states,
    eps_std_scale=eps_std_scale,
    min_cluster_weight=min_cluster_weight,
  )
