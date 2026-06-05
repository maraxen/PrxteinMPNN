"""Potts model with differentiable TRW inference on k-NN graphs from ProteinFeatures.

Architecture: PottsModel is a parallel model family (NOT a StageSet consumer).
See ADR 260605_potts-parallel-not-stageset for design rationale.

h and J carry a factor-of-2 from the directed-slot PottsMPNN convention
(counting each pairwise edge twice in the MRF). This factor is preserved
to maintain numerical consistency with weight recapture from mistypotts.
See etab_to_dense_h_j_w for reference.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

try:
  from mistypotts.potts_trw_spec import PottsTRWRunSpec
  from mistypotts.trw import DifferentiableTRW
except ImportError as e:
  msg = "mistypotts is required for Potts inference. Install via: pip install mistypotts"
  raise ImportError(msg) from e


class PottsParams(NamedTuple):
  """Inferred parameters from PottsModel forward pass.

  Attributes:
      marginals: Node marginals (N, num_aa)
      h: Node unary potentials (N, num_aa) with x2 scale factor
      J: Pairwise potentials (N, N, num_aa, num_aa) with x2 scale factor
      rho: Tree-reweighting parameters (N, N)
      W: Graph adjacency matrix (N, N)
  """

  marginals: Float[Array, "n num_aa"]
  h: Float[Array, "n num_aa"]
  J: Float[Array, "n n num_aa num_aa"]
  rho: Float[Array, "n n"]
  W: Float[Array, "n n"]


class PottsModel(eqx.Module):
  """Potts MPNN with TRW inference head on k-NN geometric graphs.

  This model:
  - Builds k-NN edges from structure coordinates using ProteinFeatures
  - Projects edge features and node context → unary (h) and pairwise (J) potentials
  - Runs differentiable tree-reweighted message passing (TRW) for node marginals
  - Supports multiple rho backends and loop strategies via PottsTRWRunSpec

  All static fields (hidden_dim, num_aa, k_neighbors) must be eqx.field(static=True)
  to enable JIT compilation and PyTree filtering.

  k_neighbors is sourced from model metadata (checkpoint) and is never exposed
  as a constructor argument to users.
  """

  hidden_dim: int = eqx.field(static=True)
  num_aa: int = eqx.field(static=True)
  k_neighbors: int = eqx.field(static=True)

  # Learned parameters
  node_lin: eqx.nn.Linear
  project_h: eqx.nn.Linear
  project_j: eqx.nn.Linear

  # TRW inference engine (static config)
  trw: DifferentiableTRW = eqx.field(static=True)
  trw_spec: PottsTRWRunSpec = eqx.field(static=True)

  # Optional auxiliary features dimension
  aux_node_dim: int = eqx.field(static=True)

  # Optional suppression of pairwise couplings
  suppress_pairwise: bool = eqx.field(static=True)

  def __init__(
    self,
    hidden_dim: int,
    num_aa: int,
    k_neighbors: int,
    edge_features_dim: int,
    trw_iters: int,
    *,
    key: PRNGKeyArray,
    trw_damping: float = 0.5,
    trw_rho_floor: float = 1e-6,
    aux_node_dim: int = 0,
    suppress_pairwise: bool = False,
    trw_spec: PottsTRWRunSpec | None = None,
  ) -> None:
    """Initialize PottsModel.

    Args:
        hidden_dim: Dimension of hidden node representations
        num_aa: Number of amino acid types (q in TRW; typically 21)
        k_neighbors: Number of k-NN neighbors (read-only from metadata)
        edge_features_dim: Dimension of edge features from ProteinFeatures
        trw_iters: Number of TRW message-passing iterations
        key: JAX random key for parameter initialization
        trw_damping: Damping factor for TRW message updates [0, 1]
        trw_rho_floor: Minimum threshold for tree-reweighting parameters
        aux_node_dim: Optional auxiliary node feature dimension
        suppress_pairwise: If True, zero out all pairwise couplings
        trw_spec: PottsTRWRunSpec with backend and solver config
                 (default: PottsTRWRunSpec.default_dense())
    """
    _, k1, k2, k3 = jax.random.split(key, 4)

    self.hidden_dim = int(hidden_dim)
    self.num_aa = int(num_aa)
    self.k_neighbors = int(k_neighbors)
    self.aux_node_dim = int(aux_node_dim)
    self.suppress_pairwise = bool(suppress_pairwise)

    # Projection layers
    node_in_dim = int(edge_features_dim) + self.aux_node_dim
    self.node_lin = eqx.nn.Linear(node_in_dim, self.hidden_dim, key=k1)
    self.project_h = eqx.nn.Linear(self.hidden_dim, self.num_aa, key=k2)

    # Pairwise projection: concatenate [h_i, h_j, edge_features]
    pair_in_dim = 2 * self.hidden_dim + int(edge_features_dim)
    self.project_j = eqx.nn.Linear(pair_in_dim, self.num_aa * self.num_aa, key=k3)

    # TRW engine
    self.trw_spec = PottsTRWRunSpec.default_dense() if trw_spec is None else trw_spec
    self.trw = DifferentiableTRW(
      q=self.num_aa,
      trw_iters=int(trw_iters),
      damping=float(trw_damping),
      rho_floor=float(trw_rho_floor),
      spec=self.trw_spec,
    )

  def __call__(
    self,
    key: PRNGKeyArray,
    coords: Float[Array, "n 37 3"],
    mask: Float[Array, " n"],
    residue_index: Int[Array, " n"],
    chain_index: Int[Array, " n"],
    aux_node_features: Array | None = None,
  ) -> tuple[Array, Array, Array, Array]:
    """Forward pass: extract Potts h/J from coords, run TRW.

    Args:
        key: JAX random key (for future stochastic features)
        coords: Structure coordinates shape (N, 37, 3) with B-factor padding
        mask: Valid residue indicator (N,) in [0, 1]
        residue_index: Residue index array (N,)
        chain_index: Chain assignment (N,)
        aux_node_features: Optional auxiliary features (N, aux_node_dim)

    Returns:
        tuple of four arrays:
        - marginals: Node marginals (N, num_aa)
        - h: Unary potentials (N, num_aa) with x2 scale factor
        - J: Pairwise potentials (N, N, num_aa, num_aa) with x2 scale factor
        - rho: Tree-reweighting parameters (N, N)
    """
    # In the current minimal implementation, we skip the ProteinFeatures extraction
    # and assume edge features are provided externally or computed from coords.
    # This is a placeholder for the full implementation.
    msg = (
      "PottsModel.__call__ requires integration with ProteinFeatures "
      "or external edge features. This is a placeholder."
    )
    raise NotImplementedError(msg)

  def infer_params(
    self,
    key: PRNGKeyArray,
    coords: Float[Array, "n 37 3"],
    mask: Float[Array, " n"],
    residue_index: Int[Array, " n"],
    chain_index: Int[Array, " n"],
    aux_node_features: Array | None = None,
  ) -> PottsParams:
    """Infer Potts parameters and return as namedtuple.

    Args:
        key: JAX random key
        coords: Structure coordinates (N, 37, 3)
        mask: Residue mask (N,)
        residue_index: Residue indices (N,)
        chain_index: Chain assignment (N,)
        aux_node_features: Optional auxiliary node features (N, aux_node_dim)

    Returns:
        PottsParams namedtuple with marginals, h, J, rho, W.
    """
    # Placeholder: integration with ProteinFeatures and graph construction pending.
    msg = "infer_params: integration with ProteinFeatures pending"
    raise NotImplementedError(msg)

  @staticmethod
  def log_prob(
    seq: Int[Array, " n"],
    h: Float[Array, "n q"],
    j: Float[Array, "n n q q"],
    w: Float[Array, "n n"],
    mask: Float[Array, " n"] | None = None,
  ) -> Array:
    """Pure function: log-probability of sequence under Potts energy.

    Computes:
        log P(seq) = h_seq + 0.5 * sum_ij W_ij * J_seq_seq - partition_normalization

    This is a pure JAX function with no model state dependency.
    It can be used as a scoring function in sampling algorithms.

    Args:
        seq: Sequence as amino acid indices (N,) in [0, q)
        h: Unary potentials (N, q) with x2 scale factor from PottsMPNN
        j: Pairwise potentials (N, N, q, q) with x2 scale factor
        w: Graph adjacency (N, N) binary or weighted edge indicator
        mask: Optional residue mask (N,)

    Returns:
        Scalar log-probability (unnormalized).
    """
    n = seq.shape[0]
    q = h.shape[-1]

    # Unary contribution
    h_seq = jnp.take_along_axis(h, seq[..., None], axis=-1).squeeze(-1)
    unary_energy = jnp.sum(h_seq)

    # Pairwise contribution with x2 scale factor
    seq_i = seq[..., None, None]  # (n, 1, 1)
    seq_j = seq[None, ..., None]  # (1, n, 1)
    j_seq = jnp.take_along_axis(
      j.reshape(n, n, q * q),
      (seq_i * q + seq_j).reshape(n, n, 1),
      axis=-1,
    ).reshape(n, n)
    pairwise_energy = 0.5 * jnp.sum(w * j_seq)

    # Total energy (log unnormalized probability)
    log_prob_val = unary_energy + pairwise_energy

    # Apply mask if provided
    if mask is not None:
      log_prob_val = jnp.where(jnp.all(mask > 0), log_prob_val, -jnp.inf)

    return log_prob_val
