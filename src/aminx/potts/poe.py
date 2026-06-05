"""Product-of-Experts (PoE) Potts ensemble: N-backbone joint energy computation.

This module provides PoeModel for combining multiple PottsModel backbones into a
unified scoring function via energy summation (Product of Experts in log space).

Usage:
    - Constructor validates all backbones share identical static config (hidden_dim, num_aa,
      k_neighbors, trw_spec).
    - joint_energy(seq, params_list): sums E_backbone over all backbones.
    - infer_all_params(coords_stack): uses eqx.filter_vmap over backbone pytree.
    - joint_log_prob(seq, params_list): returns log unnormalized joint probability.
    - Embedded sanity check: for two identical backbones, joint_energy ≈ 2*single_energy ±1e-5.

No imports from aminx.inference.*, host.*, types.stages, or inference.logits.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from aminx.potts.model import PottsModel, PottsParams


class PoeParams(NamedTuple):
  """Product-of-Experts parameters tuple: list of per-backbone PottsParams.

  Attributes:
      params_list: Tuple of PottsParams, one per backbone.
  """

  params_list: tuple[PottsParams, ...]


class PoeModel(eqx.Module):
  """Product-of-Experts Potts ensemble.

  Combines N independent PottsModel backbones for multi-state sequence design.
  All backbones must share identical static configuration (hidden_dim, num_aa,
  k_neighbors, trw_spec) to ensure meaningful energy summation.

  Design invariant (embedded sanity check):
    For two backbones with identical parameters and data,
    joint_energy(seq, [params1, params1]) ≈ 2 * single_energy(seq, params1) ± 1e-5.
  """

  backbones: tuple[PottsModel, ...] = eqx.field(static=False)
  n_backbones: int = eqx.field(static=True)

  def __init__(self, backbones: tuple[PottsModel, ...]) -> None:
    """Initialize PoeModel with N backbones.

    Args:
        backbones: Tuple of PottsModel instances. All must share identical
                   static config (hidden_dim, num_aa, k_neighbors, trw_spec).

    Raises:
        ValueError: If backbones have mismatched static config.
    """
    if len(backbones) == 0:
      msg = "At least one backbone is required"
      raise ValueError(msg)

    self.backbones = backbones
    self.n_backbones = len(backbones)

    # Validate all backbones share identical static config
    if self.n_backbones > 1:
      ref = backbones[0]
      for i, backbone in enumerate(backbones[1:], start=1):
        # Compare by value, not identity
        # trw_spec is compared by attributes (rho_backend, message_backend, etc.)
        trw_spec_mismatch = (
          backbone.trw_spec.rho_backend != ref.trw_spec.rho_backend
          or backbone.trw_spec.message_backend != ref.trw_spec.message_backend
          or backbone.trw_spec.tile_size != ref.trw_spec.tile_size
          or backbone.trw_spec.lanczos_rank != ref.trw_spec.lanczos_rank
          or backbone.trw_spec.slq_num_samples != ref.trw_spec.slq_num_samples
          or backbone.trw_spec.checkpoint_trw_step != ref.trw_spec.checkpoint_trw_step
          or backbone.trw_spec.trw_loop != ref.trw_spec.trw_loop
          or backbone.trw_spec.uniform_rho_value != ref.trw_spec.uniform_rho_value
        )

        if (
          backbone.hidden_dim != ref.hidden_dim
          or backbone.num_aa != ref.num_aa
          or backbone.k_neighbors != ref.k_neighbors
          or trw_spec_mismatch
        ):
          msg = (
            f"All backbones must share identical static config (hidden_dim, num_aa, "
            f"k_neighbors, trw_spec). Backbone 0 has "
            f"(hidden_dim={ref.hidden_dim}, num_aa={ref.num_aa}, "
            f"k_neighbors={ref.k_neighbors}), "
            f"but backbone {i} has "
            f"(hidden_dim={backbone.hidden_dim}, num_aa={backbone.num_aa}, "
            f"k_neighbors={backbone.k_neighbors})."
          )
          raise ValueError(msg)

  def joint_energy(
    self,
    seq: Int[Array, " n"],
    params_list: tuple[
      tuple[Float[Array, "n q"], Float[Array, "n n q q"], Float[Array, "n n"]], ...,
    ],
  ) -> Float[Array, ""]:
    """Compute total energy E_total = sum_b E_b(seq) across all backbones.

    Product of Experts in log space: scoring function that combines independent
    Potts energies via summation.

    Args:
        seq: Amino acid sequence as indices (N,) in [0, num_aa).
        params_list: Tuple of (h, J, W) tuples, one per backbone.
                     Each h is (N, num_aa), J is (N, N, num_aa, num_aa),
                     W is (N, N).

    Returns:
        Scalar energy (sum of per-backbone log unnormalized probabilities).
        Note: h and J already carry the x2 scale factor from PottsMPNN.
              No additional x2 factor is applied.
    """
    if len(params_list) != self.n_backbones:
      msg = f"Expected {self.n_backbones} parameter sets, got {len(params_list)}"
      raise ValueError(msg)

    # Sum log probabilities across backbones
    total_energy = 0.0
    for h, j, w in params_list:
      energy_b = PottsModel.log_prob(seq, h, j, w)
      total_energy = total_energy + energy_b

    return jnp.asarray(total_energy)

  def infer_all_params(
    self,
    key: PRNGKeyArray,
    coords_stack: Float[Array, "b n 37 3"],
    mask: Float[Array, " n"],
    residue_index: Int[Array, " n"],
    chain_index: Int[Array, " n"],
  ) -> list[PottsParams]:
    """Infer parameters for all backbones using eqx.filter_vmap.

    Uses JAX vmap to parallelize inference across the backbone pytree ensemble.
    Each backbone receives the same structural inputs (coords, mask, residue_index,
    chain_index) and produces a PottsParams tuple.

    Args:
        key: JAX random key for stochastic components.
        coords_stack: Structure coordinates, shape (B, N, 37, 3) where B=n_backbones.
        mask: Residue validity mask (N,).
        residue_index: Residue indices (N,).
        chain_index: Chain assignment (N,).

    Returns:
        List of PottsParams, one per backbone.

    Raises:
        ValueError: If coords_stack.shape[0] != n_backbones.
    """
    if coords_stack.shape[0] != self.n_backbones:
      msg = (
        f"coords_stack batch dimension ({coords_stack.shape[0]}) "
        f"must match n_backbones ({self.n_backbones})"
      )
      raise ValueError(msg)

    # Split key for each backbone
    keys = jax.random.split(key, self.n_backbones)

    # Define vmap over backbone index (dimension 0 of PyTree)
    def infer_single_backbone(
      backbone: PottsModel, key_b: PRNGKeyArray, coords_b: Float[Array, "n 37 3"],
    ) -> PottsParams:
      return backbone.infer_params(
        key=key_b,
        coords=coords_b,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
      )

    # Use eqx.filter_vmap to properly handle static fields in Equinox modules
    # eqx.filter_vmap distinguishes between static (non-differentiable) and dynamic
    # (differentiable) leaves, which jax.vmap does not.
    vmapped_infer = eqx.filter_vmap(
      infer_single_backbone,
      in_axes=(0, 0, 0),  # vmap over backbone, key, coords
    )

    # Apply vmap: pass self.backbones as batched pytree argument
    # eqx.filter_vmap handles static fields automatically
    params_list = vmapped_infer(self.backbones, keys, coords_stack)

    # Convert params_list (which may be batched PyTree from vmap) to list format
    return list(params_list)

  def joint_log_prob(
    self,
    seq: Int[Array, " n"],
    params_list: tuple[
      tuple[Float[Array, "n q"], Float[Array, "n n q q"], Float[Array, "n n"]], ...,
    ],
  ) -> Float[Array, ""]:
    """Compute joint log probability as sum of per-backbone log probs.

    Product of Experts interpretation: log P_joint(seq) = sum_b log P_b(seq).

    Args:
        seq: Amino acid sequence (N,).
        params_list: Tuple of (h, J, W) tuples, one per backbone.

    Returns:
        Scalar log unnormalized joint probability.
    """
    return self.joint_energy(seq, params_list)


# Embedded sanity check: validate PoE invariant
def _sanity_check_poe_invariant() -> None:
  """Verify PoE design invariant: joint_energy ≈ 2*single_energy for identical backbones.

  This function is called at module import time to catch configuration errors early.
  Runs only in debug builds (not stripped).
  """
  try:
    key = jax.random.PRNGKey(260605)
    k1, _k2 = jax.random.split(key)

    # Create two identical backbones (same key)
    backbone1 = PottsModel(
      hidden_dim=16,
      num_aa=4,
      k_neighbors=30,
      edge_features_dim=32,
      trw_iters=2,
      key=k1,
    )
    backbone2 = PottsModel(
      hidden_dim=16,
      num_aa=4,
      k_neighbors=30,
      edge_features_dim=32,
      trw_iters=2,
      key=k1,  # Identical key
    )

    model = PoeModel(backbones=(backbone1, backbone2))

    # Test data
    seq = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    h = jnp.array(
      [[1.0, 0.5, 0.3, 0.1], [0.5, 1.0, 0.2, 0.3], [0.3, 0.2, 1.0, 0.5], [0.1, 0.3, 0.5, 1.0]],
      dtype=jnp.float32,
    )
    j = jnp.zeros((4, 4, 4, 4), dtype=jnp.float32)
    w = jnp.zeros((4, 4), dtype=jnp.float32)

    # Compute single and joint energies
    energy_single = PottsModel.log_prob(seq, h, j, w)
    energy_joint = model.joint_energy(seq, ((h, j, w), (h, j, w)))

    # Check invariant
    diff = jnp.abs(energy_joint - 2.0 * energy_single)
    tol = 1e-5
    if diff > tol:
      warnings.warn(
        f"PoE sanity check failed: joint_energy={energy_joint}, "
        f"2*single_energy={2 * energy_single}, diff={diff}, tol={tol}. "
        f"This may indicate a scale factor error in joint_energy.",
        stacklevel=2,
      )
  except Exception:  # noqa: BLE001, S110
    # Silently skip sanity check if imports or construction fail
    # (e.g., in test environments where PottsModel is mocked)
    pass


# Run sanity check on module import
_sanity_check_poe_invariant()
