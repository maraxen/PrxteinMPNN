"""Bonded potential factories for JAX MD integration."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax_md import energy, space, util

Array = util.Array


def make_bond_energy_fn(
  displacement_fn: space.DisplacementFn,
  bond_indices: Array,
  bond_params: Array,
) -> Callable[[Array], Array]:
  """Creates a function to compute bond energy.

  Args:
      displacement_fn: JAX MD displacement function.
      bond_indices: (N_bonds, 2) array of atom indices involved in bonds.
      bond_params: (N_bonds, 2) array of (equilibrium_length, spring_constant).
                   Note: jax_md.energy.simple_spring_bond takes length then epsilon/k.

  Returns:
      A function energy(R) -> float.

  """
  length = bond_params[:, 0]
  k = bond_params[:, 1]

  return energy.simple_spring_bond(
    displacement_fn,
    bond_indices,
    length=length,
    epsilon=k,
  )


def make_angle_energy_fn(
  displacement_fn: space.DisplacementFn,
  angle_indices: Array,
  angle_params: Array,
) -> Callable[[Array], Array]:
  """Creates a function to compute angle energy using a harmonic approximation.

  Args:
      displacement_fn: JAX MD displacement function.
      angle_indices: (N_angles, 3) array of atom indices (i, j, k) where j is central.
      angle_params: (N_angles, 2) array of (equilibrium_angle_rad, spring_constant).

  Returns:
      A function energy(R) -> float.

  """
  theta0 = angle_params[:, 0]
  k = angle_params[:, 1]

  def angle_energy(r: Array, **kwargs) -> Array:  # noqa: ARG001
    # Extract positions
    r_i = r[angle_indices[:, 0]]
    r_j = r[angle_indices[:, 1]]
    r_k = r[angle_indices[:, 2]]

    # Vectors
    # vmap displacement_fn over the batch of angles
    v_ji = jax.vmap(displacement_fn)(r_i, r_j)
    v_jk = jax.vmap(displacement_fn)(r_k, r_j)

    # Distances
    d_ji = space.distance(v_ji)
    d_jk = space.distance(v_jk)

    # Cosine of angle
    # Clip to prevent NaN in arccos
    denom = d_ji * d_jk + 1e-8
    cos_theta = jnp.sum(v_ji * v_jk, axis=-1) / denom
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    theta = jnp.arccos(cos_theta)

    # Harmonic potential: E = 0.5 * k * (theta - theta0)^2
    return 0.5 * jnp.sum(k * (theta - theta0) ** 2)

  return angle_energy
