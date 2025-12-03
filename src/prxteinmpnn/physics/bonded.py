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
    cos_theta = jnp.clip(cos_theta, -0.999999, 0.999999)
    theta = jnp.arccos(cos_theta)

    # Harmonic potential: E = 0.5 * k * (theta - theta0)^2
    return 0.5 * jnp.sum(k * (theta - theta0) ** 2)

  return angle_energy


def make_dihedral_energy_fn(
  displacement_fn: space.DisplacementFn,
  dihedral_indices: Array,
  dihedral_params: Array,
) -> Callable[[Array], Array]:
  """Creates a function to compute dihedral (torsion) energy.

  E = k * (1 + cos(n * phi - gamma))

  Args:
      displacement_fn: JAX MD displacement function.
      dihedral_indices: (N_dihedrals, 4) array of atom indices (i, j, k, l).
      dihedral_params: (N_dihedrals, 3) array of (periodicity, phase, k).

  Returns:
      A function energy(R) -> float.

  """
  periodicity = dihedral_params[:, 0]
  phase = dihedral_params[:, 1]
  k = dihedral_params[:, 2]

  def dihedral_energy(r: Array, **kwargs) -> Array:  # noqa: ARG001
    # Extract positions
    r_i = r[dihedral_indices[:, 0]]
    r_j = r[dihedral_indices[:, 1]]
    r_k = r[dihedral_indices[:, 2]]
    r_l = r[dihedral_indices[:, 3]]

    # Vectors
    # b0: i -> j
    # b1: j -> k
    # b2: k -> l
    # b0: i -> j (vector from j to i for correct angle definition? No, usually i->j)
    # Wait, analysis showed we need r_i - r_j to match IUPAC with current angle logic.
    # displacement_fn(a, b) = a - b.
    # We want r_i - r_j. So displacement_fn(r_i, r_j).
    b0 = jax.vmap(displacement_fn)(r_i, r_j)
    b1 = jax.vmap(displacement_fn)(r_k, r_j)
    b2 = jax.vmap(displacement_fn)(r_l, r_k)



    # Normalize b1
    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
    b1_unit = b1 / b1_norm

    # Projections onto plane perpendicular to b1
    # v = b0 - (b0 . b1_unit) * b1_unit
    # w = b2 - (b2 . b1_unit) * b1_unit
    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    # Angle calculation
    # x = v . w
    # y = (b1_unit x v) . w
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

    phi = jnp.arctan2(y, x)  # in radians

    # Energy
    # E = k * (1 + cos(n * phi - gamma))
    return jnp.sum(k * (1.0 + jnp.cos(periodicity * phi - phase)))

  return dihedral_energy
