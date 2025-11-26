"""Tests for bonded potentials."""

import jax
import jax.numpy as jnp
import pytest
from jax_md import space

from prxteinmpnn.physics import bonded


def test_bond_energy():
  """Test simple bond energy."""
  displacement_fn, _ = space.free()

  # 2 atoms, 1 bond
  # Bond length 1.0, k=100
  # Positions: 0.0 and 1.1 (strain 0.1)
  # E = 0.5 * k * (r - r0)^2 = 0.5 * 100 * (0.1)^2 = 0.5 * 100 * 0.01 = 0.5

  r = jnp.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
  bond_indices = jnp.array([[0, 1]])
  bond_params = jnp.array([[1.0, 100.0]])

  energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_indices, bond_params)

  e = energy_fn(r)
  assert jnp.isclose(e, 0.5)

  # Check forces
  # F = -dE/dr
  # Force on atom 1 should be pulling back (-x direction)
  grad_fn = jax.grad(energy_fn)
  forces = -grad_fn(r)

  # F_spring = -k * x = -100 * 0.1 = -10
  # Atom 1 is at 1.1, should be pulled left (-10)
  # Atom 0 is at 0.0, should be pulled right (+10)
  assert jnp.isclose(forces[1, 0], -10.0, atol=1e-5)
  assert jnp.isclose(forces[0, 0], 10.0, atol=1e-5)


def test_angle_energy():
  """Test harmonic angle energy."""
  displacement_fn, _ = space.free()

  # 3 atoms: (1,0,0), (0,0,0), (0,1,0) -> 90 degrees (pi/2)
  # Equilibrium: 180 degrees (pi)
  # k = 100
  # Delta = pi/2
  # E = 0.5 * 100 * (pi/2)^2

  r = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
  angle_indices = jnp.array([[0, 1, 2]])  # 1 is central
  angle_params = jnp.array([[jnp.pi, 100.0]])

  energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_indices, angle_params)

  e = energy_fn(r)
  expected = 0.5 * 100.0 * (jnp.pi / 2.0) ** 2
  assert jnp.isclose(e, expected, rtol=1e-4)
