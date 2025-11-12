"""Debug script to understand force sign convention."""

import jax.numpy as jnp

from src.prxteinmpnn.physics.electrostatics import (
  compute_coulomb_forces,
  compute_pairwise_displacements,
)

# Two charges: +1 at origin (0,0,0), -1 at (5,0,0)
positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
charges = jnp.array([1.0, -1.0])

print("=== Displacement Analysis ===")
print("Positions:")
print(positions)

displacements, distances = compute_pairwise_displacements(positions, positions)
print("\nDisplacements[i,j] (should be pos[j] - pos[i]):")
print(displacements)
print("\nDisplacement[0,1] (from pos0 to pos1):", displacements[0, 1])
print("Displacement[1,0] (from pos1 to pos0):", displacements[1, 0])

print("\n=== Force Analysis ===")
print("Charges:", charges)
forces = compute_coulomb_forces(displacements, distances, charges, charges, exclude_self=True)
print("\nForces shape:", forces.shape)
print("Force on charge 0:", forces[0])
print("Force on charge 1:", forces[1])

print("\n=== Expected Behavior ===")
print("Charge 0 (+1) at origin, Charge 1 (-1) at (5,0,0)")
print("Opposite charges attract → Force on 0 should point toward 1 (+x direction)")
print("Actual force[0,0]:", forces[0, 0])
print("Expected: > 0 (positive x)")
print("MATCH:", forces[0, 0] > 0)
