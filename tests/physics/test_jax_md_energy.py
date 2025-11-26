"""Tests for JAX MD energy function."""

import jax
import jax.numpy as jnp
import pytest
from jax_md import space

from prxteinmpnn.physics import jax_md_bridge, system
from prxteinmpnn.utils import residue_constants


@pytest.fixture(autouse=True)
def mock_stereo_chemical_props(monkeypatch):
  """Mock stereo chemical props loading."""
  def mock_load():
    from prxteinmpnn.utils.residue_constants import Bond, BondAngle
    bonds = {
        "ALA": [
            Bond("N", "CA", 1.46, 0.01),
            Bond("CA", "C", 1.52, 0.01),
            Bond("C", "O", 1.23, 0.01),
            Bond("CA", "CB", 1.53, 0.01)
        ],
        "GLY": [
            Bond("N", "CA", 1.45, 0.01),
            Bond("CA", "C", 1.52, 0.01),
            Bond("C", "O", 1.23, 0.01)
        ]
    }
    angles = {
        "ALA": [],
        "GLY": []
    }
    return bonds, {}, angles

  monkeypatch.setattr(residue_constants, "load_stereo_chemical_props", mock_load)
  yield


@pytest.fixture
def mock_force_field():
  class MockFF:
    def get_charge(self, res, atom):
      return 0.1

    def get_lj_params(self, res, atom):
      return 3.0, 0.1
  return MockFF()


def test_energy_function_finite(mock_force_field):
  """Test that energy function returns finite values and forces."""
  residues = ["ALA", "GLY"]
  atoms_ala = residue_constants.residue_atoms["ALA"]
  atoms_gly = residue_constants.residue_atoms["GLY"]
  atom_names = atoms_ala + atoms_gly
  
  params = jax_md_bridge.parameterize_system(
      mock_force_field, residues, atom_names
  )
  
  # Create random positions
  key = jax.random.PRNGKey(0)
  n_atoms = len(atom_names)
  r = jax.random.normal(key, (n_atoms, 3)) * 10.0
  
  # Create energy function
  displacement_fn, _ = space.free()
  energy_fn = system.make_energy_fn(displacement_fn, params)
  
  # Compute energy
  e = energy_fn(r)
  assert jnp.isfinite(e)
  
  # Compute forces
  grad_fn = jax.grad(energy_fn)
  forces = -grad_fn(r)
  assert jnp.all(jnp.isfinite(forces))


def test_exclusion_mask_prevents_explosion(mock_force_field):
  """Test that bonded atoms don't have massive VDW repulsion."""
  # Create 2 atoms bonded
  residues = ["ALA"]
  # Just use N and CA
  atom_names = ["N", "CA"]
  
  # We need to hack parameterize_system or use a subset
  # parameterize_system expects full residue atoms usually.
  # Let's use full ALA but focus on N-CA
  atom_names = residue_constants.residue_atoms["ALA"]
  
  params = jax_md_bridge.parameterize_system(
      mock_force_field, residues, atom_names
  )
  
  displacement_fn, _ = space.free()
  energy_fn = system.make_energy_fn(displacement_fn, params)
  
  # Place N and CA at 1.46 A (bond length)
  # Sigma is 3.0 (from mock). 1.46 < 3.0 -> massive repulsion if not excluded.
  
  # Construct positions
  r = jnp.zeros((len(atom_names), 3))
  # N at 0,0,0. CA at 1.46,0,0
  # Indices: N=3, CA=1, C=0, O=4, CB=2
  
  r = r.at[3].set(jnp.array([0.0, 0.0, 0.0]))      # N
  r = r.at[1].set(jnp.array([1.46, 0.0, 0.0]))     # CA
  r = r.at[0].set(jnp.array([2.98, 0.0, 0.0]))     # C (1.46+1.52)
  r = r.at[4].set(jnp.array([4.21, 0.0, 0.0]))     # O (2.98+1.23)
  r = r.at[2].set(jnp.array([1.46, 1.53, 0.0]))    # CB (perp to CA)

  
  # Calculate energy
  e = energy_fn(r)
  
  # If VDW was active, E ~ (3/1.46)^12 ~ 2^12 ~ 4000 epsilon.
  # If excluded, E ~ bond energy ~ 0 (at equilibrium).
  # Plus Coulomb (0.1*0.1 / 1.46 * 332) ~ 2 kcal/mol.
  
  assert e < 100.0 # Should be small
