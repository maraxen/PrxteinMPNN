"""Tests for force field loading and saving."""

import jax.numpy as jnp
import pytest

from prxteinmpnn.physics.force_fields import (
    FullForceField,
    load_force_field,
    save_force_field,
)


def test_force_field_creation():
    """Test creating a FullForceField object."""
    ff = FullForceField(
        charges_by_id=jnp.array([0.0, -0.5, 0.5]),
        sigmas_by_id=jnp.array([3.5, 3.0, 3.2]),
        epsilons_by_id=jnp.array([0.1, 0.15, 0.12]),
        atom_key_to_id={("ALA", "N"): 0, ("ALA", "CA"): 1, ("ALA", "C"): 2},
        id_to_atom_key=[("ALA", "N"), ("ALA", "CA"), ("ALA", "C")],
        atom_class_map={"ALA_N": "N", "ALA_CA": "CT1", "ALA_C": "C"},
        bonds=[],
        angles=[],
        propers=[],
        impropers=[],
        source_files=["test.xml"],
    )
    
    assert len(ff.charges_by_id) == 3
    assert len(ff.id_to_atom_key) == 3


def test_force_field_get_charge():
    """Test getting charge for specific atom."""
    ff = FullForceField(
        charges_by_id=jnp.array([0.0, -0.5, 0.5]),
        sigmas_by_id=jnp.array([3.5, 3.0, 3.2]),
        epsilons_by_id=jnp.array([0.1, 0.15, 0.12]),
        atom_key_to_id={("ALA", "N"): 0, ("ALA", "CA"): 1, ("ALA", "C"): 2},
        id_to_atom_key=[("ALA", "N"), ("ALA", "CA"), ("ALA", "C")],
        atom_class_map={},
        bonds=[],
        angles=[],
        propers=[],
        impropers=[],
        source_files=[],
    )
    
    charge = ff.get_charge("ALA", "CA")
    assert charge == -0.5


def test_force_field_get_charge_unknown_atom():
    """Test that unknown atom returns zero charge."""
    ff = FullForceField(
        charges_by_id=jnp.array([0.0]),
        sigmas_by_id=jnp.array([3.5]),
        epsilons_by_id=jnp.array([0.1]),
        atom_key_to_id={("ALA", "N"): 0},
        id_to_atom_key=[("ALA", "N")],
        atom_class_map={},
        bonds=[],
        angles=[],
        propers=[],
        impropers=[],
        source_files=[],
    )
    
    charge = ff.get_charge("GLY", "CA")  # Not in force field
    assert charge == 0.0


def test_force_field_get_lj_params():
    """Test getting LJ parameters for specific atom."""
    ff = FullForceField(
        charges_by_id=jnp.array([0.0]),
        sigmas_by_id=jnp.array([3.5]),
        epsilons_by_id=jnp.array([0.1]),
        atom_key_to_id={("ALA", "CA"): 0},
        id_to_atom_key=[("ALA", "CA")],
        atom_class_map={},
        bonds=[],
        angles=[],
        propers=[],
        impropers=[],
        source_files=[],
    )
    
    sigma, epsilon = ff.get_lj_params("ALA", "CA")
    assert sigma == pytest.approx(3.5, rel=1e-5)
    assert epsilon == pytest.approx(0.1, rel=1e-5)


def test_force_field_save_and_load(temp_ff_dir):
    """Test saving and loading force field."""
    ff_original = FullForceField(
        charges_by_id=jnp.array([0.0, -0.5, 0.5]),
        sigmas_by_id=jnp.array([3.5, 3.0, 3.2]),
        epsilons_by_id=jnp.array([0.1, 0.15, 0.12]),
        atom_key_to_id={("ALA", "N"): 0, ("ALA", "CA"): 1, ("ALA", "C"): 2},
        id_to_atom_key=[("ALA", "N"), ("ALA", "CA"), ("ALA", "C")],
        atom_class_map={"ALA_N": "N", "ALA_CA": "CT1", "ALA_C": "C"},
        bonds=[("N", "CT1", 1.45, 300.0)],
        angles=[],
        propers=[],
        impropers=[],
        source_files=["test.xml"],
    )
    
    # Save
    filepath = temp_ff_dir / "test_ff.eqx"
    save_force_field(filepath, ff_original)
    
    # Load
    ff_loaded = load_force_field(filepath)
    
    # Check arrays match
    assert jnp.allclose(ff_loaded.charges_by_id, ff_original.charges_by_id)
    assert jnp.allclose(ff_loaded.sigmas_by_id, ff_original.sigmas_by_id)
    assert jnp.allclose(ff_loaded.epsilons_by_id, ff_original.epsilons_by_id)
    
    # Check static fields match
    assert ff_loaded.atom_key_to_id == ff_original.atom_key_to_id
    assert ff_loaded.id_to_atom_key == ff_original.id_to_atom_key
    # Note: bonds may be converted to lists due to JSON serialization
    assert len(ff_loaded.bonds) == len(ff_original.bonds)


def test_force_field_save_load_preserves_tuple_keys(temp_ff_dir):
    """Test that tuple keys are preserved through save/load."""
    ff = FullForceField(
        charges_by_id=jnp.array([0.0, 0.1]),
        sigmas_by_id=jnp.array([3.5, 3.2]),
        epsilons_by_id=jnp.array([0.1, 0.15]),
        atom_key_to_id={("ALA", "CA"): 0, ("GLY", "CA"): 1},
        id_to_atom_key=[("ALA", "CA"), ("GLY", "CA")],
        atom_class_map={},
        bonds=[],
        angles=[],
        propers=[],
        impropers=[],
        source_files=[],
    )
    
    filepath = temp_ff_dir / "test_keys.eqx"
    save_force_field(filepath, ff)
    ff_loaded = load_force_field(filepath)
    
    # Keys should still be tuples
    assert isinstance(list(ff_loaded.atom_key_to_id.keys())[0], tuple)
    assert ("ALA", "CA") in ff_loaded.atom_key_to_id
    # id_to_atom_key should also preserve tuples
    assert isinstance(ff_loaded.id_to_atom_key[0], tuple)