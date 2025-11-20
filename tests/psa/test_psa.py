"""Test suite for the psa module."""
import jax.numpy as jnp
import pytest

from prxteinmpnn.psa.psa import (
    calculate_deformation_gradient,
    calculate_green_lagrange_strain,
    calculate_principal_strains,
    run_psa,
)
from prxteinmpnn.utils.data_structures import Protein


@pytest.fixture
def reference_coordinates():
    return jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

@pytest.fixture
def deformed_coordinates():
    return jnp.array([[1.1, 0.1, 0.0], [0.1, 1.1, 0.0], [0.0, 0.0, 1.0]])

def test_calculate_deformation_gradient(reference_coordinates, deformed_coordinates):
    deformation_gradient = calculate_deformation_gradient(
        reference_coordinates, deformed_coordinates,
    )
    assert deformation_gradient.shape == (3, 3)

def test_calculate_green_lagrange_strain(reference_coordinates, deformed_coordinates):
    deformation_gradient = calculate_deformation_gradient(
        reference_coordinates, deformed_coordinates,
    )
    green_lagrange_strain = calculate_green_lagrange_strain(deformation_gradient)
    assert green_lagrange_strain.shape == (3, 3)

def test_calculate_principal_strains(reference_coordinates, deformed_coordinates):
    deformation_gradient = calculate_deformation_gradient(
        reference_coordinates, deformed_coordinates,
    )
    green_lagrange_strain = calculate_green_lagrange_strain(deformation_gradient)
    principal_strains = calculate_principal_strains(green_lagrange_strain)
    assert principal_strains.shape == (3,)

def test_run_psa(protein_structure):
    # Use only alpha-carbon coordinates for simplicity
    ref_coords = protein_structure.coordinates[:, 1, :]
    deformed_coords = ref_coords * 1.1

    ref_protein = Protein(
        coordinates=ref_coords,
        aatype=protein_structure.aatype,
        one_hot_sequence=jnp.eye(21)[protein_structure.aatype],
        mask=protein_structure.mask,
        residue_index=protein_structure.residue_index,
        chain_index=protein_structure.chain_index,
    )
    deformed_protein = Protein(
        coordinates=deformed_coords,
        aatype=protein_structure.aatype,
        one_hot_sequence=jnp.eye(21)[protein_structure.aatype],
        mask=protein_structure.mask,
        residue_index=protein_structure.residue_index,
        chain_index=protein_structure.chain_index,
    )

    results = run_psa(ref_protein, deformed_protein)
    assert "deformation_gradient" in results
    assert "green_lagrange_strain" in results
    assert "principal_strains" in results

    # Check that the deformation gradient is close to 1.1 * identity
    assert jnp.allclose(results["deformation_gradient"], jnp.eye(3) * 1.1, atol=1e-5)
