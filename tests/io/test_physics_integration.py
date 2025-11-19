"""Test physics parameter integration and feature computation."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from prxteinmpnn.io.parsing.biotite import load_structure_with_hydride
from prxteinmpnn.io.parsing.physics_utils import populate_physics_parameters
from prxteinmpnn.io.parsing.dispatch import parse_input
from prxteinmpnn.physics.features import compute_electrostatic_node_features


def test_force_field_parameter_population():
    """Test that physics parameters are populated from force field."""
    pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      12.000  11.500  11.000  1.00  0.00           O
ATOM      5  CB  ALA A   1      11.500   9.000  10.000  1.00  0.00           C
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)
    
    try:
        atom_array = load_structure_with_hydride(
            pdb_file,
            add_hydrogens=False,
            remove_solvent=True
        )
        
        # Populate parameters
        charges, sigmas, epsilons = populate_physics_parameters(atom_array)
        
        # Verify shapes
        assert charges.shape[0] == atom_array.array_length()
        assert sigmas.shape[0] == atom_array.array_length()
        assert epsilons.shape[0] == atom_array.array_length()
        
        # Verify non-zero values (at least some should be populated)
        assert np.any(sigmas > 0), "Sigmas should have non-zero values"
        assert np.any(epsilons > 0), "Epsilons should have non-zero values"
        
        # Verify element-based defaults are reasonable
        # Carbon should have sigma ~1.9, epsilon ~0.086
        carbon_indices = np.where(atom_array.element == 'C')[0]
        if len(carbon_indices) > 0:
            assert np.all(sigmas[carbon_indices] > 1.5), "Carbon sigma should be > 1.5 Å"
            assert np.all(sigmas[carbon_indices] < 2.5), "Carbon sigma should be < 2.5 Å"
        
        print("✓ Force field parameter population test passed")
        
    finally:
        pdb_file.unlink()


def test_physics_parameter_propagation():
    """Test that physics parameters propagate through the pipeline."""
    pqr_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000 -0.4157 1.8500
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  0.0337 2.2750
ATOM      3  C   ALA A   1      11.500  11.000  10.000  0.5973 2.0000
ATOM      4  O   ALA A   1      12.000  11.500  11.000 -0.5679 1.7000
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pqr', delete=False) as f:
        f.write(pqr_content)
        pqr_file = Path(f.name)
    
    try:
        # Parse through dispatch
        protein_tuples = list(parse_input(pqr_file, chain_id=None))
        
        assert len(protein_tuples) > 0, "Should yield at least one ProteinTuple"
        
        protein = protein_tuples[0]
        
        # Verify physics parameters are present
        assert protein.charges is not None, "Charges should be populated"
        assert protein.radii is not None, "Radii should be populated"
        assert protein.epsilons is not None, "Epsilons should be populated"
        
        # Verify charges match PQR values (approximately)
        expected_charges = np.array([-0.4157, 0.0337, 0.5973, -0.5679])
        assert len(protein.charges) >= 4, f"Expected at least 4 charges, got {len(protein.charges)}"
        
        # Check first few charges (allowing for some tolerance)
        for i, expected in enumerate(expected_charges):
            if i < len(protein.charges):
                assert abs(protein.charges[i] - expected) < 0.01, \
                    f"Charge {i}: expected {expected}, got {protein.charges[i]}"
        
        print("✓ Physics parameter propagation test passed")
        
    finally:
        pqr_file.unlink()


def test_electrostatic_feature_computation():
    """Test that electrostatic features can be computed from physics parameters."""
    pqr_content = """ATOM      1  N   GLY A   1       0.000   0.000   0.000 -0.4157 1.8500
ATOM      2  CA  GLY A   1       1.458   0.000   0.000  0.0337 2.2750
ATOM      3  C   GLY A   1       2.009   1.420   0.000  0.5973 2.0000
ATOM      4  O   GLY A   1       1.251   2.389   0.000 -0.5679 1.7000
ATOM      5  N   ALA A   2       3.338   1.584   0.000 -0.4157 1.8500
ATOM      6  CA  ALA A   2       4.000   2.880   0.000  0.0337 2.2750
ATOM      7  C   ALA A   2       5.520   2.720   0.000  0.5973 2.0000
ATOM      8  O   ALA A   2       6.100   1.635   0.000 -0.5679 1.7000
ATOM      9  CB  ALA A   2       3.500   3.700   1.200  0.0000 2.0600
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pqr', delete=False) as f:
        f.write(pqr_content)
        pqr_file = Path(f.name)
    
    try:
        protein_tuples = list(parse_input(pqr_file, chain_id=None))
        assert len(protein_tuples) > 0
        
        protein = protein_tuples[0]
        
        # Compute electrostatic features
        features = compute_electrostatic_node_features(protein)
        
        # Verify shape: (n_residues, 5)
        assert features.ndim == 2, f"Features should be 2D, got {features.ndim}D"
        assert features.shape[1] == 5, f"Features should have 5 components, got {features.shape[1]}"
        
        # Verify features are finite
        assert np.all(np.isfinite(features)), "All features should be finite"
        
        # Verify force magnitudes are reasonable (last column)
        force_magnitudes = features[:, -1]
        assert np.all(force_magnitudes >= 0), "Force magnitudes should be non-negative"
        
        print(f"✓ Electrostatic feature computation test passed")
        print(f"  Features shape: {features.shape}")
        print(f"  Force magnitude range: [{np.min(force_magnitudes):.2e}, {np.max(force_magnitudes):.2e}]")
        
    finally:
        pqr_file.unlink()


def test_fallback_to_amber_defaults():
    """Test that fallback to AMBER defaults works when force field unavailable."""
    pdb_content = """ATOM      1  N   UNK A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  UNK A   1      11.000  10.000  10.000  1.00  0.00           C
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)
    
    try:
        atom_array = load_structure_with_hydride(
            pdb_file,
            add_hydrogens=False,
            remove_solvent=True
        )
        
        # Populate with force field that doesn't exist (will fall back to defaults)
        charges, sigmas, epsilons = populate_physics_parameters(
            atom_array,
            force_field_name="nonexistent-ff"
        )
        
        # Should still get reasonable values from element-based defaults
        assert sigmas.shape[0] == atom_array.array_length()
        assert np.all(sigmas > 0), "Should have positive sigma values from defaults"
        assert np.all(epsilons > 0), "Should have positive epsilon values from defaults"
        
        print("✓ Fallback to AMBER defaults test passed")
        
    finally:
        pdb_file.unlink()


if __name__ == "__main__":
    test_force_field_parameter_population()
    test_physics_parameter_propagation()
    test_electrostatic_feature_computation()
    test_fallback_to_amber_defaults()
    print("\n✅ All physics integration tests passed!")
