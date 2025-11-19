"""Test solvent removal and force field parameter population."""

import tempfile
from pathlib import Path
import numpy as np

from prxteinmpnn.io.parsing.biotite import load_structure_with_hydride
from prxteinmpnn.io.parsing.physics_utils import populate_physics_parameters


def test_solvent_removal():
    """Test that solvent molecules are removed from structures."""
    # Create a minimal PDB with water
    pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      12.000  11.500  11.000  1.00  0.00           O
HETATM    5  O   HOH A 101      20.000  20.000  20.000  1.00  0.00           O
HETATM    6  O   HOH A 102      21.000  21.000  21.000  1.00  0.00           O
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)
    
    try:
        # Load with solvent removal
        atom_array_no_solvent = load_structure_with_hydride(
            pdb_file,
            add_hydrogens=False,
            remove_solvent=True
        )
        
        # Load without solvent removal
        atom_array_with_solvent = load_structure_with_hydride(
            pdb_file,
            add_hydrogens=False,
            remove_solvent=False
        )
        
        # Check that solvent was removed
        assert atom_array_with_solvent.array_length() == 6, "Should have 6 atoms with solvent"
        assert atom_array_no_solvent.array_length() == 4, "Should have 4 atoms without solvent"
        
        # Check that no water remains
        has_water = np.any(atom_array_no_solvent.res_name == "HOH")
        assert not has_water, "Water should be removed"
        
        print("✓ Solvent removal test passed")
        
    finally:
        pdb_file.unlink()


def test_force_field_parameter_population():
    """Test that physics parameters can be populated from force field."""
    # Create a minimal PDB
    pdb_content = """ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   ALA A   1      11.500  11.000  10.000  1.00  0.00           C
ATOM      4  O   ALA A   1      12.000  11.500  11.000  1.00  0.00           O
END
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        f.write(pdb_content)
        pdb_file = Path(f.name)
    
    try:
        atom_array = load_structure_with_hydride(
            pdb_file,
            add_hydrogens=False,
            remove_solvent=False
        )
        
        # Populate parameters (will use defaults if force field not available)
        charges, sigmas, epsilons = populate_physics_parameters(atom_array)
        
        # Check shapes
        assert charges.shape == (4,), f"Expected 4 charges, got {charges.shape}"
        assert sigmas.shape == (4,), f"Expected 4 sigmas, got {sigmas.shape}"
        assert epsilons.shape == (4,), f"Expected 4 epsilons, got {epsilons.shape}"
        
        # Check that parameters are non-zero (at least some)
        assert np.any(sigmas > 0), "Sigmas should have non-zero values"
        assert np.any(epsilons > 0), "Epsilons should have non-zero values"
        
        print("✓ Force field parameter population test passed")
        print(f"  Charges: {charges}")
        print(f"  Sigmas: {sigmas}")
        print(f"  Epsilons: {epsilons}")
        
    finally:
        pdb_file.unlink()


if __name__ == "__main__":
    test_solvent_removal()
    test_force_field_parameter_population()
    print("\n✅ All solvent and force field tests passed!")
