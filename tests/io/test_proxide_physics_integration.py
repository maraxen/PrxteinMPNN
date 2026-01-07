"""Integration test for Proxide physics features within PrxteinMPNN."""

import pytest
from pathlib import Path
import numpy as np

from prxteinmpnn.io.parsing import parse_structure


def test_physics_feature_parsing():
    """Test that physics features are parsed correctly via dispatch."""
    # Use 1crn.pdb from proxide tests (relative to repo root)
    pdb_path = Path("proxide/tests/data/1crn.pdb").absolute()
    
    # Use bundled force field (relative to repo root)
    # Adjust path if needed, but assuming running from workspace root
    ff_path = Path("proxide/src/proxide/assets/amber/ff14SB.xml").absolute()
    
    if not pdb_path.exists():
        pytest.skip(f"Test data {pdb_path} not found")
    if not ff_path.exists():
        pytest.skip(f"Force field {ff_path} not found")

    print(f"Testing with PDB: {pdb_path}")
    print(f"Testing with FF: {ff_path}")

    # Request physics features
    # Note: parse_structure returns a generator
    protein = parse_structure(
        pdb_path,
        compute_vdw=True,
        compute_electrostatics=True,
        force_field=str(ff_path)
    )

    # Check VdW features
    if protein.vdw_features is None:
        pytest.fail("VdW features missing (None)")
    
    vdw = np.array(protein.vdw_features)
    print(f"VdW shape: {vdw.shape}")
    assert vdw.shape == (protein.coordinates.shape[0], 5), f"VdW shape mismatch: {vdw.shape}"
    # Check for non-zero values (should have some info)
    assert np.any(vdw != 0), "VdW features are all zero"

    # Check Electrostatics
    if protein.electrostatic_features is None:
        pytest.fail("Electrostatics features missing (None)")
        
    estat = np.array(protein.electrostatic_features)
    print(f"Estat shape: {estat.shape}")
    assert estat.shape == (protein.coordinates.shape[0], 5), f"Estat shape mismatch: {estat.shape}"
    assert np.any(estat != 0), "Electrostatic features are all zero"

    # Check physics parameters (charges, etc) are populated
    assert protein.charges is not None
    assert protein.sigmas is not None
    assert protein.epsilons is not None
