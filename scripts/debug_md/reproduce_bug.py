
import jax
import numpy as np
from prxteinmpnn.io.operations import pad_and_collate_proteins
from prxteinmpnn.utils.data_structures import ProteinTuple

def test_bug():
    # Create dummy protein
    N = 5
    protein = ProteinTuple(
        coordinates=np.zeros((N, 37, 3)),
        aatype=np.zeros(N, dtype=int),
        atom_mask=np.ones((N, 37)),
        residue_index=np.arange(N),
        chain_index=np.zeros(N),
        charges=np.zeros(N*37), # Dummy
        sigmas=np.zeros(N*37),
        epsilons=np.zeros(N*37),
        full_coordinates=np.zeros((N, 37, 3)),
    )
    
    print("Calling pad_and_collate_proteins with defaults (backbone_noise_mode='direct')...")
    # We expect _apply_md_parameterization NOT to be called (no force field loading).
    # Since we can't easily mock, we rely on the fact that if it runs, it might print "Loading force field..." 
    # or we can check if md_bonds are populated (if we didn't provide them).
    
    batch = pad_and_collate_proteins([protein], backbone_noise_mode="direct", use_electrostatics=False, use_vdw=False)
    
    if batch.md_bonds is not None:
        print("BUG: md_bonds populated! _apply_md_parameterization was called.")
    else:
        print("OK: md_bonds is None.")

if __name__ == "__main__":
    test_bug()
