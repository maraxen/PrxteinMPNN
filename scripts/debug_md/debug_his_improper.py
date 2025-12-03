import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from termcolor import colored

# PrxteinMPNN Imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.physics import bonded, jax_md_bridge, force_fields
from prxteinmpnn.io.parsing import biotite as parsing_biotite
from jax_md import space

# Configuration
PDB_PATH = "data/pdb/1UBQ.pdb"
FF_EQX_PATH = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"

def debug_his_improper():
    print(colored("===========================================================", "cyan"))
    print(colored("   Debugging HIS Improper Torsion", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Load Structure
    print(colored("\n[1] Loading Structure...", "yellow"))
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    positions = jnp.array(atom_array.coord)
    
    # 2. Find Atoms
    # Missing in JAX: HIS:CE1(1086)-HIS:CD2(1085)-HIS:NE2(1087)-HIS:HE2(1094)
    # Extra in JAX: HIS:CD2(1085)-HIS:CE1(1086)-HIS:NE2(1087)-HIS:HE2(1094)
    
    # Indices (0-based from validation output, which matches atom_array)
    # Validation output indices are 0-based?
    # "HIS:CE1(1086)" -> Index 1086.
    
    idx_ce1 = 1086
    idx_cd2 = 1085
    idx_ne2 = 1087
    idx_he2 = 1094
    
    print(f"Indices: CE1={idx_ce1}, CD2={idx_cd2}, NE2={idx_ne2}, HE2={idx_he2}")
    
    # Verify atom names
    print(f"Atom {idx_ce1}: {atom_array.atom_name[idx_ce1]}")
    print(f"Atom {idx_cd2}: {atom_array.atom_name[idx_cd2]}")
    print(f"Atom {idx_ne2}: {atom_array.atom_name[idx_ne2]}")
    print(f"Atom {idx_he2}: {atom_array.atom_name[idx_he2]}")
    
    # 3. Define Torsions
    # OpenMM (Missing in JAX): CE1-CD2-NE2-HE2
    t_omm = [idx_ce1, idx_cd2, idx_ne2, idx_he2]
    
    # JAX (Extra): CD2-CE1-NE2-HE2
    t_jax = [idx_cd2, idx_ce1, idx_ne2, idx_he2]
    
    # 4. Calculate Angles
    displacement_fn, shift_fn = space.free()
    
    def compute_angle(indices):
        r = positions
        r_i = r[indices[0]]
        r_j = r[indices[1]]
        r_k = r[indices[2]]
        r_l = r[indices[3]]
        
        # Bonded.py logic (Original/Correct)
        b0 = displacement_fn(r_i, r_j) # j->i
        b1 = displacement_fn(r_k, r_j) # j->k
        b2 = displacement_fn(r_l, r_k) # k->l
        
        b1_norm = jnp.linalg.norm(b1) + 1e-8
        b1_unit = b1 / b1_norm
        
        v = b0 - jnp.sum(b0 * b1_unit) * b1_unit
        w = b2 - jnp.sum(b2 * b1_unit) * b1_unit
        
        x = jnp.sum(v * w)
        y = jnp.sum(jnp.cross(b1_unit, v) * w)
        
        phi = jnp.arctan2(y, x)
        return phi
        
    phi_omm = compute_angle(t_omm)
    phi_jax = compute_angle(t_jax)
    
    print(f"\nPhi OpenMM Definition ({t_omm}): {phi_omm:.4f} rad ({np.degrees(phi_omm):.2f} deg)")
    print(f"Phi JAX Definition ({t_jax}): {phi_jax:.4f} rad ({np.degrees(phi_jax):.2f} deg)")
    
    # 5. Get Parameters
    # We need to find the parameters for this improper.
    # We can load the force field and look it up?
    # Or just use what we know from general Amber.
    # Usually improper k is ~1.1 or 10.5 kcal/mol?
    # Phase is usually 180 (pi). Periodicity 2.
    
    # Let's try to find it in the loaded force field.
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    # We need to simulate the matching logic to see what params are assigned.
    # But we know JAX assigned *something*.
    # Let's assume standard improper params.
    # Or better, let's look at what JAX assigned in the full system.
    
    # Parameterize system
    residues = []
    atom_names = []
    atom_counts = []
    # ... (Need to reconstruct lists) ...
    # Simplified: Just run parameterize_system
    # This is slow, but accurate.
    
    # Or we can just grep the params from the validation output if available?
    # No.
    
    # Let's just print the difference in angles.
    diff = abs(phi_omm - phi_jax)
    print(f"Angle Difference: {diff:.4f} rad ({np.degrees(diff):.2f} deg)")
    
    # If angle difference is large, energy difference could be large.
    # Improper potential: E = k * (1 + cos(n*phi - phase))
    # Usually n=2, phase=180.
    # E = k * (1 + cos(2*phi - 180)) = k * (1 - cos(2*phi)).
    # Min at phi=0 or 180.
    # If phi_omm is near 180 (planar), E ~ 0.
    # If phi_jax is different?
    
    # Let's assume k=1.1 kcal/mol (common for Amber impropers).
    k = 1.1
    n = 2
    phase = np.pi
    
    e_omm = k * (1 + np.cos(n * phi_omm - phase))
    e_jax = k * (1 + np.cos(n * phi_jax - phase))
    
    print(f"\nEstimated Energy (k={k}, n={n}, phase={phase:.2f}):")
    print(f"  OpenMM E: {e_omm:.4f} kcal/mol")
    print(f"  JAX E:    {e_jax:.4f} kcal/mol")
    print(f"  Diff:     {e_jax - e_omm:.4f} kcal/mol")
    
    # Try n=3 (Periodicity 3 is also common for impropers? No, usually 2 for planar).
    
if __name__ == "__main__":
    debug_his_improper()
