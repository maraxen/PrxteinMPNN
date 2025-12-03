import os
import sys
import numpy as np
import jax.numpy as jnp
import pandas as pd
from termcolor import colored

# OpenMM Imports
try:
    import openmm
    import openmm.app as app
    import openmm.unit as unit
except ImportError:
    print(colored("Error: OpenMM not found. Please install it.", "red"))
    sys.exit(1)

# PrxteinMPNN Imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
from prxteinmpnn.physics import force_fields, jax_md_bridge
from prxteinmpnn.io.parsing import biotite as parsing_biotite
import biotite.structure.io.pdb as pdb

# Configuration
PDB_PATH = "data/pdb/1UBQ.pdb"
FF_EQX_PATH = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"

def compare_torsions():
    print(colored("===========================================================", "cyan"))
    print(colored("   Comparing JAX MD vs OpenMM Torsions", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Load Structure
    print(colored("\n[1] Loading Structure...", "yellow"))
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    
    # 2. Setup OpenMM System
    print(colored("\n[2] Setting up OpenMM System...", "yellow"))
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file_bio = pdb.PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology

    # Create ForceField
    try:
        ff19sb_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
        if not os.path.exists(ff19sb_xml):
             omm_ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
        else:
             omm_ff = app.ForceField(ff19sb_xml, 'implicit/obc2.xml')
    except Exception as e:
        print(colored(f"Error loading ff19SB: {e}", "red"))
        sys.exit(1)

    omm_system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
    
    # Extract OpenMM Torsions (Count terms)
    omm_torsion_counts = {} # (p1, p2, p3, p4) -> count
    
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                t = (p1, p2, p3, p4)
                if p1 > p4:
                    t = (p4, p3, p2, p1)
                
                omm_torsion_counts[t] = omm_torsion_counts.get(t, 0) + 1
                
    print(f"OpenMM Unique Torsions: {len(omm_torsion_counts)}")
    print(f"OpenMM Total Torsion Terms: {sum(omm_torsion_counts.values())}")

    # 3. Setup JAX MD System
    print(colored("\n[3] Setting up JAX MD System...", "yellow"))
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    residues = []
    atom_names = []
    atom_counts = []
    
    for i, chain in enumerate(topology.chains()):
        for j, res in enumerate(chain.residues()):
            residues.append(res.name)
            count = 0
            for atom in res.atoms():
                name = atom.name
                if i == 0 and j == 0 and name == "H": name = "H1"
                atom_names.append(name)
                count += 1
            atom_counts.append(count)

    system_params = jax_md_bridge.parameterize_system(ff, residues, atom_names, atom_counts)
    
    # Extract JAX Torsions
    jax_torsion_counts = {}
    
    # Proper Torsions
    if "dihedrals" in system_params:
        for d in system_params["dihedrals"]:
            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            t = (p1, p2, p3, p4)
            if p1 > p4:
                t = (p4, p3, p2, p1)
            jax_torsion_counts[t] = jax_torsion_counts.get(t, 0) + 1
            
    # Impropers
    if "impropers" in system_params:
        for d in system_params["impropers"]:
            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            # Check permutations against OpenMM to align canonical form
            # OpenMM might have i-j-k-l
            # JAX has i-j-k-l
            # Try all 24 permutations? No, improper is usually unique set of 4.
            # But OpenMM PeriodicTorsionForce defines it as ordered quadruplet.
            # Let's try to find if this set of 4 atoms exists in OMM.
            
            # For counting, we just want to know if we have the term.
            # If OMM has it as (A,B,C,D) and we have (A,C,B,D), it's a mismatch in definition?
            # Or just ordering?
            # Let's assume JAX impropers map to OMM torsions if sorted indices match?
            # No, torsion energy depends on order.
            
            # Let's just add them as is (canonicalized p1>p4)
            t = (p1, p2, p3, p4)
            if p1 > p4: t = (p4, p3, p2, p1)
            
            # If this exact tuple is in OMM, good.
            if t in omm_torsion_counts:
                jax_torsion_counts[t] = jax_torsion_counts.get(t, 0) + 1
            else:
                # Try to find a matching permutation in OMM?
                # This is tricky.
                # Let's just add it.
                jax_torsion_counts[t] = jax_torsion_counts.get(t, 0) + 1

    print(f"JAX MD Unique Torsions: {len(jax_torsion_counts)}")
    print(f"JAX MD Total Torsion Terms: {sum(jax_torsion_counts.values())}")
    
    # Compare Unique Sets
    omm_set = set(omm_torsion_counts.keys())
    jax_set = set(jax_torsion_counts.keys())
    
    missing = omm_set - jax_set
    extra = jax_set - omm_set
    
    print(f"\nMissing Unique Torsions in JAX: {len(missing)}")
    print(f"Extra Unique Torsions in JAX: {len(extra)}")
    
    # Compare Counts for Shared
    shared = omm_set & jax_set
    count_mismatch = 0
    for t in shared:
        if omm_torsion_counts[t] != jax_torsion_counts[t]:
            count_mismatch += 1
            # print(f"Count mismatch for {t}: OMM={omm_torsion_counts[t]}, JAX={jax_torsion_counts[t]}")
            
    print(f"Term Count Mismatches in Shared Torsions: {count_mismatch}")

    # Helper to get atom info
    def get_atom_str(idx):
        res_map = []
        curr = 0
        for r, c in zip(residues, atom_counts):
            for _ in range(c):
                res_map.append(r)
        
        rname = res_map[idx]
        aname = atom_names[idx]
        return f"{rname}:{aname}({idx})"
        
    def get_atom_class(idx):
        # We need to reconstruct the map or use the one from bridge if we could import it.
        # But we can't easily.
        # Let's just print residue/name and let user infer.
        return ""

    # Analyze Missing
    if missing:
        print(colored("\nTop 20 Missing Torsions:", "red"))
        for i, t in enumerate(list(missing)[:20]):
            s = f"{get_atom_str(t[0])}-{get_atom_str(t[1])}-{get_atom_str(t[2])}-{get_atom_str(t[3])}"
            print(f"  {s}")
            
    # Analyze Extra
    if extra:
        print(colored("\nTop 20 Extra Torsions:", "red"))
        for i, t in enumerate(list(extra)[:20]):
            s = f"{get_atom_str(t[0])}-{get_atom_str(t[1])}-{get_atom_str(t[2])}-{get_atom_str(t[3])}"
            print(f"  {s}")


if __name__ == "__main__":
    compare_torsions()
