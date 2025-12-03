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

def compare_torsion_params():
    print(colored("===========================================================", "cyan"))
    print(colored("   Comparing JAX MD vs OpenMM Torsion Parameters", "cyan"))
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
    
    # Extract OpenMM Torsions
    # Map: (p1, p2, p3, p4) -> list of (period, phase, k)
    omm_torsions = {}
    
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                
                # Canonicalize indices
                t = (p1, p2, p3, p4)
                if p1 > p4:
                    t = (p4, p3, p2, p1)
                    
                if t not in omm_torsions:
                    omm_torsions[t] = []
                    
                # Convert units
                # Phase: rad
                # k: kJ/mol -> kcal/mol
                k_val = k.value_in_unit(unit.kilocalories_per_mole)
                phase_val = phase.value_in_unit(unit.radians)
                
                omm_torsions[t].append({
                    'period': period,
                    'phase': phase_val,
                    'k': k_val
                })
                
    print(f"OpenMM Unique Torsions: {len(omm_torsions)}")

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
    jax_torsions = {}
    
    # Proper Torsions
    if "dihedrals" in system_params:
        dihedrals = system_params["dihedrals"]
        params = system_params["dihedral_params"]
        
        for i in range(len(dihedrals)):
            d = dihedrals[i]
            p = params[i]
            
            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            period, phase, k = float(p[0]), float(p[1]), float(p[2])
            
            t = (p1, p2, p3, p4)
            if p1 > p4:
                t = (p4, p3, p2, p1)
                
            if t not in jax_torsions:
                jax_torsions[t] = []
                
            jax_torsions[t].append({
                'period': int(period),
                'phase': phase,
                'k': k
            })
            
    # Impropers
    if "impropers" in system_params:
        impropers = system_params["impropers"]
        params = system_params["improper_params"]
        
        for i in range(len(impropers)):
            d = impropers[i]
            p = params[i]
            
            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            period, phase, k = float(p[0]), float(p[1]), float(p[2])
            
            t = (p1, p2, p3, p4)
            if p1 > p4:
                t = (p4, p3, p2, p1)
                
            if t not in jax_torsions:
                jax_torsions[t] = []
                
            jax_torsions[t].append({
                'period': int(period),
                'phase': phase,
                'k': k
            })

    print(f"JAX MD Unique Torsions: {len(jax_torsions)}")
    
    # 4. Compare Parameters
    print(colored("\n[4] Comparing Parameters...", "yellow"))
    
    shared = set(omm_torsions.keys()) & set(jax_torsions.keys())
    print(f"Shared Unique Torsions: {len(shared)}")
    
    mismatches = []
    
    for t in shared:
        omm_terms = omm_torsions[t]
        jax_terms = jax_torsions[t]
        
        # Sort terms to compare (by period, then phase)
        omm_terms.sort(key=lambda x: (x['period'], x['phase']))
        jax_terms.sort(key=lambda x: (x['period'], x['phase']))
        
        if len(omm_terms) != len(jax_terms):
            mismatches.append({
                'type': 'count',
                't': t,
                'omm': len(omm_terms),
                'jax': len(jax_terms)
            })
            continue
            
        for i in range(len(omm_terms)):
            ot = omm_terms[i]
            jt = jax_terms[i]
            
            # Check Period
            if ot['period'] != jt['period']:
                mismatches.append({
                    'type': 'period',
                    't': t,
                    'idx': i,
                    'omm': ot['period'],
                    'jax': jt['period']
                })
                
            # Check Phase (allow small diff)
            if abs(ot['phase'] - jt['phase']) > 1e-4:
                # Check if 2pi equivalent?
                # Or pi vs -pi?
                # Normalize to [0, 2pi)
                p1 = ot['phase'] % (2*np.pi)
                p2 = jt['phase'] % (2*np.pi)
                if abs(p1 - p2) > 1e-4:
                     mismatches.append({
                        'type': 'phase',
                        't': t,
                        'idx': i,
                        'omm': ot['phase'],
                        'jax': jt['phase']
                    })
            
            # Check k (allow small diff)
            if abs(ot['k'] - jt['k']) > 1e-4:
                mismatches.append({
                    'type': 'k',
                    't': t,
                    'idx': i,
                    'omm': ot['k'],
                    'jax': jt['k']
                })

    print(f"Total Mismatches: {len(mismatches)}")
    
    if mismatches:
        print(colored("\nTop 20 Mismatches:", "red"))
        
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

        for m in mismatches[:20]:
            t = m['t']
            s = f"{get_atom_str(t[0])}-{get_atom_str(t[1])}-{get_atom_str(t[2])}-{get_atom_str(t[3])}"
            print(f"  {s} | Type: {m['type']}")
            if 'idx' in m:
                print(f"    Term {m['idx']}: OMM={m['omm']}, JAX={m['jax']}")
            else:
                print(f"    OMM={m['omm']}, JAX={m['jax']}")

    # 5. Calculate Energy Manually
    print(colored("\n[5] Calculating Energy Manually...", "yellow"))
    
    positions = np.array(atom_array.coord) # (N, 3)
    
    def compute_dihedral(p1, p2, p3, p4):
        r1 = positions[p1]
        r2 = positions[p2]
        r3 = positions[p3]
        r4 = positions[p4]
        
        b0 = r1 - r2 # Vector from 2 to 1? No.
        # Bonded.py (Original/Correct): b0 = r_i - r_j (j->i)
        # Standard: b1 = r_j - r_i (i->j)
        # My code uses b0 = r_i - r_j.
        # Let's use the code's definition to match JAX MD.
        b0 = r1 - r2
        b1 = r3 - r2
        b2 = r4 - r3
        
        # Normalize b1
        b1_norm = np.linalg.norm(b1) + 1e-8
        b1_unit = b1 / b1_norm
        
        v = b0 - np.dot(b0, b1_unit) * b1_unit
        w = b2 - np.dot(b2, b1_unit) * b1_unit
        
        x = np.dot(v, w)
        y = np.dot(np.cross(b1_unit, v), w)
        
        return np.arctan2(y, x)

    total_e_omm_calc = 0.0
    total_e_jax_calc = 0.0
    
    # Iterate over all torsions in OMM dict
    for t, terms in omm_torsions.items():
        phi = compute_dihedral(t[0], t[1], t[2], t[3])
        
        for term in terms:
            k = term['k']
            n = term['period']
            phase = term['phase']
            e = k * (1.0 + np.cos(n * phi - phase))
            total_e_omm_calc += e
            
    # Iterate over all torsions in JAX dict
    for t, terms in jax_torsions.items():
        phi = compute_dihedral(t[0], t[1], t[2], t[3])
        
        for term in terms:
            k = term['k']
            n = term['period']
            phase = term['phase']
            e = k * (1.0 + np.cos(n * phi - phase))
            total_e_jax_calc += e
            
    print(f"Calculated Total Energy (OMM Params): {total_e_omm_calc:.4f} kcal/mol")
    print(f"Calculated Total Energy (JAX Params): {total_e_jax_calc:.4f} kcal/mol")
    
    # Get reported energies
    state = omm_system.context.getState(getEnergy=True) if hasattr(omm_system, 'context') else None
    # We didn't create context in this script.
    # But we know reported OMM is 382.4891.
    print(f"Reported OpenMM Energy: ~382.49 kcal/mol")
    print(f"Reported JAX MD Energy: ~374.93 kcal/mol")

if __name__ == "__main__":
    compare_torsion_params()

