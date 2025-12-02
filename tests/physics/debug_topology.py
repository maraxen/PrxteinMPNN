import os
import sys
import numpy as np
import jax.numpy as jnp
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

PDB_PATH = "data/pdb/1UBQ.pdb"
FF_EQX_PATH = "src/prxteinmpnn/physics/force_fields/eqx/protein19SB.eqx"

def run_debug():
    print(colored("===========================================================", "cyan"))
    print(colored("   Debug Topology: 1UBQ + ff19SB", "cyan"))
    print(colored("===========================================================", "cyan"))

    # 1. Setup OpenMM System
    print(colored("\n[1] Setting up OpenMM System...", "yellow"))
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w+") as tmp:
        pdb_file_bio = pdb.PDBFile()
        pdb_file_bio.set_structure(atom_array)
        pdb_file_bio.write(tmp)
        tmp.flush()
        tmp.seek(0)
        pdb_file = app.PDBFile(tmp.name)
        topology = pdb_file.topology

    # Load ForceField
    ff19sb_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../openmmforcefields/openmmforcefields/ffxml/amber/protein.ff19SB.xml"))
    if not os.path.exists(ff19sb_xml):
         print(colored(f"Warning: Could not find local protein.ff19SB.xml at {ff19sb_xml}. Using amber14-all.xml", "yellow"))
         omm_ff = app.ForceField('amber14-all.xml', 'implicit/obc2.xml')
    else:
         omm_ff = app.ForceField(ff19sb_xml, 'implicit/obc2.xml')

    omm_system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff)

    # 2. Setup JAX MD System
    print(colored("\n[2] Setting up JAX MD System...", "yellow"))
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

    # 3. Compare Torsions
    jax_dihedrals = np.array(system_params['dihedrals'])
    
    omm_dihedrals_set = set()
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                # Store as tuple of sorted ends? No, torsion is directional p1-p2-p3-p4
                # But p1-p2-p3-p4 is same as p4-p3-p2-p1
                # We need to normalize direction.
                if p1 < p4:
                    key = (p1, p2, p3, p4)
                else:
                    key = (p4, p3, p2, p1)
                
                # Also, OpenMM might have multiple terms for same atoms.
                # We want to count UNIQUE atom quadruplets.
                omm_dihedrals_set.add(key)
                
    jax_dihedrals_set = set()
    for d in jax_dihedrals:
        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        if p1 < p4:
            key = (p1, p2, p3, p4)
        else:
            key = (p4, p3, p2, p1)
        jax_dihedrals_set.add(key)
        
    print(f"\nUnique Torsion Quadruplets:")
    print(f"OpenMM: {len(omm_dihedrals_set)}")
    print(f"JAX MD: {len(jax_dihedrals_set)}")
    
    extra_in_jax = jax_dihedrals_set - omm_dihedrals_set
    missing_in_jax = omm_dihedrals_set - jax_dihedrals_set
    
    print(f"Extra in JAX: {len(extra_in_jax)}")
    print(f"Missing in JAX: {len(missing_in_jax)}")
    
    # Helper to get atom class/type
    def get_atom_info(idx):
        res_name = residues[0] # Simplified, assuming 1UBQ has residues list aligned? 
        # Wait, residues list is just names. We need to map idx to residue.
        # We can reconstruct it from atom_counts.
        
        # Reconstruct map
        current_idx = 0
        res_idx = 0
        for count in atom_counts:
            if idx < current_idx + count:
                break
            current_idx += count
            res_idx += 1
        
        r_name = residues[res_idx]
        a_name = atom_names[idx]
        
        # Get class/type from FF
        # We need to access the internal maps from FF object
        # ff.atom_class_map is dict[str, str] where key is "RES_ATOM"
        key = f"{r_name}_{a_name}"
        a_class = ff.atom_class_map.get(key, "?")
        a_type = ff.atom_type_map.get(key, "?")
        return r_name, a_name, a_class, a_type

    # Create a map from key to params for JAX
    jax_torsion_map = {}
    for i, d in enumerate(jax_dihedrals):
        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        if p1 < p4:
            key = (p1, p2, p3, p4)
        else:
            key = (p4, p3, p2, p1)
        
        # Store params index or params themselves
        # system_params['dihedral_params'] has [periodicity, phase, k]
        # But we might have multiple terms.
        # Let's store list of params
        param = system_params['dihedral_params'][i]
        if key not in jax_torsion_map:
            jax_torsion_map[key] = []
        jax_torsion_map[key].append(param)

    if extra_in_jax:
        print(colored("\n[Sample Extra JAX Torsions]", "red"))


        for i, d in enumerate(list(extra_in_jax)[:10]):

            p1, p2, p3, p4 = d
            r1, n1, c1, t1 = get_atom_info(p1)
            r2, n2, c2, t2 = get_atom_info(p2)
            r3, n3, c3, t3 = get_atom_info(p3)
            r4, n4, c4, t4 = get_atom_info(p4)
            
            print(f"{i+1}. {p1}-{p2}-{p3}-{p4}")
            print(f"   Atoms: {n1}({r1}) - {n2}({r2}) - {n3}({r3}) - {n4}({r4})")
            print(f"   Classes: {c1} - {c2} - {c3} - {c4}")
            print(f"   Types:   {t1} - {t2} - {t3} - {t4}")
            
            params = jax_torsion_map.get(d)
            if params:
                print(f"   Params ({len(params)} terms):")
                for p in params:
                    print(f"     Period: {p[0]}, Phase: {p[1]:.2f}, k: {p[2]:.4f}")

        # Check if any extra torsion has k > 1e-6
        non_zero_extra = 0
        for d in extra_in_jax:
            params = jax_torsion_map.get(d)
            for p in params:
                if abs(p[2]) > 1e-6:
                    non_zero_extra += 1
                    break
        print(f"\nExtra Torsions with k != 0: {non_zero_extra}")

    if missing_in_jax:
        print(colored("\n[Sample Missing JAX Torsions (Present in OpenMM)]", "red"))
        # We need to find these in OpenMM to print params?
        # OpenMM doesn't easily map back to params by key without iterating again.
        # But we can print atoms.
        
        for i, d in enumerate(list(missing_in_jax)[:10]):
            p1, p2, p3, p4 = d
            r1, n1, c1, t1 = get_atom_info(p1)
            r2, n2, c2, t2 = get_atom_info(p2)
            r3, n3, c3, t3 = get_atom_info(p3)
            r4, n4, c4, t4 = get_atom_info(p4)
            
            print(f"{i+1}. {p1}-{p2}-{p3}-{p4}")
            print(f"   Atoms: {n1}({r1}) - {n2}({r2}) - {n3}({r3}) - {n4}({r4})")
            
        # Check if we can find why they are missing.
        # Maybe they are Impropers in OpenMM?
        # OpenMM PeriodicTorsionForce includes Impropers.
        # JAX separates them.
        # Let's check if these missing torsions are in JAX Impropers list.
        
        jax_impropers = np.array(system_params['impropers'])
        jax_impropers_set = set()
        for d in jax_impropers:
            p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
            # Impropers in JAX are [i, j, k, l]
            # OpenMM might store them as [i, j, k, l] or permuted?
            # Improper ordering is usually specific.
            # But let's check if the set of atoms matches.
            key = tuple(sorted((p1, p2, p3, p4)))
            jax_impropers_set.add(key)
            
        found_in_impropers = 0
        for d in missing_in_jax:
            key = tuple(sorted(d))
            if key in jax_impropers_set:
                found_in_impropers += 1
                
        print(f"\nMissing Torsions found in JAX Impropers: {found_in_impropers}")


            
    # Also check total terms (including multi-terms)
    omm_terms = 0
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            omm_terms = force.getNumTorsions()
            
    jax_terms = len(jax_dihedrals)
    
    print(f"\nMissing Torsions found in JAX Impropers: {found_in_impropers}")

    # 4. Compare Parameters for Shared Torsions
    print(colored("\n[4] Comparing Parameters for Shared Torsions", "yellow"))
    
    # Build OpenMM Param Map
    omm_param_map = {}
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                if p1 < p4:
                    key = (p1, p2, p3, p4)
                else:
                    key = (p4, p3, p2, p1)
                
                if key not in omm_param_map:
                    omm_param_map[key] = []
                # OpenMM phase is in radians, k in kJ/mol
                omm_param_map[key].append((period, phase.value_in_unit(unit.radians), k.value_in_unit(unit.kilocalories_per_mole)))

    # Compare
    shared_keys = jax_dihedrals_set.intersection(omm_dihedrals_set)
    print(f"Shared Torsions: {len(shared_keys)}")
    
    max_k_diff = 0.0
    mismatch_count = 0
    
    for key in shared_keys:
        jax_params = jax_torsion_map.get(key, [])
        omm_params = omm_param_map.get(key, [])
        
        # Sort params by periodicity to compare
        jax_params.sort(key=lambda x: x[0])
        omm_params.sort(key=lambda x: x[0])
        
        # Check if number of terms matches
        # Note: JAX might have extra 0-k terms?
        # Filter JAX 0-k terms
        jax_params_nonzero = [p for p in jax_params if abs(p[2]) > 1e-6]
        
        # OpenMM might also have 0-k terms? Unlikely but possible.
        omm_params_nonzero = [p for p in omm_params if abs(p[2]) > 1e-6]
        
        if len(jax_params_nonzero) != len(omm_params_nonzero):
            # print(f"Term count mismatch for {key}: JAX={len(jax_params_nonzero)}, OMM={len(omm_params_nonzero)}")
            mismatch_count += 1
            continue
            
        for jp, op in zip(jax_params_nonzero, omm_params_nonzero):
            # jp: [period, phase, k]
            # op: [period, phase, k]
            
            # Compare Period
            if jp[0] != op[0]:
                mismatch_count += 1
                break
                
            # Compare Phase (careful with sign/units?)
            # OpenMM phase is usually 0 or Pi.
            if abs(jp[1] - op[1]) > 0.01:
                mismatch_count += 1
                break
                
            # Compare k
            if abs(jp[2] - op[2]) > 0.01:
                diff = abs(jp[2] - op[2])
                if diff > max_k_diff:
                    max_k_diff = diff
                mismatch_count += 1
                # print(f"k mismatch for {key}: JAX={jp[2]:.4f}, OMM={op[2]:.4f}")
                break
    
    print(f"Total Mismatches in Shared Torsions: {mismatch_count}")
    print(f"Max k Difference: {max_k_diff:.6f} kcal/mol")
    
    # 5. Compare Impropers (JAX Impropers vs OpenMM Torsions)
    print(colored("\n[5] Comparing JAX Impropers vs OpenMM Torsions", "yellow"))
    
    jax_impropers = np.array(system_params['impropers'])
    jax_improper_params = np.array(system_params['improper_params'])
    
    improper_mismatch = 0
    max_imp_k_diff = 0.0
    
    for i, d in enumerate(jax_impropers):
        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        key = tuple(sorted((p1, p2, p3, p4)))
        
        # Find in OpenMM
        # OpenMM key might be any permutation? 
        # Actually OpenMM Torsion is ordered. Improper is usually centered.
        # But we stored OMM torsions as (min, ..., max) ends?
        # No, we stored as (p1, p2, p3, p4) normalized by ends.
        # Impropers in OpenMM are also stored in PeriodicTorsionForce.
        # But the atom ordering for Improper in OpenMM XML is usually:
        # atom1-atom2-atom3-atom4.
        # We need to check if our key matches any key in omm_param_map?
        # The key definition in omm_param_map was:
        # if p1 < p4: key = (p1, p2, p3, p4) else (p4, p3, p2, p1)
        # This assumes connectivity p1-p2-p3-p4.
        # Impropers are connected differently (center is bonded to 3 others).
        # So the "Torsion" object in OpenMM for an improper might have indices that don't follow a chain.
        # But `getTorsionParameters` just returns 4 indices.
        # Let's try to find the set of 4 atoms in omm_param_map keys (as sets).
        
        # This is slow but robust
        found = False
        target_set = {p1, p2, p3, p4}
        
        omm_match_params = []
        
        # Optimization: Build a set-based map for OMM
        # (Do this once outside loop)
        pass

    # Build Set Map for OMM
    omm_set_map = {}
    for k, v in omm_param_map.items():
        s = frozenset(k)
        if s not in omm_set_map:
            omm_set_map[s] = []
        omm_set_map[s].extend(v)
        
    # Now check
    for i, d in enumerate(jax_impropers):
        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        target_set = frozenset({p1, p2, p3, p4})
        
        if target_set in omm_set_map:
            omm_params = omm_set_map[target_set]
            jax_p = jax_improper_params[i] # [period, phase, k]
            
            # Compare
            # JAX Improper usually has 1 term?
            # OpenMM might have multiple?
            # Let's check if ANY OMM param matches JAX param
            matched_param = False
            for op in omm_params:
                # op: (period, phase, k)
                if abs(jax_p[0] - op[0]) < 0.1 and abs(jax_p[1] - op[1]) < 0.01 and abs(jax_p[2] - op[2]) < 0.01:
                    matched_param = True
                    break
            
            if not matched_param:
                improper_mismatch += 1
                # print(f"Improper Mismatch at {d}: JAX={jax_p}, OMM={omm_params}")
                # Find max diff
                for op in omm_params:
                    diff = abs(jax_p[2] - op[2])
                    if diff > max_imp_k_diff:
                        max_imp_k_diff = diff
        else:
            # print(f"JAX Improper {d} not found in OpenMM Torsions")
            improper_mismatch += 1
            
    print(f"Improper Mismatches: {improper_mismatch}")
    print(f"Max Improper k Difference: {max_imp_k_diff:.6f} kcal/mol")
    
    # 6. Check Improper Ordering & Phase Units
    print(colored("\n[6] Checking Improper Ordering & Phase Units", "yellow"))
    
    # Check Ordering
    exact_order_match = 0
    permuted_match = 0
    
    # We need to look up OpenMM indices for the improper terms
    # We know they are in omm_param_map (which uses normalized keys)
    # But for Impropers, we want to know the RAW OpenMM ordering.
    
    omm_raw_torsions = set()
    for force in omm_system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, period, phase, k = force.getTorsionParameters(i)
                omm_raw_torsions.add((p1, p2, p3, p4))
                
    for d in jax_impropers:
        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        if (p1, p2, p3, p4) in omm_raw_torsions:
            exact_order_match += 1
        elif (p4, p3, p2, p1) in omm_raw_torsions:
            exact_order_match += 1 # Reverse is fine
        else:
            # Check if set matches
            s = {p1, p2, p3, p4}
            # This is slow, but we only have 216
            found_perm = False
            for op in omm_raw_torsions:
                if set(op) == s:
                    found_perm = True
                    break
            if found_perm:
                permuted_match += 1
                
    print(f"JAX Impropers Exact Order Match: {exact_order_match}")
    print(f"JAX Impropers Permuted (Wrong Order) Match: {permuted_match}")
    print(f"Total JAX Impropers: {len(jax_impropers)}")
    
    # Check Phase Units
    print("\nChecking Phase Units (Sample Shared Torsions):")
    for key in list(shared_keys)[:5]:
        jax_params = jax_torsion_map.get(key, [])
        omm_params = omm_param_map.get(key, [])
        
        # Filter nonzero
        jax_params = [p for p in jax_params if abs(p[2]) > 1e-6]
        omm_params = [p for p in omm_params if abs(p[2]) > 1e-6]
        
        if jax_params and omm_params:
            print(f"Torsion {key}:")
            for jp in jax_params:
                print(f"  JAX: Period={jp[0]}, Phase={jp[1]:.4f}, k={jp[2]:.4f}")
            for op in omm_params:
                print(f"  OMM: Period={op[0]}, Phase={op[1]:.4f}, k={op[2]:.4f}")
                
            # Check if JAX phase looks like degrees (e.g. 180.0)
            for jp in jax_params:
                if abs(jp[1]) > 7.0: # > 2*pi
                    print(colored(f"  WARNING: JAX Phase {jp[1]} > 2*pi! Likely Degrees.", "red"))

    # 7. Inspect Permuted Impropers
    print(colored("\n[7] Inspecting Permuted Impropers", "yellow"))
    
    for i, d in enumerate(jax_impropers):
        p1, p2, p3, p4 = int(d[0]), int(d[1]), int(d[2]), int(d[3])
        
        # Check if exact match
        if (p1, p2, p3, p4) in omm_raw_torsions or (p4, p3, p2, p1) in omm_raw_torsions:
            continue
            
        # Must be permuted
        print(f"Permuted Improper {i}: {p1}-{p2}-{p3}-{p4}")
        r1, n1, c1, t1 = get_atom_info(p1)
        r2, n2, c2, t2 = get_atom_info(p2)
        r3, n3, c3, t3 = get_atom_info(p3)
        r4, n4, c4, t4 = get_atom_info(p4)
        print(f"   Atoms: {n1}({r1}) - {n2}({r2}) - {n3}({r3}) - {n4}({r4})")
        print(f"   Classes: {c1} - {c2} - {c3} - {c4}")
        print(f"   Types:   {t1} - {t2} - {t3} - {t4}")
        
        # Print Params
        p = jax_improper_params[i]

        print(f"   Params: Period={p[0]}, Phase={p[1]:.4f}, k={p[2]:.4f}")
        
        # Find OpenMM version
        s = {p1, p2, p3, p4}
        for op in omm_raw_torsions:
            if set(op) == s:
                print(f"   OpenMM has: {op[0]}-{op[1]}-{op[2]}-{op[3]}")
                break

    # 8. Check CMAP Indices
    print(colored("\n[8] Checking CMAP Indices", "yellow"))
    
    # Get OpenMM CMAP indices
    omm_cmap_torsions = []
    for force in omm_system.getForces():
        if isinstance(force, openmm.CMAPTorsionForce):
            for i in range(force.getNumTorsions()):
                map_idx, a1, a2, a3, a4, b1, b2, b3, b4 = force.getTorsionParameters(i)
                omm_cmap_torsions.append({
                    'map_idx': map_idx,
                    'phi': (a1, a2, a3, a4),
                    'psi': (b1, b2, b3, b4)
                })
                
    jax_cmap_torsions = np.array(system_params['cmap_torsions'])
    # JAX: [i, j, k, l, m] -> Phi: i-j-k-l, Psi: j-k-l-m
    # Wait, JAX CMAP is defined by 5 atoms?
    # system.py:
    # phi_indices = cmap_torsions[:, 0:4]
    # psi_indices = cmap_torsions[:, 1:5]
    
    print(f"OpenMM CMAP Terms: {len(omm_cmap_torsions)}")
    print(f"JAX CMAP Terms: {len(jax_cmap_torsions)}")
    
    cmap_mismatch = 0
    
    # We need to match them. Order might differ.
    # Let's build a map from Phi atoms to Psi atoms for OpenMM
    omm_cmap_map = {}
    for item in omm_cmap_torsions:
        phi = item['phi']
        psi = item['psi']
        # Normalize keys?
        # Phi is a torsion.
        if phi[0] < phi[3]: phi_key = phi
        else: phi_key = (phi[3], phi[2], phi[1], phi[0])
        
        omm_cmap_map[phi_key] = psi
        
    for i, t in enumerate(jax_cmap_torsions):
        # t is [i, j, k, l, m]
        # Phi: i-j-k-l
        # Psi: j-k-l-m
        
        p1, p2, p3, p4, p5 = int(t[0]), int(t[1]), int(t[2]), int(t[3]), int(t[4])
        
        phi = (p1, p2, p3, p4)
        psi = (p2, p3, p4, p5)
        
        if phi[0] < phi[3]: phi_key = phi
        else: phi_key = (phi[3], phi[2], phi[1], phi[0])
        
        if phi_key not in omm_cmap_map:
            # Maybe JAX Phi is defined differently?
            # Or maybe we just didn't find it.
            # print(f"CMAP Phi {phi} not found in OpenMM")
            cmap_mismatch += 1
            continue
            
        omm_psi = omm_cmap_map[phi_key]
        
        # Compare Psi
        # Normalize OMM Psi
        if omm_psi[0] < omm_psi[3]: omm_psi_key = omm_psi
        else: omm_psi_key = (omm_psi[3], omm_psi[2], omm_psi[1], omm_psi[0])
        
        if psi[0] < psi[3]: psi_key = psi
        else: psi_key = (psi[3], psi[2], psi[1], psi[0])
        
        if omm_psi_key != psi_key:
            print(f"CMAP Psi Mismatch for Phi {phi}: JAX={psi}, OMM={omm_psi}")
            cmap_mismatch += 1
            
    print(f"CMAP Mismatches: {cmap_mismatch}")




if __name__ == "__main__":
    run_debug()
