
import os
import equinox as eqx
import jax.numpy as jnp
import openmm.app as app
from prxteinmpnn.physics.force_fields import FullForceField, save_force_field
import sys
sys.path.append(os.getcwd())
from scripts.convert_all_xmls import parse_xml_to_eqx

def convert_amber14():
    # Find amber14/protein.ff14SB.xml
    app_dir = os.path.dirname(app.__file__)
    data_dir = os.path.join(app_dir, 'data')
    xml_path = os.path.join(data_dir, 'amber14', 'protein.ff14SB.xml')
    
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return
        
    print(f"Converting {xml_path}...")
    output_dir = "src/prxteinmpnn/physics/force_fields/eqx"
    os.makedirs(output_dir, exist_ok=True)
    
    # We want to name it amber14-all.eqx as requested, even though it's just protein
    # Or maybe protein14SB.eqx and symlink?
    # Let's create protein14SB.eqx first.
    
    # We need to modify parse_xml_to_eqx to return the model instead of saving, 
    # or just let it save and rename.
    # But parse_xml_to_eqx saves as {ff_name}.eqx.
    # ff_name for protein.ff14SB.xml is protein.ff14SB.
    
    # Let's just run it.
    parse_xml_to_eqx(xml_path, output_dir)
    
    # Rename
    src = os.path.join(output_dir, "protein14SB.eqx")
    dst = os.path.join(output_dir, "amber14-all.eqx")
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Renamed to {dst}")
    else:
        print(f"Error: {src} not generated.")

if __name__ == "__main__":
    convert_amber14()
