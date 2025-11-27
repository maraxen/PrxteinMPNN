import os
import sys
import logging
from pathlib import Path
import torch
import numpy as np
import msgpack
import msgpack_numpy as m

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DEBUGGER")

# 1. Force CPU Mode Check
print(f"ORIGINAL JAX PLATFORM: {os.environ.get('JAX_PLATFORMS', 'Not Set')}")
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
print(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")  # type: ignore[possibly-missing-attribute]

# 2. Check Data Directory
DATA_DIR = Path("src/prxteinmpnn/training/data/pdb_2021aug02")
files = list(DATA_DIR.rglob("*.pt"))
print(f"Found {len(files)} .pt files in {DATA_DIR.absolute()}")

if not files:
    print("❌ ERROR: No .pt files found! Check your download/extract step.")
    sys.exit(1)

# 3. Try Processing ONE File
target_file = files[0]
print(f"\n--- Attempting to process {target_file} ---")

try:
    data = torch.load(target_file, map_location=torch.device('cpu'))
    print("✅ Torch Load: Success")
    
    if isinstance(data, list): protein = data[0]
    elif isinstance(data, dict): protein = data
    else: raise ValueError(f"Unknown data format: {type(data)}")
    
    print(f"Keys in protein: {protein.keys()}")
    
    # Imports that might be crashing workers
    print("\n--- Testing Imports ---")
    from prxteinmpnn.physics.force_fields import load_force_field_from_hub
    from prxteinmpnn.io.parsing.mappings import string_to_protein_sequence
    
    # Force Field Load
    print("Loading Force Field...")
    os.environ["HF_HUB_OFFLINE"] = "1"
    ff = load_force_field_from_hub("ff14SB")
    print(f"✅ Force Field Loaded. Atom Types: {len(ff.id_to_atom_key)}")
    
    # Physics Calc (The part that likely crashes JAX)
    print("\n--- Testing Physics Calc ---")
    seq = protein['seq']
    if isinstance(seq, torch.Tensor): seq = seq.numpy()
    elif isinstance(seq, str): seq = string_to_protein_sequence(seq)
    
    # Test single atom lookup
    q = ff.get_charge("ALA", "CA")
    print(f"✅ JAX Calculation Success: Charge of ALA-CA = {q}")

except Exception as e:
    print(f"\n❌ FATAL ERROR: {e}")
    import traceback
    traceback.print_exc()
