import importlib

modules_to_try = [
  "proxide.chem",
  "proxide.chem.residues",
  "proxide.chem.constants",
]

for mod in modules_to_try:
  print(f"\n--- Checking {mod} ---")
  try:
    m = importlib.import_module(mod)
    print(f"Successfully imported {mod}")
    print("Attributes:", dir(m))

    if hasattr(m, "restypes"):
      print(f"!!! Found restypes in {mod} !!!")
      print(m.restypes)

    if hasattr(m, "residue_constants"):
      print(f"!!! Found residue_constants in {mod} !!!")
      print(dir(m.residue_constants))

  except ImportError as e:
    print(f"Failed to import {mod}: {e}")
  except Exception as e:
    print(f"Error checking {mod}: {e}")
