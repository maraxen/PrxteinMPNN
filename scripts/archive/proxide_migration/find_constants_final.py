import importlib

modules_to_try = [
  "proxide.residue_constants",
  "proxide.core.residue_constants",
  "proxide.utils.residue_constants",
  "proxide.constants",
  "proxide.chem.residue_constants",
  "proxide.chemistry.residue_constants",
  "proxide.data.residue_constants",
]

for mod in modules_to_try:
  try:
    m = importlib.import_module(mod)
    print(f"!!! Found {mod} !!!")
    print(dir(m))
  except ImportError as e:
    print(f"Failed to import {mod}: {e}")

# Check proxide.core.containers for constants
try:
  from proxide.core import containers

  print("\nChecking proxide.core.containers attributes:")
  print([x for x in dir(containers) if not x.startswith("_")])
except ImportError:
  pass
