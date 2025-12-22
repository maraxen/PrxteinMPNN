import proxide
import inspect
from dataclasses import fields, is_dataclass

print("\n--- AtomicSystem Class ---")
if hasattr(proxide, "AtomicSystem"):
  AS = proxide.AtomicSystem
  # It might be a pyo3 class, so not a dataclass
  print("Type:", type(AS))
  print("Dir:", dir(AS))
  try:
    # Try to inspect docstring
    print("Doc:", AS.__doc__)
  except:
    pass
else:
  print("AtomicSystem not found.")

# Try to find Protein in submodules
import pkgutil
import importlib

print("\n--- Submodules ---")
package = proxide
for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
  # print(f"Found submodule: {modname}")
  try:
    module = importlib.import_module(modname)
    if hasattr(module, "Protein"):
      print(f"!!! Found Protein in {modname} !!!")
      Protein = getattr(module, "Protein")
      print(dir(Protein))

    if hasattr(module, "residue_constants"):
      print(f"!!! Found residue_constants in {modname} !!!")

  except Exception as e:
    pass
    # print(f"Could not import {modname}: {e}")
