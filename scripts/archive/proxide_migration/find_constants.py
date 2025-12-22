import proxide
import pkgutil
import importlib

print(f"Proxide path: {proxide.__path__}")


def find_residue_constants(package):
  for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
    try:
      module = importlib.import_module(mod_name)
      if (
        "residue_constants" in modname
        or hasattr(module, "residue_constants")
        or hasattr(module, "atom_types")
      ):
        print(f"Potential match: {modname}")
        if hasattr(module, "atom_types"):
          print(f"  - Has atom_types")
        if hasattr(module, "restypes"):
          print(f"  - Has restypes")
    except Exception:
      pass


find_residue_constants(proxide)
