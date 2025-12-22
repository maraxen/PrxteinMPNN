import importlib
import inspect
from dataclasses import fields, is_dataclass


def check_module(mod_name):
  print(f"\n--- Checking {mod_name} ---")
  try:
    module = importlib.import_module(mod_name)
    print(f"Successfully imported {mod_name}")
    print("Attributes:", dir(module))

    if hasattr(module, "Protein"):
      print(f"!!! Found Protein in {mod_name} !!!")
      Protein = getattr(module, "Protein")
      if is_dataclass(Protein):
        print("Protein is a dataclass")
        for f in fields(Protein):
          print(f"  {f.name}: {f.type}")
      else:
        print("Protein is NOT a dataclass")

    if hasattr(module, "residue_constants"):
      print(f"!!! Found residue_constants in {mod_name} !!!")

  except ImportError as e:
    print(f"Could not import {mod_name}: {e}")
  except Exception as e:
    print(f"Error checking {mod_name}: {e}")


check_module("proxide.core.containers")
check_module("proxide.core.constants")
check_module("proxide.core")
