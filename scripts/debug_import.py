import sys
print(sys.path)
try:
    from proxide.core.types import ProteinSequence
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
