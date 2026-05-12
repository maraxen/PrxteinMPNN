#!/usr/bin/env python3
"""Quick syntax check."""
import sys
try:
    import py_compile
    py_compile.compile('/home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn/payloads.py', doraise=True)
    print("PASS: payloads.py has valid Python syntax")
    sys.exit(0)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
