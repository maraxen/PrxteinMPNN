"""Inspect the _score method that does the actual forward pass."""

import jax
import jax.numpy as jnp
import inspect
from colabdesign.mpnn import mk_mpnn_model

print("="*80)
print("INSPECTING _score METHOD")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)

print("\n1. Check _score method...")
if hasattr(mpnn_model, '_score'):
    print(f"   Has _score: True")
    print(f"   Type: {type(mpnn_model._score)}")
    print(f"   Signature: {inspect.signature(mpnn_model._score)}")

    try:
        source = inspect.getsource(mpnn_model._score)
        print(f"\n   _score source:")
        lines = source.split('\n')
        for i, line in enumerate(lines[:50], 1):
            print(f"   {i:3d}: {line}")
        if len(lines) > 50:
            print(f"   ... ({len(lines)-50} more lines)")
    except:
        print("   (Could not get source)")

print("\n2. Check _model.apply...")
if hasattr(mpnn_model, '_model'):
    model = mpnn_model._model
    print(f"   _model type: {type(model)}")
    print(f"   _model dir: {[x for x in dir(model) if not x.startswith('_')]}")

    if hasattr(model, 'apply'):
        print(f"   Has apply: True")
        print(f"   Apply signature: {inspect.signature(model.apply)}")

print("\n3. Check what inputs _score receives...")
inputs = mpnn_model._inputs
print(f"   X shape: {inputs['X'].shape}")
print(f"   mask shape: {inputs['mask'].shape}")
print(f"   S shape: {inputs['S'].shape}")
print(f"   residue_idx shape: {inputs['residue_idx'].shape}")
print(f"   chain_idx shape: {inputs['chain_idx'].shape}")

print("\n4. Let's trace through score() call...")
# Call with instrumentation
import numpy as np
L = inputs["X"].shape[0]
ar_mask = np.zeros((L, L))

print(f"   Calling score with ar_mask shape: {ar_mask.shape}")

# Check if key is used
print(f"\n5. Check if score is deterministic...")
logits1 = mpnn_model.get_unconditional_logits()
logits2 = mpnn_model.get_unconditional_logits()
print(f"   Logits 1 [0, :5]: {logits1[0, :5]}")
print(f"   Logits 2 [0, :5]: {logits2[0, :5]}")
print(f"   Identical: {jnp.allclose(logits1, logits2)}")

print("\n6. Check what X contains (first 3 residues)...")
X = inputs['X']
print(f"   X shape: {X.shape}")
print(f"   X[0] (first residue, 4 atoms):")
for i in range(4):
    print(f"     Atom {i}: {X[0, i]}")

print(f"\n   X[1] (second residue, 4 atoms):")
for i in range(4):
    print(f"     Atom {i}: {X[1, i]}")
