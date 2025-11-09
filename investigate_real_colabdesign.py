"""Deep dive into real ColabDesign to understand the divergence."""

import jax
import jax.numpy as jnp
import joblib
from colabdesign.mpnn import mk_mpnn_model

# Load real ColabDesign
print("="*80)
print("INVESTIGATING REAL COLABDESIGN")
print("="*80)

pdb_path = "tests/data/1ubq.pdb"
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)

print("\n1. Inspect mpnn_model state...")
print(f"   Type: {type(mpnn_model)}")
print(f"   Dir: {[x for x in dir(mpnn_model) if not x.startswith('_')]}")

print("\n2. Check what prep_inputs loaded...")
print(f"   Has _inputs: {hasattr(mpnn_model, '_inputs')}")
if hasattr(mpnn_model, '_inputs'):
    inputs = mpnn_model._inputs
    print(f"   _inputs keys: {inputs.keys() if hasattr(inputs, 'keys') else type(inputs)}")
    if hasattr(inputs, 'keys'):
        for key in inputs.keys():
            val = inputs[key]
            if hasattr(val, 'shape'):
                print(f"     {key}: shape {val.shape}, dtype {val.dtype}")
            else:
                print(f"     {key}: {type(val)}")

print("\n3. Check model params...")
if hasattr(mpnn_model, '_params'):
    print(f"   Has _params: True")
    params = mpnn_model._params
    print(f"   Params type: {type(params)}")
    if hasattr(params, 'keys'):
        print(f"   Params keys: {list(params.keys())[:10]}")

print("\n4. Inspect get_unconditional_logits method...")
import inspect
print(f"   Source location: {inspect.getfile(mpnn_model.get_unconditional_logits)}")
print(f"   Signature: {inspect.signature(mpnn_model.get_unconditional_logits)}")

# Try to get source
try:
    source = inspect.getsource(mpnn_model.get_unconditional_logits)
    print(f"\n   Source code:")
    for i, line in enumerate(source.split('\n')[:30], 1):
        print(f"   {i:3d}: {line}")
except:
    print("   (Could not get source)")

print("\n5. Check score method...")
if hasattr(mpnn_model, 'score'):
    print(f"   Has score: True")
    print(f"   Signature: {inspect.signature(mpnn_model.score)}")
    try:
        source = inspect.getsource(mpnn_model.score)
        print(f"\n   Score source (first 30 lines):")
        for i, line in enumerate(source.split('\n')[:30], 1):
            print(f"   {i:3d}: {line}")
    except:
        print("   (Could not get source)")

print("\n6. Call get_unconditional_logits and inspect...")
logits = mpnn_model.get_unconditional_logits()
print(f"   Output shape: {logits.shape}")
print(f"   Output dtype: {logits.dtype}")
print(f"   Output range: [{logits.min():.3f}, {logits.max():.3f}]")
print(f"   Sample values [0, :5]: {logits[0, :5]}")

print("\n7. Check if there's a _model attribute...")
if hasattr(mpnn_model, '_model'):
    print(f"   Has _model: True")
    model = mpnn_model._model
    print(f"   Model type: {type(model)}")
    print(f"   Model keys: {list(model.keys()) if hasattr(model, 'keys') else 'N/A'}")

print("\n8. Check model state dict...")
if hasattr(mpnn_model, '_state'):
    print(f"   Has _state: True")
    state = mpnn_model._state
    print(f"   State type: {type(state)}")
    if hasattr(state, 'keys'):
        print(f"   State keys: {list(state.keys())}")
