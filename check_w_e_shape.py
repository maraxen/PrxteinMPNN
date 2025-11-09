"""Check the shape of W_e weights."""

import joblib

colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
params = joblib.load(colab_weights_path)['model_state_dict']

w_e_proj = params['protein_mpnn/~/W_e']['w']
b_e_proj = params['protein_mpnn/~/W_e']['b']

print(f"W_e weight shape: {w_e_proj.shape}")
print(f"W_e bias shape: {b_e_proj.shape}")

print("\nFor input shape (76, 48, 128):")
print("  ColabDesign: (76, 48, 128) @ (128, 128) + (128,) = (76, 48, 128) ✅")
print(f"  PrxteinMPNN w_e_proj.weight should be: (128, 128)")
print(f"  Actual W_e from ColabDesign: {w_e_proj.shape}")
