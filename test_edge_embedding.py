import jax.numpy as jnp
import numpy as np
import joblib

# Load weights
weights = joblib.load('/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl')
params = weights['model_state_dict']
w = params['protein_mpnn/~/protein_features/~/edge_embedding']['w']  # (416, 128)

# Input from our debug prints
input_vec = jnp.array([-0.09738271,  0.06643821, -0.08947827, -0.2426403,   0.04809081] + [0.0]*(416-5))

# Haiku applies: output = input @ w (input is (416,), w is (416, 128), result is (128,))
output_haiku = input_vec @ w
print("Haiku (input @ w)[:5]:", output_haiku[:5])

# Equinox applies: output = w @ input (w is (128, 416), input is (416,), result is (128,))
w_transposed = w.T  # (128, 416)
output_equinox = w_transposed @ input_vec
print("Equinox (w.T @ input)[:5]:", output_equinox[:5])

# Expected from ColabDesign debug: [-0.403, 0.090, -0.103, -0.581, -0.682]
# Got from PrxteinMPNN debug: [-1.335, 1.093, -0.286, -0.513, -1.684]
print("\nExpected (ColabDesign):", [-0.4029066,   0.08976443, -0.10309178, -0.5805124,  -0.6815348])
print("Got (PrxteinMPNN):", [-1.3347393,   1.0927724,  -0.28576708, -0.51301026, -1.6843283])
