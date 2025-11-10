"""Extract the full vectors from debug output and compare them."""

import re
import numpy as np

# Parse the debug output
with open('/tmp/full_edges_output.txt', 'r') as f:
    content = f.read()

# Extract ColabDesign vector
colab_match = re.search(r'E\[0,0\] FULL COLABDESIGN VECTOR: \[(.*?)\]', content, re.DOTALL)
if colab_match:
    colab_str = colab_match.group(1)
    # Parse the scientific notation numbers
    colab_values = re.findall(r'[-+]?\d+\.?\d*e?[-+]?\d*', colab_str)
    colab_vec = np.array([float(v) for v in colab_values])
    print(f"ColabDesign vector length: {len(colab_vec)}")
    print(f"First 20 values: {colab_vec[:20]}")
    print(f"Values 16-32 (RBF start): {colab_vec[16:32]}")
else:
    print("Could not find ColabDesign vector")

# Extract PrxteinMPNN vector
prx_match = re.search(r'edges\[0,0\] FULL VECTOR: \[(.*?)\]', content, re.DOTALL)
if prx_match:
    prx_str = prx_match.group(1)
    prx_values = re.findall(r'[-+]?\d+\.?\d*e?[-+]?\d*', prx_str)
    prx_vec = np.array([float(v) for v in prx_values])
    print(f"\nPrxteinMPNN vector length: {len(prx_vec)}")
    print(f"First 20 values: {prx_vec[:20]}")
    print(f"Values 16-32 (RBF start): {prx_vec[16:32]}")
else:
    print("Could not find PrxteinMPNN vector")

# Compare
if colab_match and prx_match:
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    min_len = min(len(colab_vec), len(prx_vec))
    print(f"Comparing first {min_len} values...")

    match = np.allclose(colab_vec[:min_len], prx_vec[:min_len], atol=1e-6)
    print(f"Vectors match (atol=1e-6): {match}")

    if not match:
        diffs = np.abs(colab_vec[:min_len] - prx_vec[:min_len])
        max_diff_idx = np.argmax(diffs)
        print(f"\nMax diff at index {max_diff_idx}:")
        print(f"  ColabDesign: {colab_vec[max_diff_idx]}")
        print(f"  PrxteinMPNN: {prx_vec[max_diff_idx]}")
        print(f"  Diff: {diffs[max_diff_idx]}")

        # Find first significant difference
        sig_diff_idx = np.where(diffs > 1e-5)[0]
        if len(sig_diff_idx) > 0:
            first_diff = sig_diff_idx[0]
            print(f"\nFirst significant diff (>1e-5) at index {first_diff}:")
            print(f"  ColabDesign: {colab_vec[first_diff]}")
            print(f"  PrxteinMPNN: {prx_vec[first_diff]}")
            print(f"  Diff: {diffs[first_diff]}")
    else:
        print("✅ Vectors are IDENTICAL!")
        print("\nBut wait - if inputs are identical, why are outputs different?")
        print("The issue must be in the edge_embedding layer application!")
