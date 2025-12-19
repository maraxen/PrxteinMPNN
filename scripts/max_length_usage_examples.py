"""
max_length Usage Examples

The max_length parameter is now available in all specification classes
(RunSpecification, SamplingSpecification, ScoringSpecification, etc.)
"""

from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.specs import SamplingSpecification

# ============================================================================
# Example 1: Reduce memory usage for small proteins
# ============================================================================

# If your protein is ~100 residues, use max_length=128
# This reduces memory by ~16x compared to default 512
spec = SamplingSpecification(
    inputs="structure.pdb",
    max_length=128,  # Pad to 128 instead of 512
    temperature=[0.1, 0.5, 1.0],
    num_samples=8,
    batch_size=4,
)

results = sample(spec=spec)

# Memory savings:
# - Default (512): ~35GB for your current setup
# - With 128: ~2.2GB (16x reduction!)


# ============================================================================
# Example 2: Disable padding entirely (minimal memory, more recompilation)
# ============================================================================

spec = SamplingSpecification(
    inputs="structure.pdb",
    max_length=None,  # No padding at all
    temperature=0.1,
    num_samples=8,
)

results = sample(spec=spec)

# Pros: Minimal memory usage
# Cons: Recompiles for each unique sequence length
# Use when: Memory is critical and you have few unique lengths


# ============================================================================
# Example 3: Dynamic max_length based on your dataset
# ============================================================================

# If you know your longest protein is 150 residues:
spec = SamplingSpecification(
    inputs=["protein1.pdb", "protein2.pdb", "protein3.pdb"],
    max_length=150,  # Only pad to actual max in dataset
    temperature=[0.1, 0.5, 1.0],
    batch_size=4,
)

results = sample(spec=spec)


# ============================================================================
# Example 4: With truncation for very long proteins
# ============================================================================

spec = SamplingSpecification(
    inputs="very_long_protein.pdb",  # e.g., 800 residues
    max_length=256,  # Truncate to 256
    truncation_strategy="center_crop",  # Keep middle region
    temperature=0.1,
)

results = sample(spec=spec)

# Truncation strategies:
# - "none" (default): No truncation, error if protein > max_length
# - "random_crop": Random contiguous segment
# - "center_crop": Keep middle region


# ============================================================================
# Example 5: For your current OOM issue
# ============================================================================

# Based on your error, you're using:
# - batch_size=4
# - 16 samples
# - Multiple temperatures
# - Proteins padded to 512

# Recommended configuration:
spec = SamplingSpecification(
    inputs="structure.pdb",
    max_length=128,  # Reduce from 512 (16x memory reduction)
    temperature=[0.1, 0.5, 1.0],  # 3 temperatures
    num_samples=4,  # Reduce from 16
    batch_size=1,  # Process one structure at a time
    temperature_batch_size=1,  # Process one temp at a time
    samples_batch_size=4,  # Process 4 samples in parallel
)

results = sample(spec=spec)

# This should fit in your 15.75G TPU memory!


# ============================================================================
# Example 6: Checking actual sequence length before setting max_length
# ============================================================================

from prxteinmpnn.io.parsing.biotite import load_protein_from_pdb

# Load your protein to check its length
protein = load_protein_from_pdb("structure.pdb")
seq_length = len(protein.sequence)
print(f"Protein length: {seq_length} residues")

# Set max_length to next power of 2 (for optimal XLA performance)
import math
optimal_max_length = 2 ** math.ceil(math.log2(seq_length))
print(f"Recommended max_length: {optimal_max_length}")

spec = SamplingSpecification(
    inputs="structure.pdb",
    max_length=optimal_max_length,
    temperature=0.1,
)

results = sample(spec=spec)


# ============================================================================
# Memory Calculation Reference
# ============================================================================

"""
Memory usage scales with:
- batch_size
- num_samples  
- len(temperature) if using array
- max_length (quadratically for some operations!)

Approximate memory for your setup:
- max_length=512: ~35GB (OOM on 15.75GB TPU)
- max_length=256: ~9GB (fits!)
- max_length=128: ~2.2GB (plenty of headroom)
- max_length=64: ~0.5GB (very safe)

Formula (rough): 
Memory ∝ batch_size × num_samples × num_temps × max_length²
"""
