# HDF5 Streaming Output Format

## Overview

When using the streaming output feature (via `output_h5_path` in `SamplingSpecification`), PrxteinMPNN saves sampled sequences and logits to an HDF5 file using a **group-based structure**. This design accommodates proteins of different lengths without requiring padding during the sampling process.

## File Structure

The HDF5 file contains multiple groups, one for each input structure:

```text
output.h5
├── structure_0/
│   ├── sequences (Dataset)
│   ├── logits (Dataset)
│   └── attributes (metadata)
├── structure_1/
│   ├── sequences (Dataset)
│   ├── logits (Dataset)
│   └── attributes (metadata)
└── structure_N/
    ├── sequences (Dataset)
    ├── logits (Dataset)
    └── attributes (metadata)
```

### Dataset Shapes

For each structure group:

- **`sequences`**: Shape `(num_samples, num_noise_levels, sequence_length)`
  - Data type: `int32`
  - Contains sampled amino acid indices (0-20)
  - When using averaged encodings, `num_noise_levels` = 1

- **`logits`**: Shape `(num_samples, num_noise_levels, sequence_length, 21)`
  - Data type: `float32`
  - Contains the model's output logits for each position and amino acid type
  - When using averaged encodings, `num_noise_levels` = 1

### Attributes

Each structure group contains the following metadata attributes:

- `structure_index` (int): Global index of this structure
- `num_samples` (int): Number of samples generated
- `num_noise_levels` (int): Number of noise levels used (or 1 if averaged)
- `sequence_length` (int): Length of the protein sequence
- `averaged_encodings` (bool, optional): True if encodings were averaged across noise levels

## Reading the Data

### Basic Reading

```python
import h5py

with h5py.File("output.h5", "r") as f:
    # List all structures
    structures = [key for key in f if key.startswith("structure_")]
    
    # Read first structure
    group = f["structure_0"]
    sequences = group["sequences"][:]  # (num_samples, num_noise, seq_len)
    logits = group["logits"][:]        # (num_samples, num_noise, seq_len, 21)
    
    # Access metadata
    seq_length = group.attrs["sequence_length"]
    num_samples = group.attrs["num_samples"]
```

### Using the Helper Functions

See `examples/read_h5_streaming.py` for convenient helper functions:

```python
from examples.read_h5_streaming import read_streaming_results, concatenate_all_structures

# Read individual structures
results = read_streaming_results("output.h5")
for idx, data in results.items():
    print(f"Structure {idx}: {data['sequences'].shape}")

# Concatenate all structures with padding
concatenated = concatenate_all_structures("output.h5")
sequences = concatenated['sequences']  # (num_structures, num_samples, num_noise, max_length)
masks = concatenated['masks']          # Boolean mask for valid positions
```

## Why Group-Based Structure?

1. **Variable Length Support**: Different proteins can have different sequence lengths without requiring upfront padding
2. **Memory Efficiency**: Each structure uses only the memory it needs
3. **Streaming Friendly**: Structures can be written incrementally as they're processed
4. **Easy Access**: Each structure can be accessed independently without loading the entire dataset

## Example Usage

```python
from prxteinmpnn.run import sample, SamplingSpecification
import pathlib

spec = SamplingSpecification(
    inputs=["1UBQ", "1LVM", "1CRN"],
    num_samples=100,
    sampling_strategy="temperature",
    output_h5_path=pathlib.Path("outputs/samples.h5"),
    batch_size=1,
)

result = sample(spec=spec)
print(f"Results saved to: {result['output_h5_path']}")
```

## Averaged Encodings

When `average_encodings=True` in the `SamplingSpecification`, the model will:

1. Encode the structure at each specified noise level
2. Average the node and edge features across noise levels
3. Sample sequences from the averaged representation

This approach is useful when you want to capture information from multiple noise levels while generating samples, resulting in more robust sequence predictions.

### Example with Averaged Encodings

```python
spec = SamplingSpecification(
    inputs=["1UBQ"],
    num_samples=100,
    backbone_noise=[0.0, 0.1, 0.2, 0.5],  # Multiple noise levels to average
    average_encodings=True,  # Average across noise levels
    sampling_strategy="temperature",
    output_h5_path=pathlib.Path("outputs/averaged_samples.h5"),
)

result = sample(spec=spec)
```

In this case, the output will have `num_noise_levels = 1` in the metadata since the encodings from all noise levels were averaged before sampling.

## Migration from Old Format

If you have code expecting the old flat array format, you can easily convert using the `concatenate_all_structures` helper:

```python
# Old code expecting flat arrays
concatenated = concatenate_all_structures("output.h5")
all_sequences = concatenated['sequences']  # Padded to max length
valid_mask = concatenated['masks']         # True for valid positions
lengths = concatenated['lengths']          # Original sequence lengths
```

## Performance Considerations

- **Reading**: Groups can be read independently, allowing parallel processing
- **Memory**: Only load the structures you need rather than the entire file
- **Writing**: Structures are flushed to disk incrementally, avoiding memory buildup
