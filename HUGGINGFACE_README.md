---
license: mit
tags:
- protein-design
- protein-mpnn
- jax
- equinox
- biology
- structure-based-design
library_name: equinox
---

# PrxteinMPNN

A JAX/Equinox implementation of ProteinMPNN for inverse protein folding and sequence design.

## Model Description

PrxteinMPNN is a message-passing neural network that generates amino acid sequences given a protein backbone structure. This implementation uses JAX and Equinox for efficient computation and functional programming patterns.

**Key Features:**
- Fully modular Equinox implementation
- JAX-based for GPU acceleration and automatic differentiation
- Multiple pre-trained model variants (original and soluble)
- Multiple training epochs (002, 010, 020, 030)

## Available Models

All models use the same architecture with different training:

### Original Models
- `original_v_48_002` - Trained for 2 epochs
- `original_v_48_010` - Trained for 10 epochs  
- `original_v_48_020` - Trained for 20 epochs (recommended)
- `original_v_48_030` - Trained for 30 epochs

### Soluble Models
- `soluble_v_48_002` - Trained for 2 epochs on soluble proteins
- `soluble_v_48_010` - Trained for 10 epochs on soluble proteins
- `soluble_v_48_020` - Trained for 20 epochs on soluble proteins (recommended)
- `soluble_v_48_030` - Trained for 30 epochs on soluble proteins

## Installation

```bash
pip install jax equinox huggingface_hub
```

## Usage

### Basic Usage

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from huggingface_hub import hf_hub_download

# Download model from HuggingFace
model_path = hf_hub_download(
    repo_id="maraxen/prxteinmpnn",
    filename="eqx/original_v_48_020.eqx",
    repo_type="model",
)

# Create model structure (must match saved architecture)
from prxteinmpnn.eqx_new import PrxteinMPNN

key = jax.random.PRNGKey(0)
model = PrxteinMPNN(
    node_features=128,
    edge_features=128,
    hidden_features=512,
    num_encoder_layers=3,
    num_decoder_layers=3,
    vocab_size=21,
    k_neighbors=48,
    key=key,
)

# Load weights
model = eqx.tree_deserialise_leaves(model_path, model)

# Use model for inference
# ... (see full documentation for inference examples)
```

### Using the High-Level API

```python
from prxteinmpnn.io.weights import load_model

# Automatically downloads and loads the model
model = load_model(
    model_version="v_48_020",
    model_weights="original"
)
```

## Model Architecture

**Hyperparameters:**
- Node features: 128
- Edge features: 128
- Hidden features: 512
- Encoder layers: 3
- Decoder layers: 3
- K-nearest neighbors: 48
- Vocabulary size: 21 (20 amino acids + 1 unknown)

**Architecture:**
- Message-passing encoder for structural features
- Autoregressive decoder for sequence generation
- Attention-based edge updates
- LayerNorm and residual connections

## Training Data

The models were trained on protein structures from the Protein Data Bank (PDB):
- **Original models:** Standard PDB training set
- **Soluble models:** Filtered for soluble, well-expressed proteins

## Performance

These models achieve state-of-the-art performance on:
- Native sequence recovery
- Structural compatibility (predicted structure vs. designed sequence)
- Expressibility and stability (for soluble models)

## Citation

If you use PrxteinMPNN in your research, please cite the original ProteinMPNN paper:

```bibtex
@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```

## License

MIT License - See LICENSE file for details.

## Links

- **GitHub Repository:** [maraxen/PrxteinMPNN](https://github.com/maraxen/PrxteinMPNN)
- **Original ProteinMPNN:** [dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- **Documentation:** [Full documentation](https://github.com/maraxen/PrxteinMPNN/tree/main/docs)

## Technical Details

### File Format

Models are saved using Equinox's `tree_serialise_leaves` format (`.eqx` files), which:
- Preserves PyTree structure
- Ensures bit-perfect reproducibility
- Is compatible with JAX's functional programming paradigm
- Supports efficient serialization/deserialization

### Computational Requirements

- **Memory:** ~30 MB per model
- **Inference:** CPU-compatible, GPU-accelerated
- **Batch processing:** Supported via `jax.vmap`

## Updates

**Latest (v2.0):**
- Migrated to unified Equinox architecture
- All models now in `.eqx` format
- Improved modularity and type safety
- Full JAX compatibility with JIT, vmap, and grad

---

For more information, examples, and tutorials, visit the [GitHub repository](https://github.com/maraxen/PrxteinMPNN).
