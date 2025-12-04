# PrxteinMPNN: A functional interface to ProteinMPNN in JAX

[![Test Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/maraxen/PrxteinMPNN/actions/workflows/pytest.yml)
[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maraxen/PrxteinMPNN/blob/main/examples/example_notebook.ipynb)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](http://maraxen.github.io/PrxteinMPNN)

PrxteinMPNN provides a **functional interface for ProteinMPNN**, leveraging the **JAX** ecosystem for accelerated computation and transparent protein design workflows.

## 🎯 Key Features

- **🔍 Increased Transparency**: Clear and functional interface for ProteinMPNN, enabling users to understand all the operations defining the models flow
- **⚡ JAX Compatibility**: Efficient computation with JAX's functional programming paradigm, including JIT compilation and vectorization
- **🧩 Modular Design**: Maintain a modular structure to facilitate easy updates and extensions to the model
- **🚀 Performance Optimization**: Utilize JAX's capabilities for large-scale protein design tasks
- **🔄 JAX Transformations**: Compatible with `jit`, `vmap`, and `scan` for batch processing and optimization

## 📚 Documentation

**[Complete Documentation →](http://maraxen.github.io/PrxteinMPNN)**

## ✅ Validation

PrxteinMPNN has been **rigorously validated** against the original [ColabDesign ProteinMPNN](https://github.com/sokrypton/ColabDesign) implementation:

| Decoding Path | Correlation | Status |
|---------------|-------------|---------|
| **Unconditional** | 0.984 | ✅ **Validated** |
| **Conditional** | 0.958-0.984 | ✅ **Validated** |
| **Autoregressive** | 0.953-0.970 | ✅ **Validated** |

All three decoding paths achieve **>0.95 Pearson correlation** with ColabDesign outputs, ensuring faithful reproduction of the original model's behavior.

**[View Full Validation Report →](docs/FINAL_VALIDATION_RESULTS.md)**

### Physics Validation (JAX MD vs OpenMM)

PrxteinMPNN's JAX MD implementation has been validated against OpenMM:

- **Structure Loading**: Uses `hydride` for robust hydrogenation and bond inference.
- **Parameters**: Charges, Sigmas, and Epsilons match OpenMM exactly for `ff19SB` (when using compatible XMLs).
- **Energies**:
  - **Bond**: Exact match (validated against OpenMM `ff19SB`).
  - **Angle**: Exact match (validated against OpenMM).
  - **Non-Bonded/GBSA**: Radii and Parameters match exactly. Energy differences reduced but persist due to Torsion/CMAP topology differences affecting exclusion masks.
  - **Torsions/CMAP**: Known parameterization logic gaps in JAX MD (missing some terms compared to OpenMM's implementation).
- **Forces**: Validated to match OpenMM forces when parameters are identical (via injection). End-to-end forces differ due to parameterization differences.

### Implementation Limits

- **Force Fields**: Currently supports `amber19sb` (via `protein19SB.eqx`).
- **Solvent**: Implicit solvent (GBSA-OBC2) is supported. Explicit solvent is not yet fully implemented.
- **PDB Loading**: Relies on `hydride` for adding hydrogens. For *ab initio* structures (no hydrogens), `hydride` may default to neutral N-termini (NH2) unless explicitly charged.
- **Validation**: End-to-end validation against OpenMM requires `openmmforcefields` to be installed to access `amber/protein.ff19SB.xml`.

### Running Equivalence Tests

```bash
# Install ColabDesign for validation tests
pip install git+https://github.com/sokrypton/ColabDesign.git@e31a56f

# Run equivalence tests
uv run pytest tests/model/test_colabdesign_equivalence.py -v
```

## 🚀 Quick Start

### Installation

PrxteinMPNN depends on two companion libraries:

- **[priox](https://github.com/maraxen/priox)**: Protein I/O and physics primitives for JAX
- **[prolix](https://github.com/maraxen/prolix)**: Molecular dynamics and force field integration  

These will be installed automatically as git+ dependencies:

```bash
# Clone and install
git clone https://github.com/maraxen/PrxteinMPNN.git
cd PrxteinMPNN
uv pip install -e .

# Development installation with all test dependencies
uv pip install -e ".[dev]"
```

### Basic Usage

```python
import jax
import jax.numpy as jnp
from prxteinmpnn.mpnn import get_mpnn_model
from prxteinmpnn.io import from_structure_file, protein_structure_to_model_inputs
from prxteinmpnn.scoring.score import make_score_sequence
from prxteinmpnn.sampling import make_sample_sequences, SamplingConfig
from prxteinmpnn.utils.decoding_order import random_decoding_order

# Load a protein structure
protein_structure = from_structure_file(filename="path/to/structure.pdb")

# Get the MPNN model
model = get_mpnn_model(
    model_version="v_48_020",
    model_weights="original"
)

# Get model inputs for the structure
model_inputs = protein_structure_to_model_inputs(protein_structure)

# Score sequences
key = jax.random.PRNGKey(0)
score_sequence = make_score_sequence(
    model, 
    random_decoding_order, 
    model_inputs=model_inputs
)

# Score original sequence
original_score, original_logits, decoding_order = score_sequence(
    key, 
    model_inputs.sequence
)

# Sample new sequences
config = SamplingConfig(sampling_strategy="temperature", temperature=0.1)
sample_sequence = make_sample_sequences(
    model,
    random_decoding_order,
    config=config,
    model_inputs=model_inputs
)

sampled_sequence, logits, decoding_order = sample_sequence(key)
```

## 🛠️ Requirements

- **Python >= 3.11**
- **JAX ecosystem**: jax, jaxlib, flax, equinox
- **Companion libraries**:
  - **priox**: Protein I/O and physics primitives
  - **prolix**: MD simulation and force fields
- **Core dependencies**: NumPy, joblib, jaxtyping
- **Protein handling**: foldcomp, biotite, mdtraj
- **Testing**: chex, pytest, pytest-cov

## 🏗️ Development

### Code Quality & Standards

This project follows strict coding standards:

- **JAX-idiomatic code**: Functional programming paradigm with immutable data structures
- **Linting**: Ruff with strict configuration (line length: 100, all rules enabled)
- **Type checking**: Pyright in strict mode
- **Testing**: Comprehensive unit and integration tests with pytest
- **Documentation**: Google-style docstrings with examples

### Running Tests

```bash
python -m pytest tests/
```

### Linting

```bash
ruff check src/ --fix
```

## 📖 Core Concepts

- **Functional Design**: All operations follow JAX's functional programming paradigm
- **Immutable Data**: Protein structures and model states are immutable
- **JAX Transformations**: Compatible with `jit`, `vmap`, and `scan`
- **Modular Architecture**: Clean separation of concerns across sampling, scoring, and utilities

## 🎯 Project Goals

PrxteinMPNN aims to provide:

1. **Transparency**: A clear, understandable interface to ProteinMPNN's capabilities
2. **Performance**: Leverage JAX for high-performance protein design workflows  
3. **Modularity**: Easy-to-extend components for custom protein design tasks
4. **Compatibility**: Seamless integration with the broader JAX ecosystem

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please see the [contributing guidelines](CONTRIBUTING.md) currently under development for details.

## 📞 Support

- **Documentation**: [http://maraxen.github.io/PrxteinMPNN](http://maraxen.github.io/PrxteinMPNN)
- **Issues**: [GitHub Issues](https://github.com/maraxen/PrxteinMPNN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/maraxen/PrxteinMPNN/discussions)

---

## Built with ❤️ using JAX for the protein design community
