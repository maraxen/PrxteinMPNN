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

PrxteinMPNN is validated against the upstream [LigandMPNN](https://github.com/dauparas/LigandMPNN)
reference implementation (including ProteinMPNN behavior):

| Decoding Path | Correlation | Status |
|---------------|-------------|---------|
| **Unconditional** | 0.984 | ✅ **Validated** |
| **Conditional** | 0.958-0.984 | ✅ **Validated** |
| **Autoregressive** | 0.953-0.970 | ✅ **Validated** |

All three decoding paths achieve **>0.95 Pearson correlation** with reference outputs, ensuring
faithful reproduction of the original model's behavior.

**Canonical parity/equivalence docs (source of truth):**
- [Final validation summary (Markdown)](docs/FINAL_VALIDATION_RESULTS.md)
- [Parity report (Markdown)](docs/parity/parity_report.md)
- [Parity report (HTML)](docs/parity/parity_report.html)
- [Parity report (PDF)](docs/parity/parity_report.pdf)

Legacy root-level parity stubs are non-canonical; use the links above.

### Running Equivalence Tests

```bash
# Install project dependencies (CPU/dev/tests path)
uv sync --extra cpu --extra dev --extra tests --group dev
source .venv/bin/activate

# Checkout reference implementation (pinned commit used in CI)
git clone https://github.com/dauparas/LigandMPNN.git reference_ligandmpnn_clone
cd reference_ligandmpnn_clone && git checkout 3870631 && cd ..

# Optional strict preflight per parity tier
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_prereqs.py --reference-path "$REFERENCE_PATH" --project-root . --tier parity_heavy
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_prereqs.py --reference-path "$REFERENCE_PATH" --project-root . --tier parity_audit

# Validate parity asset cache/checksums
uv run python scripts/check_parity_assets.py --tier parity_fast
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_assets.py --tier parity_heavy
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/check_parity_assets.py --tier parity_audit

# Run fast deterministic parity checks
uv run pytest tests/parity -m parity_fast -v

# Run reference-backed heavy parity checks
REFERENCE_PATH=./reference_ligandmpnn_clone \
  PRXTEIN_PARITY_TIER=parity_heavy \
  uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v

# Convert full checkpoint families and run parity_audit checks
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/convert_parity_family_weights.py \
    --project-root . \
    --reference-path "$REFERENCE_PATH" \
    --tier parity_audit \
    --skip-existing
REFERENCE_PATH=./reference_ligandmpnn_clone \
  PRXTEIN_PARITY_TIER=parity_audit \
  uv run pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_audit -v

# Collect expanded parity evidence (multi-backbone + synthetic random cases)
REFERENCE_PATH=./reference_ligandmpnn_clone \
  uv run python scripts/collect_parity_evidence.py \
    --project-root . \
    --case-corpus tests/parity/parity_case_corpus.json \
    --output-dir docs/parity/reports/evidence

# Render Markdown/HTML report and export PDF with embedded plots/tables
uv run python scripts/generate_parity_report.py --project-root . --output-dir docs/parity --pdf
```

CI tier routing:
- pull_request/main CI excludes `parity_heavy` and `parity_audit` from the default pytest matrix.
- `parity.yml` runs heavy reference-backed checks on `main` push and manual dispatch.
- `parity-audit.yml` runs full-family audit checks on weekly schedule and manual dispatch.
- `ligand-tied-positions-and-multi-state` is staged as warn-only in `parity_heavy` and fail in
  `parity_audit`.

## 🚀 Quick Start

### Installation

```bash
uv sync --extra cuda  # For GPU
uv sync --extra tpu   # For TPU
uv sync --extra cpu   # For CPU-only (default)
```

### Basic Usage

```python
import jax
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.run.sampling import sample
from prxteinmpnn.run.specs import SamplingSpecification

# 1. Load the pre-trained model (Equinox module)
model = load_model(
    model_version="v_48_020",
    model_weights="original"
)

# 2. Configure sampling specification
spec = SamplingSpecification(
    inputs="path/to/structure.pdb",
    num_samples=10,
    temperature=0.1,
    random_seed=42,
    # Multi-state support (optional)
    # multi_state_strategy="product" 
)

# 3. Sample new sequences
results = sample(spec)

# 4. Access results
sequences = results["sequences"]  # (num_samples, seq_len)
logits = results["logits"]        # (num_samples, seq_len, 21)
```

### Scoring Sequences

```python
from prxteinmpnn.run.scoring import score
from prxteinmpnn.run.specs import ScoringSpecification

spec = ScoringSpecification(
    inputs="path/to/structure.pdb",
    sequences_to_score=["MV..."],
    temperature=1.0
)

results = score(spec)
average_scores = results["scores"]  # Negative log-likelihood
```

## 🛠️ Requirements

- **Python >= 3.11**
- **JAX ecosystem**: jax, jaxlib, flax
- **Core dependencies**: NumPy, joblib, jaxtyping
- **Protein handling**: foldcomp, biotite
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
