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

**Composition & Extensibility:**
- [Composition Guide](docs/COMPOSITION_GUIDE.md) — Build inference pipelines using `StageSet`, `InferencePlan`, and composable stages

## ✅ Validation

PrxteinMPNN is validated against the upstream [LigandMPNN](https://github.com/dauparas/LigandMPNN)
reference implementation (including ProteinMPNN behavior):

| Decoding Path | Tolerance | Status |
|---------------|-----------|---------|
| **Unconditional** | atol/rtol 1e-4, corr ≥ 0.999 | ✅ **Validated** |
| **Conditional** | atol/rtol 1e-4, corr ≥ 0.999 | ✅ **Validated** |
| **Autoregressive** | atol/rtol 1e-4, corr ≥ 0.999 | ✅ **Validated** |
| **Membrane** | atol/rtol 1e-4, corr ≥ 0.999 | ✅ **Validated** |
| **Side-chain packer** | atol 1e-4/1e-3, corr ≥ 0.999 | ✅ **Validated** |

Full parity suite: **30/30 `parity_heavy` tests pass** on Engaging cluster (job 14203624).
575 fast tests pass locally (575 passed, 6 skipped, 2 xfailed).

**Canonical parity/equivalence docs (source of truth):**
- [Final validation summary (Markdown)](docs/FINAL_VALIDATION_RESULTS.md)
- [Parity report (Markdown)](docs/parity/parity_report.md)
- [Parity report (HTML)](docs/parity/parity_report.html)
- [Parity report (PDF)](docs/parity/parity_report.pdf)

Legacy root-level parity stubs are non-canonical; use the links above.

## Related Tools

The following packages were extracted from prxteinmpnn during refactoring:

- **`ensemble_tools`** — Clustering and conformational inference algorithms (GMM, EM, KMeans, DBSCAN, PCA, BIC, VMM). Located at `~/projects/ensemble_prxteinmpnn_tools_WIP/`. Experimental, not published to PyPI. Install with `uv pip install -e ~/projects/ensemble_prxteinmpnn_tools_WIP`. Import as `from ensemble_tools.xxx import yyy`. The `ConformationalStates` type used by `RunSpecification.conformational_states` comes from `ensemble_tools.dbscan`.

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

**`PRXTEINMPNN_VERIFY` (runtime jaxtyping + beartype):** tests under `tests/parity/` set
`PRXTEINMPNN_VERIFY=1` via `tests/parity/conftest.py` (refactor roadmap §13 Q5). Elsewhere, opt in with:

```bash
PRXTEINMPNN_VERIFY=1 uv run pytest path/to/test.py -v
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

### Basic Usage (via `sample()` high-level API)

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
    # multi_state_strategy="arithmetic_mean",
    # state_weights=[1.0, 0.8, 0.6]
)

# 3. Sample new sequences
results = sample(spec)

# 4. Access results
sequences = results["sequences"]  # (num_samples, seq_len)
logits = results["logits"]        # (num_samples, seq_len, 21)
```

### Composable Inference API (Sprint 2)

For fine-grained control over fusion strategies, encoding, and decoding stages:

```python
import jax
import jax.numpy as jnp
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.host.plan import make_inference_plan
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.types.stages import StageSet
from prxteinmpnn.inference.logits import GeometricMeanLogits, ARLogitFuse

# 1. Load model
model = load_model(model_version="v_48_020", model_weights="original")

# 2. Create spec with multi-state parameters
spec = SamplingSpecification(
    inputs="path/to/structure.pdb",
    num_samples=10,
    temperature=0.1,
    multi_state_strategy="geometric_mean",  # or "arithmetic_mean", "product", etc.
    state_weights=[1.0, 0.8, 0.6],
)

# 3. Create a composable inference plan
plan = make_inference_plan(model, spec)  # Resolves stages and encoding strategy

# 4. Sample or score using the plan
# plan.sample(bundle, key, config) → SampleResult
# plan.score(bundle, key, config)  → Logits

# Or customize by building InferencePlan components directly
# (see docs/COMPOSITION_GUIDE.md for advanced patterns)
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
- **Composable Inference** (Sprint 2): Use `StageSet` to swap fusion strategies, encoding methods, and decode variants without touching kernel math. See [Composition Guide](docs/COMPOSITION_GUIDE.md)

## 🎯 Project Goals

PrxteinMPNN aims to provide:

1. **Transparency**: A clear, understandable interface to ProteinMPNN's capabilities
2. **Performance**: Leverage JAX for high-performance protein design workflows  
3. **Modularity**: Easy-to-extend components for custom protein design tasks
4. **Compatibility**: Seamless integration with the broader JAX ecosystem

## Multiprocessing

Importing `prxteinmpnn` does **not** set the multiprocessing start method. If your notebook or script spawns worker processes, call `configure_multiprocessing()` once at startup (see `prxteinmpnn.runtime`); the campaign CLI does this for you.

---

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
