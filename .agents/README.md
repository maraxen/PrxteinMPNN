# PrxteinMPNN Agents Directory

This directory contains documentation and instructions for AI agents (LLMs) working on the PrxteinMPNN project. These documents provide context, guidelines, and implementation plans for various aspects of the codebase.

## Quick Start for Agents

1. **Read this README first** to understand the project structure
2. **Consult `/AGENTS.md`** (root level) for coding standards and development practices
3. **Check relevant documents** in this directory for specific tasks

---

## Project Overview

### What is PrxteinMPNN?

PrxteinMPNN is a **functional JAX implementation of ProteinMPNN** for protein sequence design. It provides:

- A clean, functional API for running ProteinMPNN
- JAX-native implementation with JIT compilation support
- Physics-augmented features (electrostatics, vdW)
- Training pipeline with mixed precision support
- Ensemble processing and conformational analysis

### Core Architecture

```text
src/prxteinmpnn/
├── model/           # Neural network architecture (Equinox)
│   ├── mpnn.py      # Main PrxteinMPNN model
│   ├── encoder.py   # Structure encoder
│   ├── decoder.py   # Sequence decoder
│   └── features.py  # Geometric feature computation
├── io/              # Data loading and parsing
│   ├── loaders.py   # Dataset creation (Grain-based)
│   ├── operations.py # Collation and batching
│   └── parsing/     # Structure file parsing
├── physics/         # Physics-based features
│   └── features.py  # SE(3)-invariant node features
├── training/        # Training pipeline
│   ├── trainer.py   # Main training loop
│   └── specs.py     # Training configuration
├── run/             # User-facing specifications
│   └── specs.py     # RunSpecification and variants
├── sampling/        # Sequence sampling strategies
├── scoring/         # Sequence scoring
└── utils/           # Utilities and data structures
```

### Key Technologies

| Technology | Purpose |
|------------|---------|
| **JAX** | Numerical computation, JIT, vmap, grad |
| **Equinox** | PyTree-based neural network modules |
| **Optax** | Optimizers and learning rate schedules |
| **Orbax** | Checkpoint management |
| **Grain** | Data loading and preprocessing |
| **Biotite** | Structure file parsing |

---

## Related Projects

### Proxide (`maraxen/proxide`)

Structure parsing and force field handling library.

- **oxidize**: Rust extension for fast parsing
- Structure file parsers (PDB, PQR, mmCIF)
- Force field loading and parameter lookup

### Prolix (`maraxen/prolix`)

Molecular dynamics and physics calculations.

- Electrostatics (Coulomb, PME)
- Van der Waals (Lennard-Jones)
- Implicit solvent (GBSA)
- Integrators and minimization

---

## Documents in This Directory

### Active Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | This file - overview and navigation |
| `TRAINING_MERGE.md` | Integration guide for training branch + proxide/prolix migration |
| `TECHNICAL_DEBT.md` | Known issues, experimental features, and TODOs |

### Usage

**Before starting any major task**, check if a relevant document exists here. If you're:

- **Merging training branch**: Read `TRAINING_MERGE.md`
- **Debugging issues**: Check `TECHNICAL_DEBT.md` for known problems
- **Adding new features**: Update relevant docs after completion

---

## Development Guidelines

### Code Style

- **Linter**: Ruff with `select = ["ALL"]`
- **Line length**: 100 characters
- **Indent**: 2 spaces
- **Docstrings**: Google style with type hints

### Running Commands

Always use `uv run` for Python commands:

```bash
uv run pytest tests/ -v       # Run tests
uv run ruff check src/ --fix  # Lint and fix
uv run python script.py       # Run scripts
```

### Testing

- Tests are in `tests/` mirroring `src/` structure
- Use `pytest` and `chex` for JAX-aware testing
- Aim for high coverage on critical paths

### Type Checking

```bash
uv run ty check        # Astral Ty (preferred)
uv run pyright src/    # Alternative
```

---

## Git Workflow

### Branch Naming

- `main` - Stable, production-ready code
- `dev/*` - Development branches
- `merge/*` - Integration branches
- `fix/*` - Bug fixes
- `feat/*` - New features

### Submodules (Development Only)

For cross-repository development (proxide/prolix), use submodules on development branches only. **Never merge submodules to main.** See `TRAINING_MERGE.md` Section 16 for details.

---

## Specification Classes

PrxteinMPNN uses dataclass-based specifications for configuration:

```python
from prxteinmpnn.run.specs import SamplingSpecification

spec = SamplingSpecification(
    inputs="protein.pdb",
    num_samples=10,
    temperature=0.1,
)
```

| Specification | Purpose |
|---------------|---------|
| `RunSpecification` | Base class with common parameters |
| `SamplingSpecification` | Sequence sampling |
| `ScoringSpecification` | Sequence scoring |
| `TrainingSpecification` | Model training |
| `JacobianSpecification` | Categorical Jacobians |

---

## Physics Features

PrxteinMPNN can augment the MPNN with physics-based node features:

1. **Electrostatic features**: Coulomb forces projected onto backbone frame
2. **Van der Waals features**: Lennard-Jones forces projected onto backbone frame

These are SE(3)-invariant and computed in `src/prxteinmpnn/physics/features.py`.

**Note**: The underlying physics calculations are migrating to prolix, but the feature computation remains in PrxteinMPNN.

---

## Common Tasks

### Running Training

```python
from prxteinmpnn.training.specs import TrainingSpecification
from prxteinmpnn.training.trainer import train

spec = TrainingSpecification(
    inputs="data/train/",
    validation_data="data/val/",
    batch_size=8,
    num_epochs=10,
    precision="bf16",
)
result = train(spec)
```

### Running Inference

```python
from prxteinmpnn.run.specs import SamplingSpecification
from prxteinmpnn.run import sample

spec = SamplingSpecification(
    inputs="protein.pdb",
    num_samples=10,
    temperature=0.1,
)
results = sample(spec)
```

---

## Updating This Documentation

When making significant changes to the codebase:

1. **Update relevant `.agents/` documents**
2. **Add new documents** for new major features
3. **Mark completed items** in checklists
4. **Update status** (🟢 Planned, 🟡 In Progress, ✅ Complete)

---

## See Also

- `/AGENTS.md` - Coding standards and JAX idioms
- `/README.md` - Project overview and setup
- `/pyproject.toml` - Dependencies and tool configuration
