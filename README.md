# PrxteinMPNN: A functional interface to ProteinMPNN in JAX

[![Test Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/maraxen/PrxteinMPNN/actions/workflows/pytest.yml)
[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maraxen/PrxteinMPNN/blob/main/examples/example_notebook.ipynb)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](http://maraxen.github.io/PrxteinMPNN)

PrxteinMPNN provides a **functional interface for ProteinMPNN**, leveraging the **JAX** ecosystem for accelerated computation and transparent protein design workflows.

## Key Features

- **Faithful parity with LigandMPNN**: All five decoding paths validated to ≥ 0.999 Pearson correlation (see Validation section)
- **Composable inference**: Swap fusion strategies, encode paths, and decode variants via `StageSet` without touching kernel math
- **JAX-native**: `jit`, `vmap`, `scan` throughout; Equinox modules as PyTrees for full AD compatibility
- **Multi-state and membrane support**: physics-conditioned encoder, tied-position product-of-experts, side-chain packer
- **CLI + JSON spec**: `prxteinmpnn spec validate / roundtrip` for portable run specifications

## Documentation

**[Complete Documentation →](http://maraxen.github.io/PrxteinMPNN)**

- [Composition Guide](docs/COMPOSITION_GUIDE.md) — `StageSet`, `InferencePlan`, and the five extension points
- [Parity Validation](docs/parity/parity_report.md) — Numerical parity report vs LigandMPNN reference

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

## Quick Start

### Installation

```bash
uv sync --extra cuda  # For GPU
uv sync --extra tpu   # For TPU
uv sync --extra cpu   # For CPU-only (default)
```

### High-level API

```python
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.run import sample, score, SamplingSpecification, ScoringSpecification

model = load_model(model_version="v_48_020", model_weights="original")

# --- Sampling ---
spec = SamplingSpecification(
    inputs="path/to/structure.pdb",
    num_samples=10,
    temperature=0.1,
    random_seed=42,
)
results = sample(spec)
sequences = results["sequences"]   # (num_samples, seq_len)
logits    = results["logits"]      # (num_samples, seq_len, 21)

# --- Scoring ---
spec = ScoringSpecification(
    inputs="path/to/structure.pdb",
    sequences_to_score=["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"],
    temperature=1.0,
)
results = score(spec)
scores = results["scores"]         # negative log-likelihood per sequence
```

### Composable Inference API

For fine-grained control over fusion strategy, encode path, and decode variant — without touching kernel math:

```python
import jax.numpy as jnp
from prxteinmpnn.host.plan import make_inference_plan, InferencePlan, InferenceComponents
from prxteinmpnn.inference.encode import make_encode_fn
from prxteinmpnn.inference import driver
from prxteinmpnn.inference.logits import GeometricMeanLogits, ARLogitFuse
from prxteinmpnn.run import SamplingSpecification
from prxteinmpnn.types.stages import StageSet

# Option A: factory (resolves stages from spec automatically)
spec = SamplingSpecification(
    inputs="structure.pdb",
    num_samples=10,
    multi_state_strategy="geometric_mean",
    state_weights=[1.0, 0.8, 0.6],
)
plan = make_inference_plan(model, spec)

# Option B: manual assembly for full control
state_weights = jnp.array([1.0, 0.8, 0.6])
stage_set = StageSet(
    logit_transform=GeometricMeanLogits(weights=state_weights, temperature=1.2),
    ar_logit_transform=ARLogitFuse(),
)
plan = InferencePlan(
    model=model,
    components=InferenceComponents(
        encode_fn=make_encode_fn(model, use_rolling_state=False),
        driver=driver.decode,
        stage_set=stage_set,
    ),
)

result = plan.sample(bundle, key, config)   # → SampleResult(sequence, logits)
logits = plan.score(bundle, key, config)    # → (L, 21)
```

Plain callables and lambdas work in all `StageSet` slots (the driver uses `eqx.filter_jit`). Use `eqx.Module` only when the callable carries JAX array leaves (e.g. weights that need grad).

See [Composition Guide](docs/COMPOSITION_GUIDE.md) for the five extension points (`logit_transform`, `ar_logit_transform`, `decode_step`, `sample_step`, `tie_group_fuse`).

### CLI

```bash
# Validate a run specification JSON file
prxteinmpnn spec validate run_spec.json

# Check JSON round-trip fidelity
prxteinmpnn spec roundtrip run_spec.json

# Check portable subset round-trip
prxteinmpnn spec portable-roundtrip portable_spec.json

# Serialize a spec to JSON (from Python)
from prxteinmpnn.run import run_specification_to_json
json_str = run_specification_to_json(spec)
```

## Requirements

- Python ≥ 3.11
- JAX + Equinox (GPU/TPU/CPU via extras)
- `uv sync --extra cpu` for CPU-only; `--extra cuda` for GPU

## Development

| Command | Purpose |
|---------|---------|
| `uv run pytest` | Fast test suite (excludes `parity_heavy`) |
| `uvx ruff check src` | Lint |
| `uv run ty check` | Type check (ty strict) |
| `uv run ruff format .` | Auto-format |

All three decoding paths + membrane + packer are validated via `parity_heavy` tests; see [Running Equivalence Tests](#running-equivalence-tests) below.

## Architecture

```
prxteinmpnn.run          ← SamplingSpecification, ScoringSpecification, sample(), score()
prxteinmpnn.host.plan    ← InferencePlan, InferenceComponents, make_inference_plan()
prxteinmpnn.types.stages ← StageSet (the composition interface)
prxteinmpnn.inference    ← driver.decode, logits (LOGIT_STRATEGIES, TIE_GROUP_STRATEGIES)
prxteinmpnn.model        ← LigandMPNN, Packer (Equinox modules, JIT-safe)
prxteinmpnn.sampling     ← sample() kernel
prxteinmpnn.scoring      ← score() kernel
prxteinmpnn.cli          ← prxteinmpnn spec validate/roundtrip
```

`StageSet` is the composition seam between the host layer and JAX-traced kernels. Everything above it is Python-land; everything below it is traced. See [Composition Guide](docs/COMPOSITION_GUIDE.md).

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
