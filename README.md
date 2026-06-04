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
- **Temperature array sweep**: pass a list of temperatures for M simultaneous temperatures in one JIT-compiled forward pass — near-ideal M× per-temperature scaling
- **Deduplicated scoring**: score only K unique backbones from an N-structure batch — constant throughput regardless of redundancy ratio
- **Mixed-length batch support**: length-bucketed kernels eliminate padding waste across diverse-length libraries

## Performance

Benchmarked on H200 (NVIDIA SXM5), A100 (PCIe), L40s, and Blackwell (SM120). All figures are warm-call medians; see [full benchmark reports](reports/) for all hardware and configurations.

**H200 — single-structure latency (seq_len=76)**

| Mode | prxteinmpnn | ColabDesign (JAX) | LigandMPNN (PyTorch) | Speedup vs PyTorch |
|---|---|---|---|---|
| Autoregressive sample | 17 ms | 38 ms | 149 ms | **8.7×** |
| Score conditional | 1.5 ms | 7.0 ms | 92 ms | **61×** |

**H200 — Sprint 23 capability benchmarks**

| Capability | Config | prxteinmpnn | PyTorch | Speedup vs PyTorch |
|---|---|---|---|---|
| DedupGather | K=1 unique / N=32 total | 1.1 ms | 92 ms | **80×** |
| DedupGather | K=32 / N=32 (no dedup) | 36 ms | 2958 ms | **82×** |
| Mixed-length batch | lengths [76, 150, 300, 500] | 4.1 ms | 2280 ms | **554×** |
| Temperature array | M=8 temperatures | 2.2 ms/temp | — | **8× per-temp** |

Speedups are hardware-consistent: A100 shows 8–91×, L40s 8–85×, Blackwell (SM120) 8–84× across the same capability suite.

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

### Advanced Capabilities

#### Temperature Array Sweep

Pass a list of temperatures to run all M temperatures in a **single JIT-compiled call**. The kernel vmaps over the temperature dimension — per-temperature cost scales near-ideally (8× cheaper per-temp at M=8 than at M=1, vs 8× ideal).

```python
from prxteinmpnn.run import sample, SamplingSpecification

spec = SamplingSpecification(
    inputs="structure.pdb",
    temperature=[0.1, 0.3, 0.7, 1.0],  # M=4 temperatures, one JIT call
    num_samples=10,
)
results = sample(spec)
# One result dict per temperature, each with "sequences" and "logits"
for temp, result in zip(spec.temperature, results):
    print(f"T={temp}: {result['sequences'].shape}")
```

On H200 at M=8 (seq_len=76), per-temperature cost is **2.2 ms** — the same total wall-clock as M=1 (17 ms) split across 8 temperatures. Measured scaling: 7.97–8.47× across H200/A100/L40s/Blackwell.

---

#### Deduplicated Scoring (DedupGather pattern)

When scoring a large ensemble where many sequences share the same backbone, score only the **K unique** structures rather than all N. prxteinmpnn JIT-caches a compiled kernel per length bucket — K sequential `plan.score()` calls are fast regardless of N.

```python
from prxteinmpnn.host.plan import make_inference_plan
from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.tiling.bucketing import BucketingConfig
import jax.random as random

plan = make_inference_plan(model, spec)
bucket_cfg = BucketingConfig()
key = random.PRNGKey(0)

# Build one bundle per unique backbone (K unique out of N total)
unique_bundles = []
for coords, mask, residue_index, chain_index, sequence in unique_structures:
    bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        ligand_coords=None,
        ligand_atom_types=None,
        ligand_mask=None,
        temperature=1.0,
        mode="score_conditional",
        inference=True,
        bucket_config=bucket_cfg,
    )
    unique_bundles.append((bundle, config))

# Score K unique structures; scatter the K scores to your N positions
scores = [plan.score(bundle, key, config) for bundle, config in unique_bundles]
```

At K=1/N=32 on H200, latency is **1.1 ms** vs 92 ms for PyTorch scoring all 32 structures — **80× speedup**. Speedup is stable across K (80–82× at K=1 through K=32) because PyTorch's cost scales linearly with N while prxteinmpnn's scales linearly with K.

---

#### Mixed-Length Batch Scoring

Score a library of proteins with different sequence lengths without padding waste. prxteinmpnn uses **length bucketing** (rounding to the next power-of-2 boundary) to reuse JIT-compiled kernels across structures that fall in the same bucket — one compile per bucket, not one per structure.

```python
from prxteinmpnn.host.plan import make_inference_plan
from prxteinmpnn.inference.bundle_builder import build_inference_bundle
from prxteinmpnn.tiling.bucketing import BucketingConfig
import equinox as eqx

plan = make_inference_plan(model, spec)
bucket_cfg = BucketingConfig()

# Build one bundle per structure — lengths can differ freely
bundles = []
for coords, mask, residue_index, chain_index, sequence in your_library:
    bundle, config = build_inference_bundle(
        coords=coords,
        mask=mask,
        residue_index=residue_index,
        chain_index=chain_index,
        sequence=sequence,
        ligand_coords=None,
        ligand_atom_types=None,
        ligand_mask=None,
        temperature=1.0,
        mode="score_conditional",
        inference=True,
        bucket_config=bucket_cfg,
    )
    bundles.append((bundle, config))

# Each call reuses the compiled XLA kernel for its length bucket
score_one = eqx.filter_jit(plan.score)
scores = [score_one(bundle, key, config) for bundle, config in bundles]
```

For a batch of [76, 150, 300, 500]-residue structures on H200, total latency is **4.1 ms** versus 2280 ms for PyTorch (padded sequential) — **554× speedup**. Throughput improvement comes from two sources: JAX's compiled kernels (vs PyTorch eager dispatch) and avoiding padding to the longest sequence.

---

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
