# Aminx: A functional interface to ProteinMPNN in JAX

[![PyPI](https://img.shields.io/pypi/v/aminx)](https://pypi.org/project/aminx/)
[![Python](https://img.shields.io/pypi/pyversions/aminx)](https://pypi.org/project/aminx/)
[![Coverage](https://codecov.io/gh/maraxen/Aminx/branch/main/graph/badge.svg)](https://codecov.io/gh/maraxen/Aminx)
[![Run on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maraxen/Aminx/blob/main/examples/example_notebook.ipynb)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](http://maraxen.github.io/Aminx)

> [!WARNING]
> **Alpha release (v0.1.0a1).** aminx is under active development. The API is functional and validated against the LigandMPNN reference, but may change between releases. You may encounter bugs or rough edges — please open an issue if something breaks.

Aminx is a JAX/Equinox reimplementation of the [LigandMPNN](https://github.com/dauparas/LigandMPNN) codebase. It reproduces the PyTorch reference to ≥ 0.999 Pearson correlation across all five decoding paths, and runs 8–61× faster on a single structure (H200) by trading eager dispatch for `jit`/`vmap`/`scan` kernels.

What you get:

- A functional `sample()` / `score()` API — no model objects to wire up, no inference loop to write.
- A composable inference layer (`StageSet`) for swapping logit transforms, encode paths, and decode variants without touching kernel math.
- Numerical parity with upstream LigandMPNN, validated across unconditional, conditional, autoregressive, membrane, and side-chain-packer paths.
- Three capabilities that aren't in vanilla ProteinMPNN: temperature array sweeps, deduplicated scoring, and mixed-length batching — each benchmarked against the PyTorch baseline.

## Performance

Benchmarked on H200 (NVIDIA SXM5), A100 (PCIe), L40s, and Blackwell (SM120). All figures are warm-call medians; see the [full benchmark reports](reports/) for every hardware and configuration.

**H200 — single-structure latency (seq_len=76)**

| Mode | aminx | ColabDesign (JAX) | LigandMPNN (PyTorch) | Speedup vs PyTorch |
|---|---|---|---|---|
| Autoregressive sample | 17 ms | 38 ms | 149 ms | **8.7×** |
| Score conditional | 1.5 ms | 7.0 ms | 92 ms | **61×** |

### Advanced Capability Benchmarks

These are capabilities ProteinMPNN doesn't expose, measured against the PyTorch baseline doing the equivalent work.

| Capability | Config | aminx | PyTorch | Speedup vs PyTorch |
|---|---|---|---|---|
| DedupGather | K=1 unique / N=32 total | 1.1 ms | 92 ms | **80×** |
| DedupGather | K=32 / N=32 (no dedup) | 36 ms | 2958 ms | **82×** |
| Mixed-length batch | lengths [76, 150, 300, 500] | 4.1 ms | 2280 ms | **554×** |
| Temperature array | M=8 temperatures | 2.2 ms/temp | — | **8× per-temp** |

Speedups are hardware-consistent: A100 shows 8–91×, L40s 8–85×, Blackwell (SM120) 8–84× across the same suite.

## Documentation

**[Complete Documentation →](http://maraxen.github.io/Aminx)**

- [Composition Guide](docs/COMPOSITION_GUIDE.md) — `StageSet`, `InferencePlan`, and the five extension points
- [Parity Validation](docs/parity/parity_report.md) — numerical parity report vs the LigandMPNN reference

## Validation

Aminx is validated against the upstream [LigandMPNN](https://github.com/dauparas/LigandMPNN) reference (which includes ProteinMPNN behavior):

| Decoding Path | Tolerance | Status |
|---------------|-----------|---------|
| **Unconditional** | atol/rtol 1e-4, corr ≥ 0.999 | Validated |
| **Conditional** | atol/rtol 1e-4, corr ≥ 0.999 | Validated |
| **Autoregressive** | atol/rtol 1e-4, corr ≥ 0.999 | Validated |
| **Membrane** | atol/rtol 1e-4, corr ≥ 0.999 | Validated |
| **Side-chain packer** | atol 1e-4/1e-3, corr ≥ 0.999 | Validated |

Full parity suite: **30/30 `parity_heavy` tests pass** on the Engaging cluster (job 14203624). 575 fast tests pass locally (575 passed, 6 skipped, 2 xfailed).

Canonical parity docs (source of truth):

- [Final validation summary (Markdown)](docs/FINAL_VALIDATION_RESULTS.md)
- [Parity report (Markdown)](docs/parity/parity_report.md)
- [Parity report (HTML)](docs/parity/parity_report.html)
- [Parity report (PDF)](docs/parity/parity_report.pdf)

Root-level parity stubs are non-canonical; use the links above.

## Quick Start

### Installation

```bash
uv sync --extra cuda  # GPU (CUDA)
uv sync --extra tpu   # TPU
uv sync --extra cpu   # CPU-only (default)
```

Install as a standalone CLI tool (no virtual environment):

```bash
uv tool install aminx
aminx spec validate run_spec.json
```

Or invoke directly without installing:

```bash
uvx aminx spec validate run_spec.json
```

### High-level API

```python
from aminx.io.weights import load_model
from aminx.run import sample, score, SamplingSpecification, ScoringSpecification

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

For control over fusion strategy, encode path, and decode variant without touching kernel math:

```python
import jax.numpy as jnp
from aminx.host.plan import make_inference_plan, InferencePlan, InferenceComponents
from aminx.inference.encode import make_encode_fn
from aminx.inference import driver
from aminx.inference.logits import GeometricMeanLogits, ARLogitFuse
from aminx.run import SamplingSpecification
from aminx.types.stages import StageSet

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

See the [Composition Guide](docs/COMPOSITION_GUIDE.md) for the five extension points (`logit_transform`, `ar_logit_transform`, `decode_step`, `sample_step`, `tie_group_fuse`).

### Advanced Capabilities

#### Temperature Array Sweep

Pass a list of temperatures to run all M in a single JIT-compiled call. The kernel vmaps over the temperature dimension, so per-temperature cost scales near-ideally.

```python
from aminx.run import sample, SamplingSpecification

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

When scoring a large ensemble where many sequences share a backbone, score only the **K unique** structures rather than all N. aminx JIT-caches a compiled kernel per length bucket, so K sequential `plan.score()` calls stay fast regardless of N.

```python
from aminx.host.plan import make_inference_plan
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.tiling.bucketing import BucketingConfig
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

At K=1/N=32 on H200, latency is **1.1 ms** vs 92 ms for PyTorch scoring all 32 structures — an **80× speedup**. The speedup is stable across K (80–82× from K=1 through K=32) because PyTorch's cost scales linearly with N while aminx's scales linearly with K.

---

#### Mixed-Length Batch Scoring

Score a library of proteins with different sequence lengths without padding waste. aminx rounds each length to the next power-of-2 boundary and reuses the JIT-compiled kernel across structures in the same bucket — one compile per bucket, not one per structure.

```python
from aminx.host.plan import make_inference_plan
from aminx.inference.bundle_builder import build_inference_bundle
from aminx.tiling.bucketing import BucketingConfig
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

For a batch of [76, 150, 300, 500]-residue structures on H200, total latency is **4.1 ms** versus 2280 ms for PyTorch (padded sequential) — a **554× speedup**. The gain comes from two sources: JAX's compiled kernels (vs PyTorch eager dispatch) and avoiding padding to the longest sequence.

---

### CLI

```bash
# Validate a run specification JSON file
aminx spec validate run_spec.json

# Check JSON round-trip fidelity
aminx spec roundtrip run_spec.json

# Check portable subset round-trip
aminx spec portable-roundtrip portable_spec.json

# Serialize a spec to JSON (from Python)
from aminx.run import run_specification_to_json
json_str = run_specification_to_json(spec)
```

## Requirements

- Python ≥ 3.12
- JAX + Equinox (GPU/TPU/CPU via extras)
- `uv sync --extra cpu` for CPU-only; `--extra cuda` for GPU

## Development

| Command | Purpose |
|---------|---------|
| `uv run pytest` | Fast test suite (excludes `parity_heavy`) |
| `uv run ruff check src` | Lint |
| `uv run ty check` | Type check (ty strict) |
| `uv run ruff format .` | Auto-format |

All five decoding paths are validated via `parity_heavy` tests — see [Validation Reference](#validation-reference) below.

## Architecture

```
aminx.run          ← SamplingSpecification, ScoringSpecification, sample(), score()
aminx.host.plan    ← InferencePlan, InferenceComponents, make_inference_plan()
aminx.types.stages ← StageSet (the composition interface)
aminx.inference    ← driver.decode, logits (LOGIT_STRATEGIES, TIE_GROUP_STRATEGIES)
aminx.model        ← LigandMPNN, Packer (Equinox modules, JIT-safe)
aminx.sampling     ← sample() kernel
aminx.scoring      ← score() kernel
aminx.cli          ← aminx spec validate/roundtrip
```

`StageSet` is the seam between the host layer and the JAX-traced kernels: everything above it is Python-land, everything below it is traced. See the [Composition Guide](docs/COMPOSITION_GUIDE.md).

## Multiprocessing

Importing `aminx` does **not** set the multiprocessing start method. If your notebook or script spawns worker processes, call `configure_multiprocessing()` once at startup (see `aminx.runtime`); the campaign CLI does this for you.

## Related Tools

`ensemble_tools` was extracted from aminx during refactoring: clustering and conformational inference algorithms (GMM, EM, KMeans, DBSCAN, PCA, BIC, VMM). It lives at `~/projects/ensemble_aminx_tools_WIP/`, is experimental, and is not on PyPI — install with `uv pip install -e ~/projects/ensemble_aminx_tools_WIP` and import as `from ensemble_tools.xxx import yyy`. The `ConformationalStates` type used by `RunSpecification.conformational_states` comes from `ensemble_tools.dbscan`.

## Validation Reference

<details>
<summary>Running the equivalence and parity suite</summary>

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

`AMINX_VERIFY` (runtime jaxtyping + beartype): tests under `tests/parity/` set `AMINX_VERIFY=1` via `tests/parity/conftest.py`. Elsewhere, opt in with:

```bash
AMINX_VERIFY=1 uv run pytest path/to/test.py -v
```

CI tier routing:

- pull_request/main CI excludes `parity_heavy` and `parity_audit` from the default pytest matrix.
- `parity.yml` runs heavy reference-backed checks on `main` push and manual dispatch.
- `parity-audit.yml` runs full-family audit checks on weekly schedule and manual dispatch.
- `ligand-tied-positions-and-multi-state` is staged as warn-only in `parity_heavy` and fail in `parity_audit`.

</details>

## License

MIT License.

## Contributing

Contributions are welcome — see the [contributing guidelines](CONTRIBUTING.md).

## Support

- **Documentation**: [http://maraxen.github.io/Aminx](http://maraxen.github.io/Aminx)
- **Issues**: [GitHub Issues](https://github.com/maraxen/Aminx/issues)
- **Discussions**: [GitHub Discussions](https://github.com/maraxen/Aminx/discussions)

---

Built with ❤️ in JAX, for anyone who'd rather design proteins than write inference loops.
