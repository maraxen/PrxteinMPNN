# PrxteinMPNN Codebase Analysis

**Generated:** 2026-05-13  
**Total Python Files:** 136  
**Working Directory:** `/home/marielle/projects/tev_design/prxteinmpnn/src/prxteinmpnn`

## Executive Summary

PrxteinMPNN is a functional interface for ProteinMPNN, organized into 13 major packages with clear separation of concerns:

1. **model** — Core neural network architectures (MPNN, ligand MPNN, encoders, decoders)
2. **sampling** — Logits generation, conditional/unconditional sampling, STE optimization
3. **scoring** — Model evaluation and scoring pipelines
4. **run** — High-level runtime APIs (specification, execution, campaign management)
5. **pipeline** — Modular pipeline definitions (autoregressive, conditional, unconditional, STE)
6. **executor** — Pipeline execution engines (autoregressive, conditional, unconditional)
7. **ensemble** — Post-hoc ensemble methods (GMM, KMeans, DBSCAN, PCA, BIC, CI)
8. **training** — Training loop, checkpointing, diffusion-based training
9. **io** — I/O utilities (designs, weights, parsing)
10. **parity** — Reference implementation comparison and parity testing
11. **psa** — Positional Sequence Annotation (PSA) utilities
12. **profiling** — Performance profiling and HLO analysis
13. **utils** — Shared utilities (coordinates, alignment, graph ops, autoregression, etc.)

## Detailed Module Breakdown

### Core Modules at Package Root

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 35 | Package entry point; exports: `RunSpecification`, `SamplingSpecification`, `ScoringSpecification`, `JacobianSpecification`, `sample`, `score`, `configure_multiprocessing` |
| `cli.py` | 83 | Command-line interface for campaign execution |
| `runtime.py` | 16 | Multiprocessing configuration utilities |
| `registry.py` | 142 | Symbol registry for model architectures and pipeline types |
| `protocols.py` | 349 | Protocol definitions for type contracts (samplers, decoders, executors) |
| `payloads.py` | 276 | Payload dataclass definitions for I/O |
| `model_inputs.py` | 343 | Input specification dataclasses for model execution |
| `pipeline_fns.py` | 151 | Pipeline function shim layer |
| `pipeline_registry.py` | 387 | Registry and factory for pipeline instantiation |
| `padding.py` | 109 | Sequence padding utilities |

### model/ — Neural Network Architectures (11 files, ~6,700 lines)

**Purpose:** Core ProteinMPNN models and supporting architecture components.

| File | Lines | Primary Classes/Functions |
|------|-------|---------------------------|
| `__init__.py` | 35 | Exports: `ProteinMPNN`, `LigandMPNN` |
| `mpnn.py` | 1406 | `ProteinMPNN` — main MPNN architecture |
| `ligand_mpnn.py` | 1389 | `LigandMPNN` — ligand-aware variant |
| `diffusion_mpnn.py` | 324 | `DiffusionMPNN` — diffusion-based model |
| `mpnn_core.py` | 37 | Core MPNN layer definitions |
| `_shared.py` | 156 | Shared model utilities and helpers |
| `encoder.py` | 312 | Node/edge encoder architectures |
| `decoder.py` | 498 | Autoregressive and state-based decoders |
| `features.py` | 284 | Feature extraction and embedding |
| `features_direct.py` | 200 | Direct feature computation |
| `packer.py` | 520 | Residue packing and encoding |
| `capabilities.py` | 37 | Model capability introspection |
| `ligand_features.py` | 359 | Ligand-specific feature extraction |
| `ligand_tiling.py` | 166 | Ligand coordinate tiling |
| `multi_state_sampling.py` | 160 | Multi-state sequence sampling |
| `multistate_stack.py` | 64 | Multi-state context stacking |
| **Autoregressive Variants (4 files, ~1,300 lines):**
| `mpnn_autoregressive_scan.py` | 500 | Scan-based autoregressive decoding |
| `mpnn_autoregressive_state_vmap_exact.py` | 427 | Exact vmap with state tracking |
| `mpnn_autoregressive_state_vmap_exact_ligand.py` | 375 | Ligand-aware exact vmap |
| **Scoring Variants (1 file):**
| `mpnn_scoring_state_vmap_exact_ligand.py` | 454 | Ligand-aware scoring with vmap |

### sampling/ — Sampling & Logit Generation (5 files, ~1,910 lines)

**Purpose:** Logits generation pipelines, sampling algorithms, and optimization.

| File | Lines | Primary Classes/Functions |
|------|-------|---------------------------|
| `__init__.py` | 73 | Exports sampler and related utilities |
| `sample.py` | 895 | Main sampling orchestration; `sample()` function |
| `unconditional_logits.py` | 200 | Unconditional logits generation |
| `conditional_logits.py` | 423 | Conditional logits with context |
| `ste_optimize.py` | 412 | STE optimization for discrete sequences |
| `state_vmap_prep.py` | 280 | State preparation for vmap execution |
| `state_vmap_payload_logits.py` | 138 | Payload logits under vmap |

### scoring/ — Scoring Pipelines (2 files, ~479 lines)

**Purpose:** Model evaluation and sequence scoring.

| File | Lines | Primary Classes/Functions |
|------|-------|---------------------------|
| `__init__.py` | 5 | Empty module stub |
| `score.py` | 474 | Main scoring orchestration; `score()` function |

### run/ — High-Level APIs & Campaign Management (19 files, ~10,000 lines)

**Purpose:** Specification, execution, and campaign management.

| File | Lines | Primary Classes/Functions |
|------|-------|---------------------------|
| `__init__.py` | 50 | Exports main APIs: `RunSpecification`, `SamplingSpecification`, `ScoringSpecification`, `JacobianSpecification` |
| `spec.py` | 309 | Base specification classes |
| `specs.py` | 525 | Extended specification definitions |
| `spec_json.py` | 235 | JSON serialization for specifications |
| `run_spec_portable_json.py` | 256 | Portable JSON format for specs |
| `sampling.py` | 1964 | High-level sampling orchestration |
| `scoring.py` | 611 | High-level scoring orchestration |
| `jacobian.py` | 647 | Jacobian computation and analysis |
| `campaign.py` | 1402 | Campaign management and execution |
| `campaign_manifest.py` | 236 | Campaign manifest handling |
| `averaging.py` | 390 | Result averaging and aggregation |
| `resources.py` | 72 | Resource allocation utilities |
| `prep.py` | 142 | Preparation and preprocessing |
| `multistate_pools.py` | 206 | Multi-state result pooling |
| `conformational_inference.py` | 347 | Conformational inference orchestration |
| `sampling_driver.py` | 54 | Low-level sampling driver |
| `scoring_driver.py` | 36 | Low-level scoring driver |
| `output_sinks.py` | 157 | Output file writing and sinking |
| `streaming_host.py` | 41 | Streaming result handling |
| `inspect.py` | 71 | Specification inspection utilities |
| `_dispatcher.py` | 164 | Internal dispatch routing |
| `decode_registry.py` | 87 | Decoder type registry |

### pipeline/ — Pipeline Definitions (4 files, ~281 lines)

**Purpose:** Modular pipeline abstractions.

| File | Lines | Primary Classes/Functions |
|------|-------|---------------------------|
| `__init__.py` | 26 | Exports pipeline types |
| `autoregressive.py` | 70 | Autoregressive pipeline |
| `conditional.py` | 75 | Conditional pipeline |
| `unconditional.py` | 63 | Unconditional pipeline |
| `ste.py` | 67 | STE optimization pipeline |

### executor/ — Pipeline Execution (5 files, ~251 lines)

**Purpose:** Runtime execution engines for pipelines.

| File | Lines | Primary Classes/Functions |
|------|-------|---------------------------|
| `__init__.py` | 13 | Exports executor types |
| `base.py` | 31 | Base executor protocol |
| `autoregressive.py` | 80 | Autoregressive executor |
| `conditional.py` | 78 | Conditional executor |
| `unconditional.py` | 79 | Unconditional executor |

### ensemble/ — Post-Hoc Ensemble Methods (9 files, ~1,189 lines)

**Purpose:** Clustering, density estimation, and ensemble aggregation.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 1 | Module stub |
| `kmeans.py` | 104 | K-means clustering |
| `gmm.py` | 108 | Gaussian Mixture Models |
| `vmm.py` | 475 | Variational Mixture Models (advanced) |
| `em_fit.py` | 348 | Expectation-Maximization fitting |
| `dbscan.py` | 346 | DBSCAN density-based clustering |
| `pca.py` | 33 | Principal Component Analysis |
| `bic.py` | 70 | Bayesian Information Criterion scoring |
| `ci.py` | 122 | Confidence interval computation |

### training/ — Training Loop & Checkpointing (8 files, ~1,600 lines)

**Purpose:** Model training, diffusion-based training, and checkpoint management.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 6 | Module stub |
| `trainer.py` | 884 | Main training loop orchestration |
| `diffusion.py` | 114 | Diffusion-based training utilities |
| `train_diffusion.py` | 142 | Diffusion training script |
| `losses.py` | 105 | Loss function definitions |
| `metrics.py` | 65 | Training metrics and logging |
| `checkpoint.py` | 119 | Checkpoint save/load utilities |
| `test_diffusion_loop.py` | 95 | Diffusion loop unit tests |
| **dataloading/ (2 files, ~373 lines):**
| `__init__.py` | 1 | Module stub |
| `preprocess.py` | 372 | Data preprocessing and loading |

### io/ — Input/Output Utilities (5 files, ~518 lines)

**Purpose:** File I/O, weight handling, and format dispatch.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 1 | Module stub |
| `designs.py` | 210 | Design result I/O |
| `weights.py` | 280 | Model weight loading and management |
| **parsing/ (2 files, ~131 lines):**
| `__init__.py` | 27 | Parsing module |
| `dispatch.py` | 104 | Format detection and dispatch |

### parity/ — Reference Comparison Testing (4 files, ~677 lines)

**Purpose:** Parity testing against reference implementations.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 14 | Module exports |
| `matrix.py` | 141 | Matrix-level parity checks |
| `evidence.py` | 358 | Evidence collection and reporting |
| `assets.py` | 164 | Reference asset management |

### psa/ — Positional Sequence Annotation (4 files, ~147 lines)

**Purpose:** PSA-based weighting and spatial utilities.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 1 | Module stub |
| `psa.py` | 93 | PSA algorithm implementation |
| `spatial.py` | 19 | Spatial utilities for PSA |
| `weights.py` | 34 | PSA weight computation |

### profiling/ — Performance Profiling (3 files, ~395 lines)

**Purpose:** Performance analysis and HLO introspection.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 1 | Module stub |
| `sampler_profile.py` | 314 | Sampler performance profiling |
| `hlo_tools.py` | 80 | JAX HLO analysis tools |

### utils/ — Shared Utilities (29 files, ~2,700 lines)

**Purpose:** Cross-cutting utilities for data structures, coordinates, alignment, and more.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 33 | Exports main utilities |
| `typing.py` | 25 | Type hints and type utilities |
| `types.py` | 111 | Core type definitions |
| `data_structures.py` | 145 | Custom data structures |
| `coordinates.py` | 266 | 3D coordinate transformations |
| `align.py` | 600 | Sequence alignment utilities |
| `aa_convert.py` | 129 | Amino acid conversion utilities |
| `atom_ordering.py` | 133 | Atom ordering conventions |
| `structure_metrics.py` | 231 | Structure quality metrics |
| `graph.py` | 34 | Graph utilities |
| `concatenate.py` | 28 | Sequence concatenation |
| `batching.py` | 140 | Batch processing utilities |
| `batching_registry.py` | 121 | Batching method registry |
| `safe_map.py` | 52 | Safe mapping over structures |
| `autoregression.py` | 226 | Autoregressive decoding utilities |
| `decoding_order.py` | 108 | Decoding order management |
| `normalize.py` | 82 | Normalization operations |
| `entropy.py` | 134 | Entropy and information measures |
| `radial_basis.py` | 79 | Radial basis functions |
| `gelu.py` | 10 | GELU activation |
| `catjac.py` | 273 | Categorical Jacobian utilities |
| `reverse_jac.py` | 52 | Reverse-mode Jacobian |
| `ste.py` | 97 | Straight-through estimator (STE) |
| `apc.py` | 84 | Average Product Correction (APC) |
| `wave_parallel.py` | 82 | Wave-parallel execution utilities |
| `atomic_write.py` | 30 | Atomic file writing |
| `testing.py` | 16 | Testing utilities |
| `_vendored_callbacks.py` | 60 | Vendored callback utilities |

### model_params/ — Model Parameters (1 file, ~1 line)

**Purpose:** Model parameter registry.

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 1 | Module stub |

## Key Dependency Graph

```
run/ (specs, campaign, API)
  ├── sampling/ (logits, sampling)
  │   ├── model/ (MPNN, ligand MPNN, encoders/decoders)
  │   └── utils/ (coordinates, align, autoregression)
  ├── scoring/ (evaluation)
  │   └── model/
  ├── pipeline/ + executor/ (pipeline definitions and execution)
  ├── io/ (I/O, weights)
  ├── training/ (training loop, checkpoints)
  ├── ensemble/ (post-hoc methods)
  ├── profiling/ (performance analysis)
  └── parity/ (reference testing)

model/
  ├── utils/ (coordinates, alignment, structure metrics)
  ├── io/weights (checkpoint loading)
  └── protocols/ (type contracts)

sampling/
  ├── model/ (logits generation)
  ├── utils/ (autoregression, coordinates, batching)
  └── protocols/ (sampler contract)
```

## Key Public APIs

### Entry Points (from `__init__.py`)

```python
from prxteinmpnn import (
    sample,  # Run sampling
    score,   # Run scoring
    RunSpecification,  # Top-level spec
    SamplingSpecification,
    ScoringSpecification,
    JacobianSpecification,
    configure_multiprocessing,
)
```

### Major Specification Classes (from `run/`)

- `RunSpecification` — Meta-specification for campaigns
- `SamplingSpecification` — Sampling execution spec
- `ScoringSpecification` — Scoring execution spec
- `JacobianSpecification` — Jacobian computation spec

### Major Protocol Definitions (from `protocols.py`)

- `Sampler` — Logits generation interface
- `Decoder` — Sequence decoding interface
- `Executor` — Pipeline execution interface

## Development Patterns

### Module Organization

1. **Root-level exports** in `__init__.py` for public APIs
2. **Protocol definitions** in `protocols.py` for type contracts
3. **Registry patterns** (`registry.py`, `pipeline_registry.py`) for pluggable components
4. **Payloads** (`payloads.py`, `model_inputs.py`) for I/O serialization
5. **Utilities** in `utils/` for cross-cutting concerns

### Testing Patterns

- Parity testing against reference (`parity/`) — ensure equivalence to original
- Unit tests in `tests/` directory (external to this analysis)
- Profiling hooks in `profiling/`

### Multiprocessing

- Configured via `runtime.configure_multiprocessing()`
- Used in campaign execution (`run/campaign.py`)
- Pool-based parallelism for large-scale inference

## Known Technical Debt

- Repository hygiene (stale docs, dead code) — see `__init__.py` TODO
- NPT ensemble instability in training (see CLUSTER.md prolix constraints)
- Module boundary clarity in some utils

---

**Next Steps for Deeper Analysis:**

1. Run semantic search on knowledge base for module relationships
2. Generate dependency graph visualization
3. Document protocol implementations per module
4. Extract API surface for each major module
5. Identify cross-module coupling hotspots
