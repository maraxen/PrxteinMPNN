# Project Status: Equinox Migration Complete

**Date**: April 10, 2026
**Status**: ✅ Complete

## TL;DR

The Equinox migration is **fully complete**. The codebase has been transitioned from a legacy PyTree-based architecture to a modern, modular Equinox-based structure. All core components (features, encoder, decoder, top-level MPNN) are now Equinox modules.

## Current State

### ✅ What's Working
- **Modular Equinox Model**: `src/prxteinmpnn/model/` contains the complete modular implementation.
  - `features.py` - Protein and Ligand feature extraction.
  - `encoder.py` - Encoder layers and stack (including Physics support).
  - `decoder.py` - Decoder layers and stack.
  - `mpnn.py` - `PrxteinMPNN` and `PrxteinLigandMPNN` top-level classes.
- **Decoding Modes**: All 3 decoding modes are supported and optimized:
  - `unconditional` (Parallel scoring)
  - `conditional` (Sequence-conditioned scoring)
  - `autoregressive` (Step-by-step sampling)
- **Multi-State Support**: Fully implemented and validated across both protein and ligand models.
  - Strategies: `arithmetic_mean`, `geometric_mean`, `product`.
  - Validated against LigandMPNN reference implementation.
- **Unified APIs**: `sampling/` and `scoring/` modules updated to work natively with Equinox.
- **Parity Validation**: Numerical equivalence with upstream LigandMPNN (dauparas/LigandMPNN) is maintained and verified in CI.

### 🧹 Cleaned Up
- Removed `src/prxteinmpnn/functional/` legacy logic.
- Removed old PyTree `ModelParameters` pattern.
- Unified model loading through `src/prxteinmpnn/io/weights.py`.

## Technical Architecture

### Type System
All core types are centralized in `prxteinmpnn.utils.types.py`.
- `DecodingApproach`: `"unconditional" | "conditional" | "autoregressive"`
- `MultiStateStrategy`: `"arithmetic_mean" | "geometric_mean" | "product"`

### Multi-State Strategies
Instead of simple logit averaging, the model supports advanced strategies for tied positions:
1. **Arithmetic Mean**: Log-sum-exp averaging for consensus.
2. **Geometric Mean**: Probability-space geometric mean (requires `multi_state_temperature`).
3. **Product**: Multiplies probabilities across states (sums logits), favoring high-probability consistency.

## Implementation Details

### Model Structure
```
src/prxteinmpnn/model/
  ├── __init__.py        # Re-exports core modules
  ├── features.py        # Protein node/edge features
  ├── ligand_features.py # Ligand node/edge features
  ├── encoder.py         # Encoder stack (Protein & Physics)
  ├── decoder.py         # Decoder stack
  ├── mpnn.py            # Top-level PrxteinMPNN & PrxteinLigandMPNN
  └── multi_state_sampling.py # Logit combination strategies
```

### High-Level Entry Points
- `prxteinmpnn.run.sampling.sample()`: Main entry point for sequence design.
- `prxteinmpnn.run.scoring.score()`: Main entry point for sequence evaluation.

## Verification Status

- [x] All unit tests passing: `uv run pytest tests/`
- [x] Numerical parity validated: `Pearson r > 0.95` across all lanes.
- [x] Type checking: `uv run pyright` clean.
- [x] Linting: `uv run ruff check .` clean.

## Documentation Reference
- [Multi-State Implementation Guide](MULTI_STATE_IMPLEMENTATION.md)
- [Final Validation Results](FINAL_VALIDATION_RESULTS.md)
- [Parity Assessment Report](parity/parity_report.md)
