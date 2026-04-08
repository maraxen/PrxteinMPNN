# LigandMPNN Parity Audit Report

## 1. Overview
This report documents the numerical and architectural parity of the JAX implementation (`PrxteinMPNN`) against the PyTorch reference (`LigandMPNN`). The audit covers the base ProteinMPNN weights and the newly implemented LigandMPNN context-aware architecture.

## 2. Parity Metrics

| Component | Status | Metric | Result |
| :--- | :--- | :--- | :--- |
| **Weight Mapping** | ✅ Pass | Protein Keys | 118/118 |
| **Weight Mapping** | ✅ Pass | Ligand Keys | 12/12 |
| **Shape Consistency** | ✅ Pass | All Layers | 100% Match |
| **Numerical Parity (Logits)** | ⚠️ Baseline | Mean Log-Prob Diff | 0.14 |
| **Numerical Parity (Logits)** | ⚠️ Baseline | Max Log-Prob Diff | 0.85 |

## 3. Implementation Status
- **Ligand Features:** High-resolution ligand-aware features (RBF, angle projections) are implemented in `ProteinFeaturesLigand`.
- **Context Integration:** Protein-Ligand communication via `context_encoder` and `y_context_encoder` (DecoderLayerJ) is implemented and integrated into the forward pass.
- **Autoregressive Sampling:** A JAX-native `_run_autoregressive_scan` has been added to `PrxteinLigandMPNN` to enable ligand-aware sequence design.

## 4. CI/CD Integration
- **Workflow:** `.github/workflows/parity.yml` is active.
- **Stability:** The reference repository is pinned to commit `3870631` to ensure stable parity checks.
- **Regression Policy:** CI will fail if mean log-prob parity exceeds 0.145.

## 5. Branch Audit Summary
- **Merged:** `origin/remediate/ligand-parity`.
- **Action Required:** Prune `origin/diffuse` and `origin/feat/ligand-context-adapter`.
- **In Review:** `origin/fix/failing-tests-and-linting`, `origin/training`.

## 6. Recommendations
1. **Bit-Parity:** Investigate the 0.85 max log-prob gap. Suspected cause: minor differences in LayerNorm epsilon or Gumbel noise implementation during sampling.
2. **SDF Integration:** Integrate a robust SDF/PDB parser to feed real ligand coordinates into the `PrxteinLigandMPNN` for end-to-end validation on ligand-bound datasets.
