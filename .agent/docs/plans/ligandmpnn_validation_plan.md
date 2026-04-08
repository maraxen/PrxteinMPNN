# LigandMPNN Validation & Equivalence Plan (Revision 2)

## 1. Status of LigandMPNN Validation
- **Architectural Parity:** Core logic (RBF, positional encodings, angle features) is implemented in `ProteinFeaturesLigand`. `PrxteinLigandMPNN` handles context integration but lacks the autoregressive sampling path.
- **Merge Status:** `origin/remediate/ligand-parity` has been merged into the current branch, consolidating parity fixes.
- **Numerical Parity:** Baseline mean log-prob diff is ~0.14 for standard ProteinMPNN. The 0.85 max gap remains to be investigated.

## 2. Codebase Gaps & Technical Depth
### Ligand Features Detail
We need to ensure high-resolution parity for:
- **Atom Pairs:** Verify all 24 atom-pair combinations (N-N, N-CA, etc.) in `ProteinFeaturesLigand.__call__`.
- **Cb Calculation:** Ensure constants `(-0.58273431, 0.56802827, -0.54067466)` match the reference exactly.
- **Angle Features:** `_make_angle_features` must project ligand coordinates into the local residue frame (N, CA, C).

### Autoregressive Sampling
- **Missing Path:** `PrxteinLigandMPNN` needs `_call_autoregressive` implemented to match the decoding logic of the reference.

## 3. Branch Management
- **Pruned:** `origin/diffuse`, `origin/feat/ligand-context-adapter` (remote pruning pending).
- **Consolidated:** `origin/remediate/ligand-parity` merged.

## 4. Thorough Audit and Report Preparation
- **Static Audit:** Compare `src/prxteinmpnn/model/ligand_features.py` constants and logic against the `dauparas/LigandMPNN` reference.
- **Reporting Script:** `scripts/generate_parity_report.py` will:
    - Map PyTorch keys to JAX keys.
    - Check for missing/extra weights.
    - Validate parameter shapes.

## 5. CI/CD Integration
- **Workflow:** `.github/workflows/parity.yml` created.
- **Reference Pinning:** Update workflow to pin `dauparas/LigandMPNN` to commit `3870631`.
- **Regression Guard:** Fail CI if mean log-prob parity for base ProteinMPNN exceeds 0.145.

---

## Execution Steps
1. **Prune Remotes:** `git push origin --delete diffuse feat/ligand-context-adapter` (Manual/Authenticated).
2. **Baseline Verification:** Conduct a static audit of `ligand_features.py` against reference source.
3. **Implement Sampling:** Implement `_call_autoregressive` in `PrxteinLigandMPNN`.
4. **Finalize CI/CD:** Commit the pinned version of `.github/workflows/parity.yml`.
5. **Generate Audit:** Output the `parity_audit.md` report.
