# PrxteinMPNN LigandMPNN Parity Validation

## Summary

PrxteinMPNN parity is validated against the upstream LigandMPNN reference implementation
(`dauparas/LigandMPNN`, commit `3870631`). The parity suite is split into:

- **Fast parity (`parity_fast`)**: deterministic fixture-backed checks for PR CI.
- **Heavy parity (`parity_heavy`)**: reference-backed numerical checks against LigandMPNN.

## Reported Correlations

| Path | Correlation | Status | Target |
|------|-------------|--------|--------|
| **Unconditional** | 0.984 | ✅ PASS | >0.95 |
| **Conditional** | 0.958-0.984 | ✅ PASS | >0.95 |
| **Autoregressive** | 0.953-0.970 | ✅ PASS | >0.95 |

## Running Validation Locally

```bash
# Install project dependencies
uv sync --extra cpu --extra dev --extra tests --group dev
source .venv/bin/activate

# 1) Get reference repo at pinned commit
git clone https://github.com/dauparas/LigandMPNN.git reference_ligandmpnn_clone
cd reference_ligandmpnn_clone && git checkout 3870631 && cd ..

# 1b) Strict heavy parity preflight
REFERENCE_PATH=./reference_ligandmpnn_clone \
  python scripts/check_parity_prereqs.py --reference-path "$REFERENCE_PATH" --project-root .

# 2) Fast parity checks
pytest tests/parity/test_golden_parity.py -m parity_fast -v

# 3) Heavy reference-backed parity checks
REFERENCE_PATH=./reference_ligandmpnn_clone \
  pytest tests/parity tests/model/test_ligandmpnn_equivalence.py -m parity_heavy -v
```

## CI Routing

- `ci.yml` runs standard tests plus `parity_fast`, while excluding `parity_heavy`.
- `parity.yml` runs heavy parity on `main` and supports manual dispatch.

## Notes

- Heavy parity tests require both the LigandMPNN reference checkout and converted `.eqx` checkpoints.
- Golden fixture regeneration:

```bash
uv run --no-project --with numpy python scripts/generate_parity_golden_fixtures.py
```
