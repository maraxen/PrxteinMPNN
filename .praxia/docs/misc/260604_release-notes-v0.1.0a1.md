---
title: Release Notes — v0.1.0a1
date: 260604
tag: v0.1.0a1
---

# prxteinmpnn v0.1.0a1 — Alpha Release

## What's Included

- **Composability primitives**: StageSet, eqx.Module-based metrics + data structures (flax dependency removed)
- **Sprint 23 benchmarks**: >0.95 Pearson correlation across all three decoding paths (unconditional, conditional, autoregressive) vs. LigandMPNN reference
- **Modern CI**: Python 3.12, idiomatic `uv run`, three workflow files (ci.yml, parity_heavy.yml, parity_audit.yml)
- **Cleaned repo**: legacy orchestration scaffolding archived, development artifacts removed, root is lean
- **Colab notebook**: `examples/colab_inference_demo.ipynb` — widget-based inference demo, Colab free-tier compatible

## Install

```bash
# Requires PyPI publication of v0.1.0a1 (run after: git push origin v0.1.0a1 + PyPI upload)
pip install prxteinmpnn==0.1.0a1

# Or with uv (once published):
uv tool install prxteinmpnn
```

## Development Setup

```bash
git clone https://github.com/maraxen/PrxteinMPNN
cd PrxteinMPNN
uv sync
uv run pytest
```

## Known Limitations

- **Alpha quality**: API may change before beta; no stability guarantee between alpha versions
- **PyPI publish**: pending — package must be uploaded after tagging (`twine upload dist/*` or `uv publish`)
- **ReadTheDocs**: `fail_on_warning` is currently disabled; Sphinx warnings tracked in `.praxia/docs/misc/260604_sphinx-warning-backlog.md`
- **Training module**: `prxteinmpnn.training` raises `NotImplementedError` — scheduled for Sprint 3

## Upgrade Path to Beta

See `.praxia/docs/adr/260604_versioning-strategy.md` for alpha→beta→stable promotion criteria.
