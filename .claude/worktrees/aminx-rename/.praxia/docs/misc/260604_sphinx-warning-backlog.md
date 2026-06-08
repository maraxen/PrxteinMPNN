---
title: Sphinx Warning Backlog
created: 260604
status: open
---

# Sphinx Build Warnings — Resolution Backlog

`fail_on_warning` is deferred in `.readthedocs.yml` until these are resolved.

## Environment Note

`sphinx-build` was not available in the build environment at the time of the Phase 2 release-prep workflow. The warning audit (DOCS-005) could not run. This file is a placeholder for any warnings that surface when Sphinx is available.

## Known Risk Areas

The following items are likely to generate warnings and should be checked:

1. **COMPOSITION_GUIDE.md toctree inclusion** — `docs/source/index.rst` does not yet reference `COMPOSITION_GUIDE.md`. If myst-parser is not configured, the `.md` extension may not resolve in a toctree. Either add a `.rst` stub wrapper or configure myst in `docs/source/conf.py`.

2. **Stale cross-references** — Sprint 23 refactor removed several symbols (flax structs, old module paths). Any autodoc directives referencing removed symbols will generate warnings.

3. **`docs/_build/` in git index** — `.gitignore` should exclude `docs/_build/` but tracked build artifacts cause issues. Run `git ls-files docs/_build/` to verify it is empty.

## Resolution Plan

- [ ] Run `uv run python -m sphinx docs/source docs/_build/html -W --keep-going` once sphinx-build is available in the dev environment
- [ ] Triage each warning and either fix the source or add a `nitpick_ignore` entry in `conf.py`
- [ ] Re-enable `fail_on_warning: true` in `.readthedocs.yml` once the warning count reaches zero
