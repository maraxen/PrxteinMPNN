---
title: Versioning Strategy
task_id: 260604_release-prep-sprint-compose
date: 260604
status: accepted
---

# Versioning Strategy: 0.1.0a1 → β → Stable

## Decision

The project version is **0.1.0a1** (alpha). The versioning progression and tagging strategy are documented below.

### Alpha Phase: 0.1.0a1

**Current state:** Alpha release, signals honest maturity and ongoing development. Indicates API is not yet stabilized and breaking changes may occur between alpha versions.

### Progression: Alpha → Beta

**Criteria for advancement to 0.1.0b1:**

- README.md is comprehensive and accurate (quick-start, installation, concepts)
- CONTRIBUTING.md is complete (setup, workflow, testing, naming conventions)
- COMPOSITION_GUIDE.md is accurate and all examples run without error
- No known API-breaking issues are open
- CI passes on main
- At least one external user or developer has tested the library with success

**Success signal:** Library is usable and documented well enough for early adopters. APIs are stabilizing.

### Progression: Beta → Stable

**Criteria for advancement to 0.1.0:**

- Two weeks of beta use with no critical bugs (security, data loss, crashes)
- All benchmark examples in docs/benchmarks/ are validated and reproducible
- ReadTheDocs builds cleanly (or fail_on_warning is re-enabled)
- Public API is stable; no breaking changes anticipated for at least one minor version

**Success signal:** Library is production-ready for core use cases. Stability is demonstrated.

## Rationale

**Why alpha signal?**
- 0.1.0a1 is honest about maturity: the API is evolving, dependencies may shift, major refactoring may happen
- Avoids the false promise of 0.1.0 (pseudo-stable)
- Signals to users they should expect turbulence during alpha, reducing frustration
- Standard practice in Python ecosystem (PEP 440) and JAX-adjacent projects

**Why two-phase progression?**
- Alpha→Beta provides a checkpoint: is the API usable? are docs working?
- Beta→Stable provides a checkpoint: has it survived real use? are examples solid?
- Reduces risk of premature stability claims
- Gives external feedback time to surface

**Why version 0.1.0 and not 1.0.0?**
- 0.1.0 signals "not a full release" but "feature-complete"
- Used by Equinox, Jax, and other emerging ML libraries
- Semantic Versioning: 0.y.z for "initial development," any y/z may break

## Tagging Convention

Git tags follow the version:
```
git tag v0.1.0a1
git tag v0.1.0b1
git tag v0.1.0
```

Each tag points to the commit where the criterion was met (e.g., v0.1.0a1 is HEAD at the time this ADR is merged).

For releases beyond 0.1.0, increment patch/minor per semantic versioning.
