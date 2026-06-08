---
title: Defer ty check and ruff check from CI gates
status: accepted
date: 260604
task_id: 260604_release-prep-sprint-compose
---

## Decision

Type checking (`ty check`) and linting (`ruff check`) are deferred from CI enforcement until the code surface stabilizes post-release.

## Rationale

At this stage of development, full coverage of type annotations and formatting compliance across the entire codebase is premature. The public API surface is still under active definition, and re-annotating large swaths of code to satisfy strict type checking would create merge conflicts and slow release preparation.

Deferral is justified because:

1. **Code surface stability**: Before enforcing CI-level type and lint compliance, the public API and major module boundaries must be verified against documentation (README, CONTRIBUTING).
2. **Documentation alignment**: Examples in README and CONTRIBUTING must be tested to ensure they compile and execute correctly, which will inform necessary refactoring.
3. **Incremental enforcement**: Enabling these checks post-release allows adoption to happen incrementally across the codebase without blocking critical functionality.

## Consequences

**Quality checks remain available and active locally:**
- Developers run `uv run ty check` and `uv run ruff check` during development
- These commands catch issues early in the local workflow
- No enforcement in CI until re-enabled

**Re-enable criteria:**
- README and CONTRIBUTING sections have been reviewed against actual source code structure
- All code examples in public documentation execute without errors
- Type annotations cover key public APIs (not necessarily 100% coverage)
- Linting standard is aligned with team agreement on formatting style

**Timeline:**
- Deferred from this release
- Target: stable release + 1 sprint after public launch
- Communicated in CONTRIBUTING.md as "short-term deferral, not permanent"

## Alternatives Considered

1. **Full enforcement in CI now**: Rejected due to surface instability; would require extensive pre-release refactoring.
2. **Selective enforcement (types only)**: Rejected; both checks have similar maturity concerns and should be enabled together.
3. **Waive entirely**: Rejected; checks should eventually be enforced to maintain code quality long-term.
