# prxteinmpnn CLAUDE.md

## Commands

| Action | Command |
| :--- | :--- |
| **Type Check** | `uv run ty check` |
| **Lint** | `uv run ruff check .` |
| **Format** | `uv run ruff format .` |
| **Tests** | `uv run pytest` |
| **JAX advisory** | `uv run jaxlint check src --no-doc` (optional; not a CI gate) |

## Tech Stack

- **Language**: Python 3.12+
- **ML Framework**: JAX + Equinox
- **Package Manager**: uv
- **Type Checking**: ty (strict)
- **Linting**: ruff
- **Testing**: pytest

## Code Style

- Strict typing with `ty`, format with `ruff`
- JAX: use `jax.jit`, `jax.vmap`, `jax.lax.scan` patterns
- Equinox: modules as dataclasses, `eqx.filter_jit` for PyTrees
- Numerical tolerance tests, cross-framework validation
