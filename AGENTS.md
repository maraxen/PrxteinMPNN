# AGENTS.md

> Agent routing document. This project is managed by the **Orbital Velocity Orchestrator**.

---

## 🔗 Orchestration Model

**This project does NOT self-orchestrate.** Work is dispatched by the global orchestrator at `~/Projects`.

### Subagent Protocol

1. **Claim dispatch**: `dispatch(action: "claim", payload: { target_prefix: "antigravity" })`
2. **Send heartbeats** every 2-3 min: `dispatch(action: "heartbeat", payload: { dispatch_id: "..." })`
3. **Complete**: `dispatch(action: "complete", payload: { dispatch_id: "...", status: "completed", result: "..." })`

For orchestration details, see `~/Projects/AGENTS.md`.

---

## 🎭 Agent Modes

Load the mode from your dispatch: `.agent/agents/{mode}.md`

| Mode | Purpose |
|:-----|:--------|
| fixer | Implementation |
| recon | Reconnaissance |
| explorer | Codebase search |
| oracle | Architecture |
| flash | Fast execution |

---

## Protocols

**Verification Visibility Protocol:** To ensure transparency and prevent hallucination, the agent must redirect output from all critical verification steps (compilation, tests, benchmarks) to a log file. This file must be dynamically filtered (e.g., `tail`, `grep`) to capture essential success/failure evidence without spamming context. The filtered artifact must be committed to the repository.

---

## Commands

| Action | Command |
| :--- | :--- |
| **Type Check** | `uv run ty check` |
| **Lint** | `uv run ruff check .` |
| **Format** | `uv run ruff format .` |
| **Tests** | `uv run pytest` |
| **Remote GPU** | `just sync && just remote-run <script>` |

---

## Tech Stack

- **Language**: Python 3.12+
- **ML Framework**: JAX + Equinox
- **Package Manager**: uv
- **Type Checking**: ty (strict)
- **Linting**: ruff
- **Testing**: pytest

---

## Directory Structure

```
united_workspace/
├── PrxteinMPNN/        # Protein design with LigandMPNN in JAX
├── prolix/             # JAX-MD molecular dynamics wrapper
├── proteinsmc/         # Sequential Monte Carlo sampling
├── proxide/            # Structure utilities  
└── trex/               # Phylogenetic inference

Each project follows:
├── src/{pkg}/          # Source code
├── tests/              # pytest tests
├── pyproject.toml      # Project config
└── AGENTS.md           # Project-specific routing
```

---

## Code Style

- **Python**: Strict typing with `ty`, format with `ruff`
- **JAX**: Use `jax.jit`, `jax.vmap`, `jax.lax.scan` patterns
- **Equinox**: Modules as dataclasses, `eqx.filter_jit` for PyTrees
- **Testing**: Numerical tolerance tests, cross-framework validation
