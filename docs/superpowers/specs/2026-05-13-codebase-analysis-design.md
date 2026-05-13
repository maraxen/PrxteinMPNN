# Codebase Analysis & Visualization Design
**Date**: 2026-05-13  
**Scope**: Full `src/prxteinmpnn/` directory analysis focused on `model/` refactoring opportunities

---

## Goal
Build a high-quality structural model of the codebase using praxia's CodeGraph, generate Mermaid diagrams, and identify refactoring opportunities (naming issues, redundancy, dead code, architectural gaps) in the `model/` directory and its dependencies.

---

## Approach

### Phase 1: CodeGraph Indexing
- Use `mcp__praxia__code_index_workspace` to build a complete AST-based CodeGraph of `src/prxteinmpnn/`
- Captures all files, imports, class definitions, function signatures, and call relationships
- Output: Indexed graph with node/edge counts and symbol relationships

### Phase 2: Graph Query & Extraction
Query the CodeGraph to extract:
- **Module structure**: All `.py` files, their exports, and relationships
- **Import graph**: Which modules import from which (identify circular deps, tight coupling)
- **Class hierarchy**: All class definitions, inheritance chains, interfaces
- **Function signatures**: Public method signatures, parameters, return types
- **Call patterns**: Key execution paths (forward pass, sampling, scoring)
- **Code metrics**: File sizes, complexity indicators, unused exports

### Phase 3: Visualization
Generate Mermaid diagrams:
1. **High-level module map** — Shows `src/prxteinmpnn/` packages and their dependencies
2. **Model directory detail** — File organization, classes, and exports in `model/`
3. **Import dependencies** — Detailed graph of which files import which (highlights cross-module patterns)
4. **Call flow diagrams** — Key execution paths (e.g., forward pass: input → encoder → MPNN core → decoder)
5. **Class hierarchy** — Type definitions and inheritance relationships

### Phase 4: Analysis & Findings
Synthesize CodeGraph data into structured findings:
- **File inventory**: All files, categorization by responsibility
- **Redundancy detection**: Similar implementations, duplicate logic
- **Naming inconsistencies**: Files or classes that violate naming conventions or confuse intent (e.g., `features.py` vs `features_direct.py`, `mpnn_autoregressive_state_vmap_exact.py` variants)
- **Dead code**: Unused exports, unreachable functions
- **Architecture health**: Circular imports, overly tight coupling, missing abstractions
- **Integration points**: How `model/` depends on and is used by the rest of codebase

### Phase 5: Recommendations
Generate actionable recommendations:
- **Quick wins**: Low-risk renames, simple consolidations (e.g., clarify file names)
- **Medium effort**: Merge similar classes, deduplicate implementations
- **Larger refactors**: Module reorganization, interface redesign
- Respect naming discipline: `SamplingInputs` for public API, no `Payload` suffix in method names

---

## Deliverables

### 1. `docs/codebase_analysis/CODEBASE_MAP.md`
- Contains all Mermaid diagrams (module structure, class hierarchy, import graph, call flows)
- Readable reference for understanding codebase structure
- Formatted for easy navigation (sections per diagram type)

### 2. `docs/codebase_analysis/ANALYSIS.md`
- Detailed findings from CodeGraph query
- File inventory with categorization
- Import relationship analysis
- Class structure summary
- Call pattern mapping

### 3. `docs/codebase_analysis/RECOMMENDATIONS.md`
- Findings organized by priority (quick wins → larger refactors)
- Concrete examples with current vs. recommended changes
- Risk assessment and effort estimates
- Respect for existing naming discipline (`SamplingInputs`, no `Payload`)

---

## Success Criteria
- [ ] CodeGraph successfully indexes `src/prxteinmpnn/` with all files and relationships
- [ ] Mermaid diagrams are accurate, readable, and show key architectural patterns
- [ ] Analysis identifies at least 3 concrete refactoring opportunities
- [ ] Recommendations are actionable and include effort estimates
- [ ] Diagrams and analysis serve as reusable documentation for future developers

---

## Constraints & Assumptions
- CodeGraph is the authoritative source for structure (not git history or memory)
- Public API naming follows established discipline (SamplingInputs, no Payload suffix)
- Analysis respects existing code organization decisions
- Diagrams should prioritize clarity over completeness (group dense subgraphs if needed)

---

## Next Steps
1. ✅ Design approved
2. Invoke `writing-plans` to create implementation plan
3. Execute: index, query, visualize, analyze, recommend
4. Review deliverables and commit to `docs/codebase_analysis/`
