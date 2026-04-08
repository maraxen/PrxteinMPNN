---
description: Critique any artifact (plan, query set, research output, architecture decision) as the Oracle role and produce a structured oracle_critique.json
---

# Oracle Critique Workflow

Adopt the **Oracle** role (`agent_assets/roles/oracle.md`) and produce a structured critique conforming to `agent_assets/schemas/oracle_critique.json`.

## When to Use

- Reviewing an implementation plan before dispatch
- Critiquing NotebookLM query sets before execution
- Evaluating research synthesis for gaps, risks, and blind spots
- Assessing architecture decisions or design proposals
- Validating any artifact that would benefit from a skeptical, evidence-grounded second opinion

## Inputs

The user will provide one or more of:

| Input | Example |
|-------|---------|
| **Inline artifact** | A plan, query set, or design pasted in chat |
| **File reference** | `@[path/to/plan.md]` |
| **Staging/backlog item** | "Critique backlog #2114" |
| **NLM query results** | Output from a prior `notebook_query` |

If the input type is unclear, ask: _"What artifact should I critique?"_

## Phase 1: Load Oracle Context

1. **Adopt the Oracle role.** Internalize the advisory philosophy, process, and guiding principles from `agent_assets/roles/oracle.md`:
   - Think in trade-offs — every decision has costs
   - Be direct — state assessments clearly, don't hedge
   - Ground in evidence — point to specific code, docs, prior art
   - Advisory only — identify issues, don't fix them

2. **Load the critique schema.** The output must conform to `agent_assets/schemas/oracle_critique.json`:
   ```
   verdict:              APPROVE | REVISE | REJECT
   confidence:           high | medium | low
   strategic_assessment:  1-3 sentence summary of overall quality
   concerns[]:           { area, severity, issue, recommendation }
   approved_for_execution: boolean
   ```

## Phase 2: Deep Read

3. **Read the artifact thoroughly.** Don't skim — the Oracle role demands evidence-grounded analysis.
4. **Gather supporting context.** Depending on the artifact type:
   - **Plans**: Read the target codebase areas, check for existing patterns, verify assumptions
   - **Query sets**: Check the target notebook sources, look for redundancy and blind spots
   - **Research**: Cross-reference claims against code reality
   - **Architecture**: Review current system design, identify constraint violations
   - **Workflows**: Compare structure against peer workflows in `.agent/workflows/`, verify frontmatter format, check that steps reference real tools/schemas/roles

## Phase 3: Multi-Axis Analysis

Evaluate the artifact along these axes (not all apply to every artifact):

| Axis | Questions |
|------|-----------|
| **Correctness** | Are factual claims accurate? Do code references exist? Are assumptions valid? |
| **Completeness** | What's missing? Are there blind spots? Edge cases not addressed? |
| **Feasibility** | Can this actually be implemented? Are dependencies realistic? Are effort estimates reasonable? |
| **Risk** | What could go wrong? What's the blast radius of failure? Are rollback paths defined? |
| **Alignment** | Does this align with project architecture, conventions, and strategic direction? |
| **Redundancy** | Does this duplicate existing work or overlap with other initiatives? |
| **Specificity** | Is the artifact concrete enough to act on, or is it hand-wavy? |
| **Consistency** | Does the artifact contradict itself? Are terms used consistently? Do phases reference each other correctly? |

## Phase 4: Generate Output & Persist

5. **Write the structured critique.** Follow the schema exactly:

```json
{
  "verdict": "APPROVE | REVISE | REJECT",
  "confidence": "high | medium | low",
  "strategic_assessment": "Concise overall assessment — what's strong, what's weak, what's the bottom line.",
  "concerns": [
    {
      "area": "Name of the concern area (e.g., 'Error Handling', 'Query Coverage', 'Phase 3 Dependencies')",
      "severity": "critical | warning | suggestion",
      "issue": "Specific description of what's wrong or missing. Point to evidence.",
      "recommendation": "Concrete, actionable fix. Not 'consider doing X' — say 'do X because Y'."
    }
  ],
  "approved_for_execution": true | false
}
```

6. **Create a human-readable companion.** Always draft a markdown version of the critique encompassing the strategic assessment, the itemized concerns (in plain language), and the verdict rationale.

7. **Persist the artifacts.**
   - Prefer submitting both the JSON and the Markdown version via the OV MCP `artifact(action: "submit")` tool.
   - **Error handling:** If the MCP submission returns an error, read the error message carefully and amend the payload to fix the issue (e.g., add a missing `id` field, correct a field type). The `artifact(action: "submit")` payload requires at minimum: `id` (string, unique identifier), `artifact_type` (string), and `content` (string). Retry after amending.
   - **Fallback mechanism (last resort only):** Save to `eph/oracle_critique.json` and `eph/oracle_critique.md` ONLY if the MCP server is critically unavailable (connection refused, timeout) or the error is not obviously fixable from the error message. Do NOT fall back to `eph/` for simple payload validation errors — fix and retry instead.

### Verdict Guidelines (and Execution Mapping)

| Verdict | Execution Setting | When to Use |
|---------|-------------------|-------------|
| **APPROVE** | `approved_for_execution: true` | Artifact is sound. Any concerns are suggestions/minor. Safe to proceed. |
| **REVISE** | `approved_for_execution: false` | Artifact has merit but has warnings/gaps that should be addressed before execution. |
| **REJECT** | `approved_for_execution: false` | Fundamental flaws — incorrect assumptions, missing critical elements, or misaligned with goals. Needs rework. |

### Severity Guidelines

| Severity | Meaning |
|----------|---------|
| **critical** | Must fix before proceeding. Blocks execution or risks correctness. |
| **warning** | Should fix. Creates risk or tech debt if ignored. |
| **suggestion** | Nice to have. Improves quality but not blocking. |

## Phase 5: Present and Iterate

8. **Present the critique** with the JSON output and a brief narrative summary.
9. **If verdict is REVISE**: Offer to help refine the artifact based on the concerns raised.
10. **If verdict is REJECT**: Explain the fundamental issues clearly and suggest a path to a viable version.
11. **If verdict is APPROVE**: Confirm the artifact is ready and suggest next steps (dispatch, execution, etc.).

## Tips

- **Be the skeptic.** The Oracle's value is in finding what others miss. Don't rubber-stamp.
- **Severity matters.** Don't mark everything as critical — a wall of criticals loses signal. Reserve critical for genuine blockers.
- **Empty concerns array is valid.** If the artifact is genuinely solid, say so with high confidence.
- **Context is king.** A plan that looks perfect in isolation might conflict with in-progress work. Check active dispatches and backlog.
- **One critique per artifact.** If the user provides multiple items, produce a separate critique for each.
