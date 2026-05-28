# Roadmaps Index

Roadmap documents are filed here with `YYMMDD_slug` naming.
Active planning lives in `.praxia/docs/plans/` and `.praxia/docs/specs/`.

---

## Documents

| File | Created | Status | Summary |
|------|---------|--------|---------|
| [260507_refactor-phases-0-6.md](260507_refactor-phases-0-6.md) | 2026-05-07 | **DEPRECATED** (2026-05-08) | Phases 0–6 structural refactor (mpnn.py split, SamplingDriver, io_callback streaming, BatchPlanner). All phases complete. **Do not update.** |
| [260508_active-roadmap.md](260508_active-roadmap.md) | 2026-05-08 | **Active** | Current active work: MODELINPUTS PR-4/5 (model.__call__ boundary + StableHLO), EncoderPreFn/PostFn wiring, multi_state_temperature removal. |

---

## Naming Convention

```
YYMMDD_<slug>.md
```

- `YYMMDD` = date the document was **created** (not last updated)
- `slug` = kebab-case descriptor of the roadmap's scope

## Related Directories

- `.praxia/docs/plans/` — active sprint plans (`YYMMDD_comp-*.md`)
- `.praxia/docs/specs/` — specifications (`YYMMDD_*.md`)
- `.praxia/docs/superpowers/plans/` — superpowers-skill outputs (older, pre-2026-05-25)
- `.agents/verification_logs/` — per-phase/PR filtered verification logs
- `.praxia/REFACTOR_MODELINPUTS.md` — active ModelInputs migration plan (PRs 1–3 done; PR-4/5 pending)
- `.praxia/TECHNICAL_DEBT.md` — open debt items
