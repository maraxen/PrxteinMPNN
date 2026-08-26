# Patch to apply to `.claude/skills/jax-deep-audit/SKILL.md`

## 1. In "What this skill does", after the U1-U4 bullet list, add:

```
- Optionally: a **vendor superset coverage matrix** (U5) against an external
  reference implementation's declared feature surface.
```

## 2. In "Universal invariants (U1–U4)" — rename section heading and append U5:

Change the heading from `## Universal invariants (U1–U4)` to
`## Universal invariants (U1–U5)`, and add after the U4 bullet:

```
- **U5 vendor/reference feature-surface superset (optional, opt-in).** For a
  target codebase being compared against an external vendor/reference
  implementation, every vendor-exposed concept the auditor maps via a
  caller-supplied alias map either has a present target field, or is
  explicitly recorded absent. Answers "does the concept exist in the target's
  declared surface", NOT "is it wired" (that's still U1, scoped to the target
  alone) -- do not conflate the two axes in one verdict. See
  `docs/vendor_superset_protocol.md`.
```

## 3. In "Files", add after the U4 checker line:

```
- `checkers/u5_checker.py` — vendor/reference feature-surface superset
  (optional, opt-in; requires an external vendor source + alias map).
- `docs/vendor_superset_protocol.md`, `docs/operational_lessons.md`.
```

## 4. Optional: config.example.yaml

If you want U5 wired into the same `config.yaml` pattern as U1-U4, add a block like:

```yaml
vendor_superset_opt_in: false   # if true, provide vendor_source_path + alias_map
vendor_source_path: "/path/to/vendor/cli_entrypoint.py"
alias_map:
  vendor_flag_name:
    target_fields: ["target.field.name"]
    category: "human-readable category"
    notes: "why this mapping, or why target_fields is empty if asserting absence"
```

Not applied here since `config.example.yaml` wasn't touched — leaving that decision to
whoever wires U5 into a live config, since the yaml shape may need to match how U1-U4's own
example is actually structured (this staging pass didn't verify that file's exact schema).
