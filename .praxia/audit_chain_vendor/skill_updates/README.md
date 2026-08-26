# Staged jax-deep-audit skill update (260826_chain-selection-vendor-superset-audit)

This directory is a staging area, NOT the live skill. The live skill lives at
`/home/marielle/projects/aminx/.claude/skills/jax-deep-audit/`, which is untracked (outside
git) and outside this worktree's write access — a worktree-isolated session cannot write
there. To apply this update:

```bash
cp .praxia/audit_chain_vendor/skill_updates/jax-deep-audit/checkers/u5_checker.py \
   /home/marielle/projects/aminx/.claude/skills/jax-deep-audit/checkers/u5_checker.py
cp .praxia/audit_chain_vendor/skill_updates/jax-deep-audit/docs/vendor_superset_protocol.md \
   /home/marielle/projects/aminx/.claude/skills/jax-deep-audit/docs/vendor_superset_protocol.md
cp .praxia/audit_chain_vendor/skill_updates/jax-deep-audit/docs/operational_lessons.md \
   /home/marielle/projects/aminx/.claude/skills/jax-deep-audit/docs/operational_lessons.md
```

Then apply the `SKILL_MD_PATCH.md` edits below by hand to
`/home/marielle/projects/aminx/.claude/skills/jax-deep-audit/SKILL.md` (small, targeted
insertions — not a full rewrite).

## What's new

- **`checkers/u5_checker.py`** — new invariant U5: vendor/reference feature-surface
  superset. AST-extracts a vendor's argparse flags, reflects the target's dataclass fields,
  applies a caller-supplied alias map, and reports PASS/FAIL per mapped concept. Ships with
  the same KP/KN admissibility harness shape as U1-U4.
- **`docs/vendor_superset_protocol.md`** — the U5 protocol writeup: why AST vs reflection,
  why the alias map is never auto-generated, the batch-ergonomics-gap category (a real finding
  this audit surfaced — PASS-but-ergonomically-worse is a third outcome, not binary), and the
  functional-pin-vs-documentation-string distinction for vendor commit pins.
- **`docs/operational_lessons.md`** — resolver fix-strategy discipline (loud refusal over
  guessing correctness-sensitive semantics) and execution-environment lessons (remote-compute
  GPU occupancy checks, git-LFS stub pitfalls, sandbox/worktree interaction signatures).

## SKILL_MD_PATCH.md

See that file for the exact insertions to `SKILL.md`'s "Universal invariants", "Files", and
"What this skill does" sections.
