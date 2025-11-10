# Prompt for Next Session

## Quick Start Prompt

```
You are continuing work on validating PrxteinMPNN against ColabDesign. The previous session improved correlation from 0.210 → 0.681, but the target is 0.871 (baseline) and ultimately >0.90.

IMPORTANT: First, read the complete handoff document at HANDOFF_VALIDATION_DEBUG.md for full context.

CURRENT STATE:
- Branch: claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS (commit d162f06)
- Current correlation: 0.681
- Target correlation: 0.871 (baseline at commit acae2d6)
- Ultimate goal: >0.90

KEY CONTEXT:
- Architecture was fixed (w_e_proj restored to ProteinFeatures)
- Alphabet conversion added (AF→MPNN order) - critical for correct comparison
- Debug prints exist in both implementations (may affect results)
- ColabDesign installed at /tmp/ColabDesign with modifications

IMMEDIATE SETUP STEPS:
1. Verify you're on the correct branch: git checkout claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
2. Check if ColabDesign exists at /tmp/ColabDesign (contains debug prints)
3. If not, clone it: cd /tmp && git clone https://github.com/sokrypton/ColabDesign.git
4. Install ColabDesign: cd /tmp/ColabDesign && uv pip install -e . --system
5. Return to PrxteinMPNN: cd /home/user/PrxteinMPNN
6. Reinstall PrxteinMPNN: uv pip install -e . --system
7. Run baseline test: uv run python test_fix_correlation.py

YOUR FIRST TASK:
Investigate why correlation is 0.681 instead of 0.871. The most likely causes are:
1. Debug prints affecting computation (test by removing them)
2. Subtle weight loading or initialization differences
3. Something changed between acae2d6 and current that we haven't identified

DEBUGGING APPROACH:
1. Remove all jax.debug.print() calls from:
   - src/prxteinmpnn/model/features.py
   - src/prxteinmpnn/model/mpnn.py
   - /tmp/ColabDesign/colabdesign/mpnn/modules.py
   - /tmp/ColabDesign/colabdesign/mpnn/score.py
2. Reinstall both packages
3. Test if correlation improves
4. If not, compare current code line-by-line with commit acae2d6
5. Check the original test file (test_final.py) at acae2d6 vs current test

CONSTRAINTS:
- Do NOT change atom ordering (keep PDB order with O before CB)
- Do NOT move w_e_proj out of ProteinFeatures
- Do NOT remove alphabet conversion
- ALWAYS use PRNG key = 42 for consistency

SUCCESS CRITERIA:
- Achieve ≥0.871 correlation (matches baseline)
- Understand root cause of 0.681 → 0.871 gap
- Document findings clearly

Read HANDOFF_VALIDATION_DEBUG.md for complete details, code snippets, and historical context.
```

---

## Detailed Session Continuation Prompt

For a more thorough start, use this expanded prompt:

```
# Context
You are an AI assistant helping to debug and validate the PrxteinMPNN implementation against ColabDesign (the reference implementation). This is a continuation of previous work.

# Background
PrxteinMPNN is a JAX/Equinox reimplementation of ProteinMPNN. We're validating it produces the same outputs as ColabDesign's implementation. The previous session made significant progress but didn't reach the baseline correlation.

# Current State
- Working on branch: claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
- Latest commit: d162f06
- Current correlation: 0.681
- Baseline correlation (commit acae2d6): 0.871
- Ultimate goal: >0.90 correlation

# What Was Fixed
The previous session:
1. Reverted incorrect architectural changes that removed w_e_proj from ProteinFeatures
2. Added critical alphabet conversion (AlphaFold order → MPNN order)
3. Fixed weight loading paths
4. Improved correlation from 0.210 → 0.681 (3.2x improvement)

# What Still Needs Work
There's a gap from 0.681 → 0.871 that needs investigation. The architecture appears correct, alphabet conversion is implemented, but something is still different from the baseline.

# Your Mission
1. **Immediate**: Read HANDOFF_VALIDATION_DEBUG.md completely
2. **Setup**: Ensure environment is correct (both packages installed, ColabDesign at /tmp/ColabDesign)
3. **Test**: Run test_fix_correlation.py to confirm 0.681 baseline
4. **Debug**: Investigate the 0.681 → 0.871 gap
5. **Fix**: Identify and fix the remaining issue(s)
6. **Validate**: Achieve ≥0.871 correlation

# First Steps (in order)
1. Fetch and checkout the branch:
   ```bash
   cd /home/user/PrxteinMPNN
   git fetch origin claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
   git checkout claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
   ```

2. Check for ColabDesign installation:
   ```bash
   ls /tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl
   ```

   If it doesn't exist:
   ```bash
   cd /tmp
   git clone https://github.com/sokrypton/ColabDesign.git
   cd ColabDesign
   uv pip install -e . --system
   ```

3. Install PrxteinMPNN:
   ```bash
   cd /home/user/PrxteinMPNN
   uv pip install -e . --system
   ```

4. Run the validation test:
   ```bash
   uv run python test_fix_correlation.py
   ```

   Expected output: Correlation: 0.681324

5. Read the handoff document:
   ```bash
   cat HANDOFF_VALIDATION_DEBUG.md
   ```

# Investigation Plan
The handoff document suggests these likely causes:
1. Debug prints affecting computation
2. Subtle initialization differences (PRNG key splitting)
3. Weight loading issues we haven't caught
4. Differences in test methodology

Start by removing all debug prints and retesting.

# Critical Knowledge
- **Alphabet conversion is REQUIRED**: ColabDesign outputs in AF order, PrxteinMPNN expects MPNN order
- **Atoms are intentionally in "wrong" order**: PDB order (O before CB) not atom37, baseline was trained this way
- **w_e_proj belongs in ProteinFeatures**: NOT in main model
- **Always use seed 42**: For reproducibility

# Resources
- Handoff document: HANDOFF_VALIDATION_DEBUG.md (complete context)
- Current test: test_fix_correlation.py
- Baseline test: git show acae2d6:test_final.py
- Baseline commit: acae2d6 (0.871 correlation)
- Bad commit: d66c91e (0.210 correlation - incorrect architecture)
- Current commit: d162f06 (0.681 correlation - architecture fixed)

# Success Metrics
- Primary: Correlation ≥ 0.871 (matches baseline)
- Stretch: Correlation > 0.90 (original goal)
- Complete: All three decoding paths validated (unconditional, conditional, autoregressive)

# Constraints & Warnings
- Do NOT change atom ordering (keep PDB order)
- Do NOT move w_e_proj from ProteinFeatures
- Do NOT remove alphabet conversion
- DO reinstall packages after code changes (uv pip install -e . --system)
- DO use PRNG key = 42 consistently
- DO read the handoff document before making changes

Begin by reading HANDOFF_VALIDATION_DEBUG.md, then proceed with the investigation plan.
```

---

## Minimal Quick-Start Prompt

For fastest startup (assumes some context):

```
Continue validation debugging for PrxteinMPNN. Current state: 0.681 correlation (target 0.871).

Setup:
git checkout claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
# Ensure ColabDesign at /tmp/ColabDesign
uv pip install -e . --system
uv run python test_fix_correlation.py

Read HANDOFF_VALIDATION_DEBUG.md for full context.

First task: Remove all jax.debug.print() calls and test if correlation improves.

Key facts:
- Architecture is correct (w_e_proj in ProteinFeatures)
- Alphabet conversion implemented (AF→MPNN)
- Atoms in PDB order (intentional)
- Gap: 0.681 → 0.871 to investigate
```

---

## Template for Specific Investigations

Use this template to focus on specific issues:

```
[SPECIFIC INVESTIGATION: Debug Prints Impact]

Context: PrxteinMPNN validation at 0.681 correlation, target is 0.871.
Branch: claude/validate-all-decoding-paths-011CUvaUYpgAE2LGtx2ThFgS
Hypothesis: Debug prints may be affecting numerical computation.

Task: Remove all debug prints and test impact on correlation.

Files to modify:
1. src/prxteinmpnn/model/features.py (lines 162-178)
2. src/prxteinmpnn/model/mpnn.py (lines 176-178)
3. /tmp/ColabDesign/colabdesign/mpnn/modules.py
4. /tmp/ColabDesign/colabdesign/mpnn/score.py

After removing prints:
1. Reinstall both packages
2. Run test_fix_correlation.py
3. Compare correlation before/after
4. Document findings

See HANDOFF_VALIDATION_DEBUG.md section "Next Steps" for details.
```

---

## Best Practices for Handoff

When starting the next session:

1. **Always read HANDOFF_VALIDATION_DEBUG.md first**
2. **Verify environment setup before coding**
3. **Test baseline before making changes**
4. **Keep notes of what you try**
5. **Commit working changes incrementally**
6. **Update handoff document with findings**

---

## Emergency Recovery

If something goes wrong:

```bash
# Reset to last known good state
cd /home/user/PrxteinMPNN
git checkout d162f06
uv pip install -e . --system

# Restore ColabDesign if needed
cd /tmp
rm -rf ColabDesign
git clone https://github.com/sokrypton/ColabDesign.git
cd ColabDesign
uv pip install -e . --system

# Test baseline
cd /home/user/PrxteinMPNN
uv run python test_fix_correlation.py
# Should show: Correlation: 0.681324
```

---

## Questions to Guide Investigation

1. What exactly is different between acae2d6 (0.871) and d162f06 (0.681)?
2. Do debug prints affect JIT compilation or numerical precision?
3. Are we using the exact same test methodology as the baseline?
4. Is there a subtle weight loading issue we missed?
5. Could PRNG key splitting be different?
6. Are all encoder/decoder layers loading correctly?

Answer these systematically and document findings in the handoff document.
