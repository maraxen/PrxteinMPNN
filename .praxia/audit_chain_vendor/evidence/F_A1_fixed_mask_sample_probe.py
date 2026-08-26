"""Phase 2 differential probe -- does runner.sample() actually honor spec.fixed_mask?

Axis A top-priority open question from seed_findings.md: fixed_mask is wired into
runner.inspect()'s unconditional_logits branch directly, but into runner.sample() only via
a derived chain_mask that appears gated behind ligand_context. This probe checks empirically:
if residues [0, 1, 2] are marked fixed_mask=1 (fixed), does the sampled output actually keep
those three positions identical to the native sequence, or does sample() ignore fixed_mask
entirely on the pure-protein (no ligand_context) path?

Run with: uv run python .praxia/audit_chain_vendor/evidence/F_A1_fixed_mask_sample_probe.py
"""

from __future__ import annotations

import json

import numpy as np

from aminx.host.runner import sample
from aminx.run.specs import SamplingSpecification
from aminx.utils.aa_convert import MPNN_ALPHABET

PDB = "tests/data/1ubq.pdb"
# Chosen deliberately distinct from 1ubq's native N-terminus (Met-Gln-Ile-Phe-Val-Lys...)
# so a PASS can't be confused with "the model just happened to pick native residues anyway".
FORCED_TOKEN_LETTER = "I"  # Isoleucine
FORCED_TOKEN_ID = MPNN_ALPHABET.index(FORCED_TOKEN_LETTER)


def _native_sequence_from_result(result: dict) -> str:
    seqs = result.get("sequences")
    if seqs is None:
        msg = f"no 'sequences' key in result; keys={list(result.keys())}"
        raise KeyError(msg)
    return seqs[0] if isinstance(seqs, list) else seqs


def main() -> None:
    fixed_positions = [0, 1, 2]
    n_residues = 76  # 1ubq length, matches F004 audit evidence's max_length=76 usage

    fixed_mask = np.zeros((n_residues,), dtype=np.float32)
    fixed_tokens = np.zeros((n_residues,), dtype=np.int32)
    for i in fixed_positions:
        fixed_mask[i] = 1.0
        fixed_tokens[i] = FORCED_TOKEN_ID

    spec_with_fixed = SamplingSpecification(
        inputs=[PDB],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=42,
        model=1,
        max_length=n_residues,
        return_logits=False,
        fixed_mask=fixed_mask,
        fixed_tokens=fixed_tokens,
    )
    spec_without_fixed = SamplingSpecification(
        inputs=[PDB],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=42,
        model=1,
        max_length=n_residues,
        return_logits=False,
    )

    result_fixed = sample(spec_with_fixed)
    result_unfixed = sample(spec_without_fixed)

    seq_fixed = _native_sequence_from_result(result_fixed)
    seq_unfixed = _native_sequence_from_result(result_unfixed)

    fixed_positions_forced_correctly = [seq_fixed[i] == FORCED_TOKEN_LETTER for i in fixed_positions]

    report = {
        "probe": "F_A1_fixed_mask_sample_probe",
        "verdict": "PASS -- sample() honors fixed_mask+fixed_tokens on the pure-protein path"
        if all(fixed_positions_forced_correctly)
        else "FAIL -- fixed_mask/fixed_tokens did not force the requested identity at all fixed positions",
        "forced_token_letter": FORCED_TOKEN_LETTER,
        "fixed_positions": fixed_positions,
        "seq_fixed_run": seq_fixed,
        "seq_unfixed_run": seq_unfixed,
        "fixed_positions_forced_correctly": fixed_positions_forced_correctly,
        "fixed_positions_identical_across_runs (sanity: unconstrained run should usually differ)": (
            [seq_fixed[i] == seq_unfixed[i] for i in fixed_positions] if seq_fixed and seq_unfixed else None
        ),
        "raw_result_fixed_keys": list(result_fixed.keys()) if hasattr(result_fixed, "keys") else str(type(result_fixed)),
    }
    print(json.dumps(report, indent=2))
    with open(".praxia/audit_chain_vendor/evidence/F_A1_fixed_mask_sample_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
