"""Phase 2 differential probe -- do runner.score() and runner.jacobian() read spec.fixed_mask
at all? grep found zero textual reference to spec.fixed_mask in either function body (unlike
sample(), which turned out to wire it indirectly via _prepare_fixed_controls -- see F_A1, PASS).

This probe does not assume what fixed_mask *should* do for score()/jacobian() (unclear
semantics -- score() evaluates a fixed given sequence, so masking could mean "exclude from the
returned per-position score" or something else entirely). It only checks the weakest possible
invariant: does changing fixed_mask change the output AT ALL. Bit-identical output regardless
of fixed_mask is strong evidence the field is a structurally inert no-op on that surface -- the
same "silently discarded field" shape as F002/F005 in the prior audit.

Run with: uv run python .praxia/audit_chain_vendor/evidence/F_A2_fixed_mask_score_jacobian_probe.py
"""

from __future__ import annotations

import json

import numpy as np

from aminx.host.runner import jacobian, score
from aminx.run.specs import JacobianSpecification, ScoringSpecification

PDB = "tests/data/1ubq.pdb"
N_RESIDUES = 76
# Arbitrary valid 76-residue sequence (not required to be biologically exact -- this probe only
# needs SOME fixed input sequence to check whether fixed_mask changes score()'s output at all).
ARBITRARY_SEQUENCE = "MKIFVKFEDGTTLELEVEPSDTIAKLKEKIQEKTGIPPEEQVLIYKGKVLEDDKTLADYNIKEGDTIELKLKPKGG"


def _arr_summary(x) -> dict:
    a = np.asarray(x)
    return {
        "shape": list(a.shape),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "sum": float(np.sum(a)),
    }


def main() -> None:
    fixed_mask_all = np.ones((N_RESIDUES,), dtype=np.float32)
    fixed_mask_none = np.zeros((N_RESIDUES,), dtype=np.float32)

    report = {"probe": "F_A2_fixed_mask_score_jacobian_probe"}

    # --- score() ---
    score_spec_masked = ScoringSpecification(
        inputs=[PDB], batch_size=1, random_seed=42, model=1, max_length=N_RESIDUES,
        fixed_mask=fixed_mask_all, sequences_to_score=[ARBITRARY_SEQUENCE],
    )
    score_spec_unmasked = ScoringSpecification(
        inputs=[PDB], batch_size=1, random_seed=42, model=1, max_length=N_RESIDUES,
        fixed_mask=fixed_mask_none, sequences_to_score=[ARBITRARY_SEQUENCE],
    )
    result_score_masked = score(score_spec_masked)
    result_score_unmasked = score(score_spec_unmasked)
    score_key = "scores" if "scores" in result_score_masked else next(iter(result_score_masked))
    score_masked_summary = _arr_summary(result_score_masked[score_key])
    score_unmasked_summary = _arr_summary(result_score_unmasked[score_key])
    report["score"] = {
        "output_key_used": score_key,
        "masked_all_fixed": score_masked_summary,
        "masked_none_fixed": score_unmasked_summary,
        "bit_identical": score_masked_summary == score_unmasked_summary,
        "verdict": (
            "FAIL -- fixed_mask has NO measurable effect on score() output (structurally inert)"
            if score_masked_summary == score_unmasked_summary
            else "fixed_mask changes score() output -- effect confirmed, semantics still need characterizing"
        ),
    }

    # --- jacobian() ---
    jac_spec_masked = JacobianSpecification(
        inputs=[PDB], batch_size=1, random_seed=42, model=1, max_length=N_RESIDUES,
        fixed_mask=fixed_mask_all,
    )
    jac_spec_unmasked = JacobianSpecification(
        inputs=[PDB], batch_size=1, random_seed=42, model=1, max_length=N_RESIDUES,
        fixed_mask=fixed_mask_none,
    )
    result_jac_masked = jacobian(jac_spec_masked)
    result_jac_unmasked = jacobian(jac_spec_unmasked)
    jac_key = "categorical_jacobians" if "categorical_jacobians" in result_jac_masked else next(iter(result_jac_masked))
    jac_masked_summary = _arr_summary(result_jac_masked[jac_key])
    jac_unmasked_summary = _arr_summary(result_jac_unmasked[jac_key])
    report["jacobian"] = {
        "output_key_used": jac_key,
        "masked_all_fixed": jac_masked_summary,
        "masked_none_fixed": jac_unmasked_summary,
        "bit_identical": jac_masked_summary == jac_unmasked_summary,
        "verdict": (
            "FAIL -- fixed_mask has NO measurable effect on jacobian() output (structurally inert)"
            if jac_masked_summary == jac_unmasked_summary
            else "fixed_mask changes jacobian() output -- effect confirmed, semantics still need characterizing"
        ),
    }

    print(json.dumps(report, indent=2))
    with open(".praxia/audit_chain_vendor/evidence/F_A2_fixed_mask_score_jacobian_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
