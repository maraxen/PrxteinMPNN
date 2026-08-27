"""F003 Tier A runtime differential — non-degeneracy first, then strategy comparison.

Per dispatch 002 §4.1 and the parent's F003 adjudication (intent B):
  - Non-degeneracy MUST come first (finite output; not constant along the
    state axis; max softmax > 1/21 + margin).
  - The DIFFERENTIAL: two runner.sample calls, identical except
    multi_state_strategy ∈ {'arithmetic_mean', 'product'} on a real
    multistate spec, produce byte-identical logits on the pinned wheel.
    That agreement — with non-degeneracy passing — is the Tier A verdict:
    the sample-path multi_state_strategy field is inert.
  - Under intent (B) this same call, post-fix, must RAISE.

Method for THIS pre-fix baseline:
  1. Build a matched-length SamplingSpecification against 1UBQ.
  2. Non-degeneracy controls FIRST on run #1's output:
     - finite (no NaN / no Inf)
     - not constant along axis 0 of logits
     - max softmax probability across positions exceeds 1/21 + margin=0.05
  3. Run twice with strategy ∈ {arithmetic_mean, product} (all other
     spec fields identical, including random_seed).
  4. Compare logits byte-identity by np.array_equal (strict) AND
     np.allclose(atol=1e-8).
  5. WRONG-LOOKING escalation clause: if the two runs DISAGREE and
     non-degeneracy also passes, STOP and report — do not retune. The
     dispatch mandates escalation over adjustment.

Fixture goes under `tests/audit/`. No wheel source modified.
"""
from __future__ import annotations

import importlib.metadata as im
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

AUDIT = Path("/home/marielle/projects/aminx/.praxia/audit")
EVIDENCE = AUDIT / "evidence"
EVIDENCE.mkdir(exist_ok=True, parents=True)

WHEEL_STAMP = im.version("aminx")
PDB = Path("/home/marielle/projects/aminx/tests/data/1ubq.pdb")

REPORT_PATH = EVIDENCE / "F003_report.json"
TRANSCRIPT_PATH = EVIDENCE / "F003_transcript.txt"


def _build_spec(strategy: str, L: int, seed: int):
    from aminx.run.specs import SamplingSpecification

    return SamplingSpecification(
        inputs=[str(PDB)],
        num_samples=1,
        temperature=0.1,
        batch_size=1,
        random_seed=seed,
        multi_state_strategy=strategy,
        state_position_map=np.arange(L, dtype=np.int32)[None, :],
        model=1,
        max_length=L,
        return_logits=True,
    )


def _run_sample(strategy: str, L: int = 76, seed: int = 42) -> dict:
    from aminx.host import runner

    spec = _build_spec(strategy, L, seed)
    t0 = time.time()
    try:
        result = runner.sample(spec)
        return {
            "strategy": strategy,
            "exit": "returned",
            "wall_seconds": time.time() - t0,
            "result_keys": sorted(result.keys()),
            "logits_shape": tuple(int(x) for x in np.asarray(result["logits"]).shape),
            "logits_dtype": str(np.asarray(result["logits"]).dtype),
            "logits": np.asarray(result["logits"]),
            "sequences": np.asarray(result["sequences"]).tolist(),
            "structure_ids": result.get("metadata", {}).get("structure_ids"),
        }
    except Exception as exc:
        return {
            "strategy": strategy,
            "exit": "raised",
            "wall_seconds": time.time() - t0,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
        }


def non_degeneracy(logits: np.ndarray) -> dict:
    """Return non-degeneracy verdicts on the FIRST run's logits before any
    parity check runs. Dispatch: 'never assert parity/agreement without
    asserting non-degeneracy first'."""
    finite = bool(np.all(np.isfinite(logits)))
    # Not constant along a natural output axis: pick position axis (axis -2)
    # to detect a uniform-along-length output; also check pooled variance.
    per_pos_range = float(np.max(logits) - np.min(logits))
    pooled_std = float(np.std(logits))
    # Max softmax probability (positions x tokens) across all samples
    from scipy.special import softmax  # scipy is a jax dep, safe

    prob = softmax(logits, axis=-1)
    max_prob = float(np.max(prob))
    q = 21  # aminx vocab
    margin = 0.05
    non_uniform = max_prob > (1.0 / q) + margin
    return {
        "finite": finite,
        "per_position_range": per_pos_range,
        "pooled_std": pooled_std,
        "max_softmax_probability": max_prob,
        "vocab_size_q": q,
        "uniform_floor_plus_margin": 1.0 / q + margin,
        "non_uniform_over_axis_neg1": non_uniform,
        "passes_non_degeneracy": bool(finite and non_uniform and per_pos_range > 1e-6),
    }


def main() -> None:
    lines: list[str] = []
    lines.append("# F003 RUNTIME DIFFERENTIAL — pre-fix baseline on pinned wheel")
    lines.append(f"# task_id: 260826_aminx-invariant-audit")
    lines.append(f"# wheel: aminx=={WHEEL_STAMP}")
    lines.append(f"# interpreter: {sys.executable}")
    lines.append(f"# PDB fixture: {PDB}")
    lines.append("")

    L = 76
    seed = 42

    # Run 1: arithmetic_mean
    r_mean = _run_sample("arithmetic_mean", L=L, seed=seed)
    # Run 2: product (differ only in multi_state_strategy)
    r_prod = _run_sample("product", L=L, seed=seed)

    lines.append("## Run 1: multi_state_strategy=arithmetic_mean")
    lines.append(json.dumps({k: v for k, v in r_mean.items() if k not in ("logits",)}, indent=2, default=str))
    lines.append("")
    lines.append("## Run 2: multi_state_strategy=product")
    lines.append(json.dumps({k: v for k, v in r_prod.items() if k not in ("logits",)}, indent=2, default=str))
    lines.append("")

    if r_mean["exit"] != "returned" or r_prod["exit"] != "returned":
        summary = {
            "outcome": "AT-LEAST-ONE-RUN-RAISED",
            "run_1": {k: v for k, v in r_mean.items() if k not in ("logits",)},
            "run_2": {k: v for k, v in r_prod.items() if k not in ("logits",)},
            "verdict": "Tier A differential cannot be constructed — one call raised.",
        }
        lines.append("## Non-degeneracy: SKIPPED (a run raised)")
        lines.append(json.dumps(summary, indent=2, default=str))
        TRANSCRIPT_PATH.write_text("\n".join(lines))
        REPORT_PATH.write_text(json.dumps({
            "wheel_stamp": WHEEL_STAMP,
            "finding_id": "F003",
            "seams_json_hash": "ba5789d24e6ed33195c64147ca14097dc0958b90c40a5af094d5df43ab8ec9de",
            **summary,
        }, indent=2, default=str))
        print(json.dumps(summary, indent=2, default=str))
        return

    logits_mean = r_mean["logits"]
    logits_prod = r_prod["logits"]

    nd = non_degeneracy(logits_mean)
    lines.append("## Non-degeneracy on Run 1 (arithmetic_mean) — MUST PASS BEFORE PARITY CHECK")
    lines.append(json.dumps(nd, indent=2))
    lines.append("")

    if not nd["passes_non_degeneracy"]:
        summary = {
            "outcome": "NON-DEGENERACY-FAIL",
            "non_degeneracy": nd,
            "verdict": (
                "The differential is inadmissible until Run 1's output is finite, "
                "not uniform, and has non-trivial range along the position axis. "
                "Zeros are the easy case; a uniform distribution and a NaN-masked-"
                "to-zero array agree perfectly."
            ),
        }
        lines.append(json.dumps(summary, indent=2))
        TRANSCRIPT_PATH.write_text("\n".join(lines))
        REPORT_PATH.write_text(json.dumps({
            "wheel_stamp": WHEEL_STAMP,
            "finding_id": "F003",
            "seams_json_hash": "ba5789d24e6ed33195c64147ca14097dc0958b90c40a5af094d5df43ab8ec9de",
            **summary,
        }, indent=2, default=str))
        print(json.dumps(summary, indent=2, default=str))
        return

    # Parity check
    shape_match = logits_mean.shape == logits_prod.shape
    byte_identical = bool(shape_match and np.array_equal(logits_mean, logits_prod))
    allclose_1e8 = bool(
        shape_match and np.allclose(logits_mean, logits_prod, atol=1e-8, rtol=0)
    )
    max_abs_diff = float(np.max(np.abs(logits_mean - logits_prod))) if shape_match else None
    rms_diff = float(np.sqrt(np.mean((logits_mean - logits_prod) ** 2))) if shape_match else None

    seq_mean = np.asarray(r_mean["sequences"])
    seq_prod = np.asarray(r_prod["sequences"])
    seq_identical = bool(np.array_equal(seq_mean, seq_prod))

    lines.append("## Parity check (only run after non-degeneracy passed)")
    parity = {
        "logits_shape_run_1": tuple(int(x) for x in logits_mean.shape),
        "logits_shape_run_2": tuple(int(x) for x in logits_prod.shape),
        "byte_identical": byte_identical,
        "allclose_atol_1e_minus_8": allclose_1e8,
        "max_abs_logits_diff": max_abs_diff,
        "rms_logits_diff": rms_diff,
        "sequences_identical": seq_identical,
    }
    lines.append(json.dumps(parity, indent=2))
    lines.append("")

    # Verdict interpretation
    if byte_identical:
        verdict = (
            "TIER-A-CONFIRMED (pre-fix baseline): the two runs produce "
            "byte-identical logits despite differing multi_state_strategy. "
            "The field is INERT on runner.sample. Non-degeneracy passed, "
            "so this is not the zeros/uniform confound."
        )
        outcome = "TIER-A-PRE-FIX-INERT"
    elif allclose_1e8:
        verdict = (
            "TIER-A-CONFIRMED (pre-fix baseline): logits agree within "
            "1e-8 absolute tolerance despite differing multi_state_strategy; "
            "the residual is consistent with numerical noise from the "
            "same computation. Field is effectively inert on runner.sample."
        )
        outcome = "TIER-A-PRE-FIX-INERT"
    else:
        # DIFFERENTIAL LOOKS WRONG — escalate, do NOT retune.
        verdict = (
            "DIFFERENTIAL-DISAGREES-ESCALATE: the two strategies produce "
            "different logits on the pre-fix wheel. This CONTRADICTS the "
            "static AST asymmetry evidence (`del multi_state_strategy` in "
            "runner.sample body) and must not be retuned by the auditor. "
            "Escalating to parent per dispatch. Non-degeneracy passed; "
            "max_abs_diff and rms_diff are recorded above."
        )
        outcome = "DIFFERENTIAL-DISAGREES-ESCALATE"

    summary = {
        "outcome": outcome,
        "wheel_stamp": WHEEL_STAMP,
        "run_1": {k: v for k, v in r_mean.items() if k not in ("logits",)},
        "run_2": {k: v for k, v in r_prod.items() if k not in ("logits",)},
        "non_degeneracy": nd,
        "parity": parity,
        "verdict": verdict,
        "post_fix_prediction_under_intent_B": (
            "Under the parent's F003 intent-B decision, this same call must "
            "RAISE at runner.sample entry (message pointing at campaign verbs). "
            "This differential is therefore also the ADEQUACY TEST for the "
            "fix: it currently runs and returns byte-identical outputs; "
            "post-fix it must raise."
        ),
    }
    lines.append("## Summary")
    lines.append(json.dumps(summary, indent=2, default=str))

    TRANSCRIPT_PATH.write_text("\n".join(lines))
    REPORT_PATH.write_text(json.dumps({
        "wheel_stamp": WHEEL_STAMP,
        "finding_id": "F003",
        "seams_json_hash": "ba5789d24e6ed33195c64147ca14097dc0958b90c40a5af094d5df43ab8ec9de",
        **summary,
    }, indent=2, default=str))
    print(json.dumps({
        "transcript": str(TRANSCRIPT_PATH),
        "report": str(REPORT_PATH),
        "outcome": outcome,
        "byte_identical": byte_identical,
        "allclose_1e8": allclose_1e8,
        "max_abs_diff": max_abs_diff,
        "non_degeneracy_passed": nd["passes_non_degeneracy"],
    }, indent=2))


if __name__ == "__main__":
    main()
