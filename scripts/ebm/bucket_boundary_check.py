"""Mandatory xtrax HiTL gate for residue-axis bucket boundaries (backlog node E4.5).

The ProteinEBM epic's residue-axis bucket boundaries `(64, 128, 256, 512)` were
already reviewed and confirmed by the human user during the design-review gate
(EPIC doc `.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md` §1 Fork 6).
This script does **not** re-litigate those numbers. It performs the remaining
*mechanical* half of the `using-xtrax` `bucket_boundaries` HiTL stop: running
xtrax's own EDA tooling against a length distribution to confirm the
compile-count-vs-padding-waste tradeoff is reasonable, and printing the result
so it is a matter of record rather than a silently-skipped check.

Proxy distribution (READ THIS BEFORE TRUSTING THE NUMBERS BELOW)
------------------------------------------------------------------
The real ProteinEBM training corpora (32k CATH domains + 590k AFDB domains +
18k complexes) are **not** downloaded in this environment -- only the
data-prep scripts exist (`~/repos/ProteinEBM/protein_ebm/data/data_scripts/`).
Downloading the real corpus (AFDB alone is enormous) was out of scope and not
authorized for validating four bucket boundaries. This script therefore uses a
**defensible proxy**, built from two transparent sources, and is NOT a
substitute for re-running this exact check against real corpus statistics once
they are available (tracked as a known limitation -- see module-level PASS
report below and the E4.5 task's final writeup):

  1. Real local PDB/mmCIF structures already checked into the repo
     (`tests/data/*.pdb`, `*.cif`), parsed via `aminx.io.parsing.parse_structure`
     for their actual residue counts (a handful of real proteins: 1ubq, 1mbn,
     3pgk, 5awl, 1BC8, 1SMD).
  2. A synthetic-but-documented length distribution: a two-component mixture
     of log-normals informed by well-known public facts about protein length
     statistics -- single-domain proteins cluster ~50-300 residues; the human
     proteome median is ~375 residues; CASP/design benchmarks commonly range
     50-500 with a long tail to 1000+. Component A (weight 0.70) is
     ``LogNormal(median=140, sigma=0.30)`` standing in for single-domain/CATH-
     domain-like entries; Component B (weight 0.30) is
     ``LogNormal(median=380, sigma=0.45)`` standing in for larger single
     chains / multi-domain AFDB entries and the long tail. Lengths are clipped
     to ``[20, 1500]``.

This is explicitly labeled a proxy/synthetic distribution informed by public
domain knowledge, NOT CATH/AFDB statistics -- do not cite it as such.

xtrax API surface actually used
--------------------------------
``xtrax.eda.analyze_bucket``/``explain_plan`` operate on a *single*
`BatchPlan` decision and report the configured `bucket_boundaries` (they don't
histogram a length distribution -- there's no such distribution-level input in
their signature). This script therefore does both:

  * Builds one representative `xtrax.tiling.AxisSpec` (residue axis,
    `bucket_boundaries=(64, 128, 256, 512)`), plans it with `BatchPlanner`, and
    runs `xtrax.eda.explain_plan`/`analyze_bucket` on the resulting decision --
    demonstrating the real EDA entry points resolve to the `Bucket` strategy
    and report the confirmed boundaries with non-empty reasoning.
  * Separately assigns every length in the proxy distribution to a bucket
    using `xtrax.tiling.select_bucket` -- the exact same host-side primitive
    `Bucket`/`BatchPlanner` rely on internally (`xtrax/tiling/bucket.py`) --
    to compute the per-bucket population counts and padding waste that
    `analyze_bucket` itself does not expose across a distribution.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from xtrax.eda import analyze_bucket, explain_plan
from xtrax.tiling import AxisSpec, BatchPlanner, select_bucket

logger = logging.getLogger(__name__)

DEFAULT_BOUNDARIES: tuple[int, ...] = (64, 128, 256, 512)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "tests" / "data"
EGREGIOUS_WASTE_PCT = 50.0

# Two-component synthetic mixture (see module docstring for the public-domain
# rationale). Documented here, not buried, so the construction is reviewable.
SYNTHETIC_WEIGHT_SINGLE_DOMAIN = 0.70
SYNTHETIC_MEDIAN_SINGLE_DOMAIN = 140.0
SYNTHETIC_SIGMA_SINGLE_DOMAIN = 0.30
SYNTHETIC_WEIGHT_LARGE_MULTI_DOMAIN = 0.30
SYNTHETIC_MEDIAN_LARGE_MULTI_DOMAIN = 380.0
SYNTHETIC_SIGMA_LARGE_MULTI_DOMAIN = 0.45
SYNTHETIC_MIN_LENGTH = 20
SYNTHETIC_MAX_LENGTH = 1500


@dataclass
class ProxyDistribution:
  """A residue-length proxy distribution tagged by provenance.

  Attributes:
    real_lengths: Residue counts parsed from real local PDB/mmCIF fixtures.
    real_sources: File paths the ``real_lengths`` entries were parsed from.
    synthetic_lengths: Residue counts drawn from the documented synthetic mixture.
  """

  real_lengths: list[int] = field(default_factory=list)
  real_sources: list[str] = field(default_factory=list)
  synthetic_lengths: list[int] = field(default_factory=list)

  @property
  def all_lengths(self) -> list[int]:
    """All lengths, real fixtures followed by synthetic draws."""
    return [*self.real_lengths, *self.synthetic_lengths]


def load_real_lengths(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[list[int], list[str]]:
  """Parse residue counts from the real local PDB/mmCIF fixtures.

  Uses ``aminx.io.parsing.parse_structure`` (proxide-backed) and reads the
  parsed ``Protein.aatype`` leading-axis length as the residue count. Files
  that fail to parse are skipped with a warning rather than aborting the
  whole proxy build -- a handful of missing fixtures should not block this
  mechanical check.

  Args:
    data_dir: Directory to scan for ``*.pdb``/``*.cif`` files.

  Returns:
    ``(lengths, source_paths)`` -- parallel lists, one entry per successfully
    parsed structure.
  """
  from aminx.io.parsing import parse_structure  # noqa: PLC0415 (heavy import; keep lazy)

  lengths: list[int] = []
  sources: list[str] = []
  paths = sorted(data_dir.glob("*.pdb")) + sorted(data_dir.glob("*.cif"))
  for path in paths:
    try:
      protein = parse_structure(str(path))
    except Exception:  # noqa: BLE001 -- a bad fixture must not abort the whole gate
      logger.warning("skipping unparseable real fixture: %s", path, exc_info=True)
      continue
    n_res = int(protein.aatype.shape[0])
    lengths.append(n_res)
    sources.append(str(path))
    logger.info("real fixture %s -> %d residues", path.name, n_res)
  return lengths, sources


def build_synthetic_lengths(n: int, seed: int) -> list[int]:
  """Draw ``n`` residue lengths from the documented two-component log-normal mixture.

  See the module docstring for the public-domain rationale behind the
  component medians/weights. This is a **synthetic proxy**, not a fit to any
  real corpus.

  Args:
    n: Number of synthetic lengths to draw.
    seed: PRNG seed (for reproducibility of this proxy run).

  Returns:
    A list of ``n`` positive integer residue lengths, clipped to
    ``[SYNTHETIC_MIN_LENGTH, SYNTHETIC_MAX_LENGTH]``.
  """
  rng = np.random.default_rng(seed)
  n_single = int(round(n * SYNTHETIC_WEIGHT_SINGLE_DOMAIN))
  n_large = n - n_single

  single_domain = rng.lognormal(
    mean=np.log(SYNTHETIC_MEDIAN_SINGLE_DOMAIN),
    sigma=SYNTHETIC_SIGMA_SINGLE_DOMAIN,
    size=n_single,
  )
  large_multi_domain = rng.lognormal(
    mean=np.log(SYNTHETIC_MEDIAN_LARGE_MULTI_DOMAIN),
    sigma=SYNTHETIC_SIGMA_LARGE_MULTI_DOMAIN,
    size=n_large,
  )
  lengths = np.concatenate([single_domain, large_multi_domain])
  lengths = np.clip(np.round(lengths), SYNTHETIC_MIN_LENGTH, SYNTHETIC_MAX_LENGTH)
  return [int(v) for v in lengths]


def build_proxy_distribution(
  n_synthetic: int,
  seed: int,
  data_dir: Path = DEFAULT_DATA_DIR,
) -> ProxyDistribution:
  """Build the combined real + synthetic proxy length distribution."""
  real_lengths, real_sources = load_real_lengths(data_dir)
  synthetic_lengths = build_synthetic_lengths(n_synthetic, seed)
  return ProxyDistribution(
    real_lengths=real_lengths,
    real_sources=real_sources,
    synthetic_lengths=synthetic_lengths,
  )


def demonstrate_xtrax_eda(boundaries: tuple[int, ...], representative_length: int) -> dict:
  """Run xtrax's own plan-construction + EDA tooling on one representative axis.

  Builds an ``AxisSpec`` for the residue axis with the confirmed
  ``bucket_boundaries``, plans it with ``BatchPlanner`` (selection rule 1:
  ``bucket_boundaries is not None`` -> ``Bucket`` strategy), then calls
  ``xtrax.eda.explain_plan``/``analyze_bucket`` on the resulting decision.

  This does not, by itself, tell us anything about a length *distribution* --
  ``analyze_bucket`` only reports the boundary configuration of the single
  decision passed to it. Its purpose here is to prove xtrax's own EDA entry
  points resolve to the `Bucket` strategy and report the confirmed boundaries
  with non-empty reasoning, per the `using-xtrax` HiTL gate requirement.

  Args:
    boundaries: The confirmed bucket boundaries.
    representative_length: Cardinality to plan for (e.g. the proxy median).

  Returns:
    A dict with the ``explain_plan`` stats and the ``analyze_bucket`` entry
    for the residue axis decision.
  """
  spec = AxisSpec(
    name="residue",
    cardinality=representative_length,
    default_batch_size=1,
    bucket_boundaries=boundaries,
  )
  plan = BatchPlanner().plan([spec])
  stats = explain_plan(plan)
  bucket_entry = analyze_bucket(plan.decisions[0])
  logger.info(
    "xtrax EDA: axis=%s strategy=%s reasoning=%r bucket_boundaries=%s",
    stats["axes"][0]["name"],
    stats["axes"][0]["strategy"],
    stats["axes"][0]["reasoning"],
    bucket_entry["bucket_boundaries"],
  )
  return {"explain_plan": stats, "analyze_bucket": bucket_entry}


@dataclass
class BucketStat:
  """Per-bucket population + padding-waste statistics."""

  boundary: int
  count: int
  count_pct: float
  mean_length: float | None
  mean_pad_waste_pct: float | None
  max_pad_waste_pct: float | None


def analyze_distribution(
  lengths: list[int],
  boundaries: tuple[int, ...],
) -> tuple[list[BucketStat], list[int]]:
  """Assign every length in ``lengths`` to a bucket via ``xtrax.tiling.select_bucket``.

  Uses the identical host-side primitive the `Bucket` strategy relies on
  internally, so bucket assignment here matches production dispatch exactly
  (not a reimplementation of the bucketing rule).

  Args:
    lengths: Proxy residue lengths.
    boundaries: Confirmed, sorted, strictly-ascending bucket boundaries.

  Returns:
    ``(bucket_stats, overflow_lengths)`` -- per-boundary stats (in boundary
    order) and the sub-list of lengths that exceed the largest boundary
    (would need truncation/rejection under the current contract).
  """
  assigned: Counter[int] = Counter()
  by_bucket: dict[int, list[int]] = {b: [] for b in boundaries}
  overflow_lengths: list[int] = []

  for length in lengths:
    try:
      bucket = select_bucket(length, boundaries)
    except ValueError:
      overflow_lengths.append(length)
      continue
    assigned[bucket] += 1
    by_bucket[bucket].append(length)

  total = len(lengths)
  stats: list[BucketStat] = []
  for boundary in boundaries:
    members = by_bucket[boundary]
    if members:
      mean_length = float(np.mean(members))
      mean_pad_waste_pct = (1.0 - mean_length / boundary) * 100.0
      # Max waste corresponds to the smallest member of the bucket.
      max_pad_waste_pct = (1.0 - min(members) / boundary) * 100.0
    else:
      mean_length = None
      mean_pad_waste_pct = None
      max_pad_waste_pct = None
    stats.append(
      BucketStat(
        boundary=boundary,
        count=len(members),
        count_pct=(len(members) / total * 100.0) if total else 0.0,
        mean_length=mean_length,
        mean_pad_waste_pct=mean_pad_waste_pct,
        max_pad_waste_pct=max_pad_waste_pct,
      )
    )
  return stats, overflow_lengths


def render_report(
  proxy: ProxyDistribution,
  boundaries: tuple[int, ...],
  bucket_stats: list[BucketStat],
  overflow_lengths: list[int],
) -> str:
  """Render the summary table + PASS/CONCERN verdict as a printable string."""
  lines: list[str] = []
  total = len(proxy.all_lengths)
  lines.append("=" * 78)
  lines.append("E4.5 xtrax HiTL gate: residue-axis bucket boundaries")
  lines.append("=" * 78)
  lines.append(
    f"Proxy distribution: {len(proxy.real_lengths)} real fixtures "
    f"({', '.join(Path(s).name for s in proxy.real_sources)}) + "
    f"{len(proxy.synthetic_lengths)} synthetic draws = {total} total lengths."
  )
  lines.append(
    "LIMITATION: synthetic component is a documented log-normal-mixture proxy "
    "informed by public protein-length statistics, NOT a fit to the real "
    "CATH/AFDB/complex training corpus (not downloaded in this environment). "
    "Re-run against real corpus length stats once available."
  )
  lines.append("-" * 78)
  lines.append(f"Confirmed bucket boundaries: {boundaries}  (compile count = {len(boundaries)})")
  lines.append("-" * 78)
  header = (
    f"{'boundary':>9} {'count':>7} {'count%':>8} {'mean_len':>9} "
    f"{'mean_waste%':>12} {'max_waste%':>11}"
  )
  lines.append(header)
  for stat in bucket_stats:
    mean_len = f"{stat.mean_length:.1f}" if stat.mean_length is not None else "-"
    mean_waste = f"{stat.mean_pad_waste_pct:.1f}" if stat.mean_pad_waste_pct is not None else "-"
    max_waste = f"{stat.max_pad_waste_pct:.1f}" if stat.max_pad_waste_pct is not None else "-"
    lines.append(
      f"{stat.boundary:>9} {stat.count:>7} {stat.count_pct:>7.1f}% {mean_len:>9} "
      f"{mean_waste:>12} {max_waste:>11}"
    )
  lines.append("-" * 78)

  n_overflow = len(overflow_lengths)
  overflow_pct = (n_overflow / total * 100.0) if total else 0.0
  lines.append(
    f"Structures exceeding largest boundary ({boundaries[-1]}): "
    f"{n_overflow} / {total} ({overflow_pct:.1f}%)."
  )
  if overflow_lengths:
    preview = sorted(overflow_lengths)[:10]
    lines.append(f"  overflow length sample (sorted, first 10): {preview}")
    lines.append(
      "  These would hit select_bucket's ValueError under the confirmed "
      "boundaries today (recompilation-risk contract) -- they need an "
      "explicit truncation or rejection policy upstream of dispatch; this "
      "script does not decide that policy."
    )
  lines.append("-" * 78)

  egregious = [s for s in bucket_stats if (s.mean_pad_waste_pct or 0.0) > EGREGIOUS_WASTE_PCT]
  if egregious:
    verdict_waste = "CONCERN"
    waste_detail = ", ".join(f"bucket={s.boundary} mean_waste={s.mean_pad_waste_pct:.1f}%" for s in egregious)
  else:
    verdict_waste = "PASS"
    waste_detail = f"no bucket's mean padding waste exceeds {EGREGIOUS_WASTE_PCT:.0f}%"

  verdict_overflow = "CONCERN" if overflow_lengths else "PASS"

  lines.append(f"Padding-waste verdict: {verdict_waste} ({waste_detail}).")
  lines.append(
    f"Overflow verdict: {verdict_overflow} "
    f"({n_overflow} of {total} proxy lengths exceed {boundaries[-1]})."
  )
  overall = "CONCERN" if (verdict_waste == "CONCERN" or verdict_overflow == "CONCERN") else "PASS"
  lines.append(f"OVERALL: {overall}")
  lines.append(
    "NOTE (per E4.5 scope): the confirmed boundary VALUES (64,128,256,512) "
    "are NOT being changed by this script regardless of verdict -- any "
    "CONCERN above is flagged for the epic owner to consider, not acted on."
  )
  lines.append("=" * 78)
  return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
  doc_summary = (__doc__ or "").splitlines()
  parser = argparse.ArgumentParser(description=doc_summary[0] if doc_summary else None)
  parser.add_argument(
    "--n-synthetic",
    type=int,
    default=2000,
    help="Number of synthetic proxy lengths to draw (default: 2000).",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="PRNG seed for the synthetic component (default: 0, for reproducibility).",
  )
  parser.add_argument(
    "--data-dir",
    type=Path,
    default=DEFAULT_DATA_DIR,
    help="Directory to scan for real PDB/mmCIF fixtures (default: tests/data).",
  )
  parser.add_argument(
    "--boundaries",
    type=int,
    nargs="+",
    default=list(DEFAULT_BOUNDARIES),
    help="Bucket boundaries to analyze (default: 64 128 256 512, the confirmed values).",
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="Optional path to write a JSON dump of the analysis alongside the printed report.",
  )
  parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging.")
  args = parser.parse_args(argv)

  logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

  boundaries = tuple(args.boundaries)
  proxy = build_proxy_distribution(args.n_synthetic, args.seed, args.data_dir)
  all_lengths = proxy.all_lengths

  eda_demo = demonstrate_xtrax_eda(boundaries, representative_length=int(np.median(all_lengths)))
  bucket_stats, overflow_lengths = analyze_distribution(all_lengths, boundaries)

  report = render_report(proxy, boundaries, bucket_stats, overflow_lengths)
  print(report)  # noqa: T201 -- this is the deliverable's printed summary, not debug noise

  if args.out is not None:
    payload = {
      "boundaries": list(boundaries),
      "n_real": len(proxy.real_lengths),
      "real_sources": proxy.real_sources,
      "n_synthetic": len(proxy.synthetic_lengths),
      "seed": args.seed,
      "bucket_stats": [vars(s) for s in bucket_stats],
      "overflow_count": len(overflow_lengths),
      "overflow_lengths": overflow_lengths,
      "xtrax_eda_demo": {
        "strategy": eda_demo["explain_plan"]["axes"][0]["strategy"],
        "reasoning": eda_demo["explain_plan"]["axes"][0]["reasoning"],
        "bucket_boundaries": eda_demo["analyze_bucket"]["bucket_boundaries"],
      },
    }
    args.out.write_text(json.dumps(payload, indent=2))
    logger.info("wrote JSON dump to %s", args.out)

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
