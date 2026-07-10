"""Mutation DeltaDeltaG / stability application logic on top of E4's dispatch wiring (backlog node **E6**).

Per the EPIC DAG (``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md``
§2, E6 row) and the design spec
(``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §1, §3.3 row
2), this module owns the *real* accuracy-gated DeltaDeltaG-stability
application: given a wildtype structure + sequence, build point-mutant
sequences, and apply the Monte-Carlo unfolded-state correction the design
spec's formula requires --

    ddG(mutant) = E(x, s_mutant) - E(x, s_wildtype) - mean_{u in p_UF}[E(x_u, s)]

-- on top of ``aminx.ebm.dispatch.score_mutant_ensemble`` (backlog node E4,
**not modified here** -- only imported).

**IMPORTANT scope limitation (read before trusting any number this module
produces).** The real ProteinGym/Tsuboyama-cDNA-proteolysis benchmark the
design spec's Spearman >= 0.686 target refers to is **NOT downloaded in this
environment** (confirmed absent; the reference's own download script fetches
multi-GB ZIP archives from ``marks.hms.harvard.edu`` -- not authorized for
this backlog node). This module therefore cannot and does not compute a real
Spearman correlation against experimental ddG labels. Every validation in
``tests/ebm/test_ddg_stability.py`` is **pipeline-sanity** validation only:
proving the wiring/composition behaves correctly and that mutations produce
numerically sensible, non-degenerate energy differences on real PDB
geometry -- not a reproduction of the published benchmark number. See that
test module's docstring for the exact, honestly-bounded claims made.

**Deliberate deviation from ``score_mutant_ensemble``'s ``unfolded_ensemble_
aatype`` slot.** Reading E4's dispatch.py in full: that keyword varies
*sequence* on the *same fixed (folded) backbone coordinates* shared with the
mutant/wildtype axes (see its docstring: "coords: Fixed backbone
coordinates, shared across all mutants/WT/ensemble members" --
``aminx/ebm/dispatch.py::score_mutant_ensemble``). That is a legitimate
E4 wiring-scope simplification (E4 is explicitly "axis-dispatch + mean-Fuse
wiring only", not the real unfolded-state protocol), but it is *not* what the
design spec's MC unfolded-state correction physically means: an ensemble of
**unfolded/denatured backbone conformations** of the same sequence length,
each with genuinely different (non-native) coordinates, whose mean energy is
subtracted as a reference baseline. Since ``score_mutant_ensemble``'s API has
no slot for per-ensemble-member *coordinates*, this module does **not** force
the coordinate-varying correction through that sequence-only slot. Instead:

* ``score_mutant_ensemble`` is called with only ``wildtype_aatype`` (no
  ``unfolded_ensemble_aatype``) to get the real, correct
  ``E(mutant) - E(wildtype)`` term on the shared folded backbone.
* The MC unfolded-state correction is computed independently by
  :func:`unfolded_state_correction`, built from the *same* generic
  ``aminx.ebm.plan`` primitives (``plan_axis``/``dispatch_axis``/
  ``mean_fuse``) that ``aminx.ebm.dispatch`` itself is built from -- reusing
  the pattern, not editing the file that pattern lives in.
* The two terms are combined by this module's own :func:`compute_ddg_stability`,
  matching the design spec's formula exactly, with genuinely varying
  unfolded-ensemble coordinates.

**Synthetic unfolded-backbone generator (documented substitute, not a port).**
The reference's own ``generate_random_backbone_coords`` is not ported here
and does not need to be exactly replicated for this wiring-validation scope
(per this node's task brief). :func:`generate_synthetic_unfolded_ensemble`
is a deliberately simple substitute: a plain 3-D random walk with a fixed
step length approximating a real CA-CA bond length (~3.8 Angstrom, i.e.
``0.38`` in the ``coordinate_scaling``-scaled space -- see
``aminx.ebm.diffusion``'s ``DEFAULT_COORDINATE_SCALING``), clearly
non-native (no secondary structure, no tertiary packing, no self-avoidance).
This is *sufficient* to prove the mean-Fuse correction behaves correctly
(shape, convergence, matches a manual mean) but is **not** a physically
rigorous unfolded-state ensemble sampler and must not be cited as one.

**PDB loading / mutation generation are new, self-contained helpers.** No
lightweight (proxide-free) PDB-to-CA-coordinates parser exists elsewhere in
aminx (the existing ``aminx.io.parsing`` path goes through the full
``proxide`` atom37 structure pipeline); this module uses ``Bio.PDB``
(``biopython>=1.83``, an existing aminx dependency) directly for a minimal
CA-only extraction, since this node needs only CA coordinates + a residue
mask + amino-acid identities, not a full atom37 structure. The amino-acid
index vocabulary matches the rest of aminx's MPNN-style alphabet
(``aminx.utils.aa_convert.MPNN_ALPHABET`` = ``"ACDEFGHIKLMNPQRSTVWYX"``, ``X``
= mask/unknown at index 20), consistent with
``aminx.ebm.contracts.AAType``'s documented "21-way embedding incl. mask
token 20" contract. **Caveat, stated plainly:** whether the ProteinEBM
reference checkpoint's ``sequence_embedding`` table was actually trained
against *this exact* letter ordering was not independently verified anywhere
in this epic so far (E3.5's parity gate validates shapes/values on
arbitrary/random ``aatype`` integers, not the semantic letter-to-index
mapping) -- this module inherits that same, already-existing assumption
rather than introducing a new one.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from aminx.ebm.dispatch import score_mutant_ensemble
from aminx.ebm.plan import EBMAxisNames, dispatch_axis, plan_axis
from aminx.ebm.plan import mean_fuse as _default_mean_fuse
from aminx.utils.aa_convert import MPNN_ALPHABET

if TYPE_CHECKING:
  from collections.abc import Sequence
  from pathlib import Path

  from jaxtyping import Array, Float, Int, PRNGKeyArray

  from aminx.ebm.contracts import AAType, Coords, DiffusionTime, Energy, ResidueMask
  from aminx.ebm.model import ProteinEBMModel
  from aminx.ebm.plan import EnergyFusionFn, EnergyVector

# ``coordinate_scaling`` -- matches ``aminx.ebm.diffusion.DEFAULT_COORDINATE_SCALING``
# exactly (0.1, Angstrom -> the scaled-nanometre ``Coords`` space every other
# ``aminx.ebm`` function consumes). Duplicated as a plain float constant here
# (not imported) to keep this module import-isolated from ``aminx.ebm.
# diffusion``'s VP-SDE machinery, which this module has no other reason to
# depend on; the two constants must stay numerically identical by
# construction -- see the module docstring's "CoordinateScalingContract" cross-
# reference in ``aminx.ebm.model``/``aminx.ebm.readout``.
_COORDINATE_SCALING = 0.1

# Real CA-CA bond length is ~3.8 Angstrom; in the coordinate_scaling-scaled
# space that is 3.8 * 0.1 = 0.38.
_DEFAULT_UNFOLDED_STEP_LENGTH = 3.8 * _COORDINATE_SCALING

_MASK_LETTER = "X"
_MASK_AA_INDEX = MPNN_ALPHABET.index(_MASK_LETTER)  # 20, matches AAType's documented mask token.

# A coarse, textbook hydrophobic-core set (Kyte-Doolittle "hydrophobic" tier),
# used only by :func:`identify_buried_hydrophobic_positions`'s selection
# heuristic -- not a physical claim, a sanity-check convenience.
_HYDROPHOBIC_LETTERS = frozenset("AVLIMFWY")


@dataclasses.dataclass(frozen=True)
class WildtypeStructure:
  """A parsed CA-only backbone: coordinates, sequence, mask, and PDB residue numbers.

  ``coords`` are already in the ``coordinate_scaling``-scaled space (see
  module docstring) -- ready to pass directly to
  ``aminx.ebm.model.ProteinEBMModel.energy``/``score`` or
  ``aminx.ebm.dispatch.score_mutant_ensemble`` without further scaling.
  """

  coords: Coords
  aatype: AAType
  mask: ResidueMask
  residue_ids: tuple[int, ...]
  sasa: tuple[float, ...] | None = None


def load_ca_backbone_from_pdb(
  path: str | Path,
  chain_id: str | None = None,
  *,
  compute_sasa: bool = False,
) -> WildtypeStructure:
  """Parse a local PDB file into a CA-only :class:`WildtypeStructure`.

  Minimal, self-contained ``Bio.PDB``-based extraction (see module
  docstring: no lightweight CA-only parser exists elsewhere in aminx, and
  this node needs only CA coordinates + sequence + mask, not a full atom37
  structure). Skips heteroatoms/waters (``residue.id[0] != " "``) and any
  residue missing a CA atom. Non-standard/unrecognized residue names map to
  the mask token (index 20), matching ``Bio.SeqUtils.seq1``'s
  ``undef_code="X"`` default.

  Args:
      path: Path to a local ``.pdb`` file.
      chain_id: Which chain to extract (first chain in the first model if
          ``None``).
      compute_sasa: If ``True``, also compute per-residue solvent-accessible
          surface area (``Bio.PDB.SASA.ShrakeRupley``, residue-level) and
          populate :attr:`WildtypeStructure.sasa`, aligned to the same
          residue ordering as ``coords``/``aatype``. Used by
          :func:`identify_buried_hydrophobic_positions`; off by default
          since it is extra (cheap but non-trivial) computation most callers
          don't need.

  Returns:
      A :class:`WildtypeStructure` with ``coords`` already
      ``coordinate_scaling``-scaled.

  Raises:
      ValueError: If no CA atoms are found in the requested chain.

  """
  from Bio.PDB import PDBParser  # noqa: PLC0415
  from Bio.SeqUtils import seq1  # noqa: PLC0415

  path_str = str(path)
  parser = PDBParser(QUIET=True)
  structure = parser.get_structure("wildtype", path_str)
  model = next(iter(structure))
  chain = model[chain_id] if chain_id is not None else next(iter(model))

  residues = [r for r in chain if r.id[0] == " " and "CA" in r]
  if not residues:
    msg = f"No CA-bearing residues found in {path_str!r} (chain {chain_id!r})"
    raise ValueError(msg)

  coords_angstrom = np.stack([r["CA"].get_coord() for r in residues]).astype(np.float32)
  one_letter = [seq1(r.get_resname(), undef_code=_MASK_LETTER) for r in residues]
  aatype_np = np.array(
    [MPNN_ALPHABET.index(aa) if aa in MPNN_ALPHABET else _MASK_AA_INDEX for aa in one_letter],
    dtype=np.int32,
  )
  residue_ids = tuple(int(r.id[1]) for r in residues)

  sasa_tuple: tuple[float, ...] | None = None
  if compute_sasa:
    from Bio.PDB.SASA import ShrakeRupley  # noqa: PLC0415

    ShrakeRupley().compute(structure, level="R")
    sasa_tuple = tuple(float(r.sasa) for r in residues)

  coords = jnp.asarray(coords_angstrom) * _COORDINATE_SCALING
  aatype = jnp.asarray(aatype_np)
  mask = jnp.ones((len(residues),), dtype=bool)
  return WildtypeStructure(coords=coords, aatype=aatype, mask=mask, residue_ids=residue_ids, sasa=sasa_tuple)


def identify_buried_hydrophobic_positions(
  wildtype: WildtypeStructure,
  *,
  sasa_threshold: float = 20.0,
) -> list[int]:
  """Return 0-indexed positions of buried, hydrophobic wildtype residues.

  A coarse sanity-check heuristic (not a rigorous burial definition): a
  position qualifies if its residue-level SASA (``Bio.PDB.SASA.ShrakeRupley``,
  square Angstrom) is below ``sasa_threshold`` *and* its wildtype identity is
  in the textbook Kyte-Doolittle hydrophobic tier (``AVLIMFWY``). Used to
  pick biologically meaningful positions for the "buried core residue ->
  Pro/Gly is destabilizing" pipeline-sanity check (see design spec's
  application scope, task brief item 3) -- requires
  ``wildtype.sasa`` to be populated (``load_ca_backbone_from_pdb(...,
  compute_sasa=True)``).

  Args:
      wildtype: A structure loaded with ``compute_sasa=True``.
      sasa_threshold: Residues at or below this SASA (Angstrom^2) count as
          "buried".

  Returns:
      0-indexed positions (aligned with ``wildtype.coords``/``aatype``),
      sorted by increasing SASA (most-buried first).

  Raises:
      ValueError: If ``wildtype.sasa`` is ``None``.

  """
  if wildtype.sasa is None:
    msg = "identify_buried_hydrophobic_positions requires a WildtypeStructure loaded with compute_sasa=True"
    raise ValueError(msg)

  letters = [MPNN_ALPHABET[int(a)] for a in np.asarray(wildtype.aatype)]
  candidates = [
    (sasa, i)
    for i, (sasa, letter) in enumerate(zip(wildtype.sasa, letters, strict=True))
    if sasa <= sasa_threshold and letter in _HYDROPHOBIC_LETTERS
  ]
  candidates.sort()
  return [i for _sasa, i in candidates]


def make_point_mutants(
  wildtype_aatype: AAType,
  mutations: Sequence[tuple[int, str]],
) -> Int[Array, "M N"]:
  """Build ``M`` stacked mutant-sequence rows, one point substitution each.

  Args:
      wildtype_aatype: The reference sequence, shape ``(N,)``.
      mutations: ``(position, new_aa_letter)`` pairs, one per mutant row.
          ``position`` is 0-indexed into ``wildtype_aatype``; ``new_aa_letter``
          is a single-character code in ``aminx.utils.aa_convert.
          MPNN_ALPHABET``.

  Returns:
      Stacked mutant sequences, shape ``(len(mutations), N)``.

  Raises:
      ValueError: If a letter is not in ``MPNN_ALPHABET`` or a position is
          out of range.

  """
  n = wildtype_aatype.shape[0]
  rows = []
  for position, letter in mutations:
    if letter not in MPNN_ALPHABET:
      msg = f"make_point_mutants: unrecognized amino-acid letter {letter!r} (expected one of {MPNN_ALPHABET!r})"
      raise ValueError(msg)
    if not (0 <= position < n):
      msg = f"make_point_mutants: position {position} out of range for sequence length {n}"
      raise ValueError(msg)
    rows.append(wildtype_aatype.at[position].set(MPNN_ALPHABET.index(letter)))
  return jnp.stack(rows, axis=0)


def random_point_mutants(
  key: PRNGKeyArray,
  wildtype_aatype: AAType,
  positions: Sequence[int],
  *,
  exclude_wildtype: bool = True,
) -> tuple[Int[Array, "M N"], list[tuple[int, str]]]:
  """Draw one random (by default, non-wildtype) substitution per requested position.

  Convenience generator for validation-only synthetic mutants (this node has
  no real experimental mutant list -- see module docstring). Each position
  gets an independent draw via ``jax.random.fold_in(key, position)`` (DAG
  Fork 10's per-element-key discipline), so results are reproducible given
  the same ``key``/``positions`` and independent of draw order.

  Args:
      key: Base PRNG key.
      wildtype_aatype: The reference sequence, shape ``(N,)``.
      positions: 0-indexed positions to mutate, one mutant row per position.
      exclude_wildtype: If ``True`` (default), redraw until the sampled
          amino acid differs from the wildtype identity at that position
          (a "point mutation" should change the residue); the mask token
          (index 20) is also excluded from the draw pool either way, since
          "mutating to unknown" is not a meaningful synthetic mutation.

  Returns:
      ``(mutant_aatype, mutations)`` -- the stacked ``(len(positions), N)``
      array (see :func:`make_point_mutants`) and the human-readable
      ``(position, new_letter)`` list used to build it.

  """
  standard_letters = MPNN_ALPHABET[:_MASK_AA_INDEX]  # exclude the mask token from the draw pool.
  mutations: list[tuple[int, str]] = []
  for position in positions:
    wt_index = int(wildtype_aatype[position])
    position_key = jax.random.fold_in(key, position)
    draw_attempt = 0
    while True:
      attempt_key = jax.random.fold_in(position_key, draw_attempt)
      candidate_index = int(jax.random.randint(attempt_key, (), 0, len(standard_letters)))
      if not exclude_wildtype or candidate_index != wt_index:
        break
      draw_attempt += 1
    mutations.append((position, standard_letters[candidate_index]))
  return make_point_mutants(wildtype_aatype, mutations), mutations


def generate_synthetic_unfolded_ensemble(
  key: PRNGKeyArray,
  n_residues: int,
  n_ensemble: int,
  *,
  step_length: float = _DEFAULT_UNFOLDED_STEP_LENGTH,
) -> Float[Array, "U N 3"]:
  """Sample ``U`` synthetic unfolded-backbone conformations of length ``n_residues``.

  **Documented, simple substitute, not a reference port** (see module
  docstring): a plain 3-D random walk per ensemble member, fixed step length
  approximating a real CA-CA bond, each step's direction an independent
  isotropic unit vector. This has no secondary structure, no tertiary
  packing, and is not self-avoiding -- clearly non-native, which is exactly
  the "large-variance / Ramachandran-random-like" role the design spec's MC
  unfolded-state correction needs a reference ensemble to play. Each member
  is centered at its own centroid (translation-invariance is irrelevant to
  ``ProteinEBMModel.energy``, which is non-equivariant per the design spec,
  but centering keeps the ensemble's coordinate scale comparable across
  members regardless of walk length).

  Args:
      key: Base PRNG key, split once per ensemble member via
          ``jax.random.split`` (Fork 10 discipline: no key is reused).
      n_residues: Sequence length ``N`` (must match the wildtype/mutant
          structure this ensemble corrects).
      n_ensemble: Ensemble size ``U``.
      step_length: Per-step displacement magnitude, in the same
          ``coordinate_scaling``-scaled space as ``Coords`` (default: real
          ~3.8 Angstrom CA-CA bond length, scaled).

  Returns:
      Stacked unfolded-ensemble coordinates, shape ``(n_ensemble, n_residues, 3)``.

  """
  member_keys = jax.random.split(key, n_ensemble)

  def _one_walk(member_key: PRNGKeyArray) -> Float[Array, "N 3"]:
    directions = jax.random.normal(member_key, (n_residues, 3))
    unit_directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    steps = unit_directions * step_length
    walk = jnp.cumsum(steps, axis=0)
    return walk - jnp.mean(walk, axis=0, keepdims=True)

  return jax.vmap(_one_walk)(member_keys)


def unfolded_state_correction(
  model: ProteinEBMModel,
  aatype: AAType,
  t: DiffusionTime,
  mask: ResidueMask,
  unfolded_coords_ensemble: Float[Array, "U N 3"],
  *,
  mean_fuse: EnergyFusionFn = _default_mean_fuse,
  default_batch_size: int | None = None,
) -> Energy:
  """Mean energy over a genuinely coordinate-varying unfolded-backbone ensemble.

  The real MC unfolded-state correction (design spec §1/§3.3 row 2): unlike
  ``aminx.ebm.dispatch.score_mutant_ensemble``'s ``unfolded_ensemble_aatype``
  slot (which varies *sequence* on the *same fixed folded coordinates* -- see
  module docstring), this dispatches ``model.energy`` over the *ensemble*
  axis of genuinely different (unfolded) coordinates, one fixed sequence
  shared across all members -- built from the same generic
  ``aminx.ebm.plan`` primitives ``aminx.ebm.dispatch`` itself uses.

  Args:
      model: The assembled ProteinEBM-equivalent model.
      aatype: Fixed sequence scored against every unfolded conformation
          (typically the wildtype sequence -- see
          :func:`compute_ddg_stability`).
      t: Fixed diffusion time.
      mask: Fixed residue mask.
      unfolded_coords_ensemble: Stacked unfolded-backbone coordinates,
          leading axis = ensemble size ``U`` (e.g. from
          :func:`generate_synthetic_unfolded_ensemble`).
      mean_fuse: The mean-``Fuse`` primitive. Defaults to
          ``aminx.ebm.plan.mean_fuse``.
      default_batch_size: Override the ``n_ensemble`` axis's Vmap/SafeMap
          cutoff.

  Returns:
      The scalar mean energy over the ensemble.

  """
  n_ensemble = unfolded_coords_ensemble.shape[0]
  decision = plan_axis(EBMAxisNames.N_ENSEMBLE, n_ensemble, default_batch_size=default_batch_size)

  def _score_one(c: Coords) -> Energy:
    return model.energy(c, aatype, t, mask)

  ensemble_energies: EnergyVector = dispatch_axis(decision.strategy, _score_one, unfolded_coords_ensemble)
  return mean_fuse(ensemble_energies)


@dataclasses.dataclass(frozen=True)
class DDGResult:
  """Full breakdown of a :func:`compute_ddg_stability` call.

  ``ddg`` is the design spec's headline quantity; the other fields are the
  intermediate terms, exposed for pipeline-sanity inspection (this is
  precisely what ``tests/ebm/test_ddg_stability.py`` checks against manual
  recomputation).
  """

  ddg: EnergyVector
  raw_ddg: EnergyVector
  wildtype_energy: Energy
  unfolded_correction: Energy
  mutations: tuple[tuple[int, str], ...]


def compute_ddg_stability(
  model: ProteinEBMModel,
  wildtype: WildtypeStructure,
  mutations: Sequence[tuple[int, str]],
  *,
  t: DiffusionTime | float = 0.05,
  n_unfolded_ensemble: int = 8,
  unfolded_step_length: float = _DEFAULT_UNFOLDED_STEP_LENGTH,
  key: PRNGKeyArray,
  mean_fuse: EnergyFusionFn = _default_mean_fuse,
  default_batch_size: int | None = None,
) -> DDGResult:
  """Full ddG-stability pipeline: point mutants -> mutant-vs-WT -> MC unfolded correction.

  Implements the design spec's §1 formula
  ``ddG = E(mutant) - E(wildtype) - mean_UF[E(unfolded)]`` on a real wildtype
  structure, using ``aminx.ebm.dispatch.score_mutant_ensemble`` (E4, imported
  not modified) for the mutant-vs-wildtype term and this module's own
  :func:`unfolded_state_correction` for the MC correction (see module
  docstring for why the two are not both routed through
  ``score_mutant_ensemble``'s built-in ensemble slot).

  Args:
      model: The assembled ProteinEBM-equivalent model.
      wildtype: A structure loaded via :func:`load_ca_backbone_from_pdb`.
      mutations: ``(position, new_aa_letter)`` pairs (see
          :func:`make_point_mutants`); one mutant row each.
      t: Diffusion time. Design spec's MVP parity target is
          ProteinEBM-x @ ``t=0.05`` (near-zero-noise scoring of the actual
          structure, not a diffused sample -- matches ``score_decoy_batch``/
          ``score_mutant_ensemble``'s own fixed-``t`` convention).
      n_unfolded_ensemble: Size of the synthetic unfolded-backbone ensemble.
      unfolded_step_length: Passed through to
          :func:`generate_synthetic_unfolded_ensemble`.
      key: PRNG key for the unfolded-ensemble sampling. Not reused elsewhere.
      mean_fuse: The mean-``Fuse`` primitive for the unfolded correction.
      default_batch_size: Override for both the mutant and unfolded-ensemble
          axes' Vmap/SafeMap cutoff.

  Returns:
      A :class:`DDGResult` with the final ``ddg`` vector (shape
      ``(len(mutations),)``) and every intermediate term.

  """
  mutant_aatype = make_point_mutants(wildtype.aatype, mutations)
  raw_ddg = score_mutant_ensemble(
    model,
    wildtype.coords,
    mutant_aatype,
    jnp.asarray(t),
    wildtype.mask,
    wildtype_aatype=wildtype.aatype,
    default_batch_size=default_batch_size,
  )
  wildtype_energy = model.energy(wildtype.coords, wildtype.aatype, jnp.asarray(t), wildtype.mask)

  unfolded_coords = generate_synthetic_unfolded_ensemble(
    key,
    wildtype.coords.shape[0],
    n_unfolded_ensemble,
    step_length=unfolded_step_length,
  )
  correction = unfolded_state_correction(
    model,
    wildtype.aatype,
    jnp.asarray(t),
    wildtype.mask,
    unfolded_coords,
    mean_fuse=mean_fuse,
    default_batch_size=default_batch_size,
  )

  return DDGResult(
    ddg=raw_ddg - correction,
    raw_ddg=raw_ddg,
    wildtype_energy=wildtype_energy,
    unfolded_correction=correction,
    mutations=tuple(mutations),
  )


__all__ = [
  "DDGResult",
  "WildtypeStructure",
  "compute_ddg_stability",
  "generate_synthetic_unfolded_ensemble",
  "identify_buried_hydrophobic_positions",
  "load_ca_backbone_from_pdb",
  "make_point_mutants",
  "random_point_mutants",
  "unfolded_state_correction",
]
