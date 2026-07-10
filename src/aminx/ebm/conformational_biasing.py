"""Conformational-biasing / multistate energy-gap application (backlog node **E7**).

Wiring primitive for the "conformational biasing" case study from the design
spec (``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §1's
LplA open/closed example, Cavanaugh et al., and §5's multistate mechanism):
for a *fixed* sequence and a small number ``S`` of conformational states of
the *same* protein (typically ``S=2``, e.g. an "open" and a "closed" crystal
structure), score the per-state energy gap

    ``ΔE = E(x_closed, s) - E(x_open, s)``

via **E4's** :func:`aminx.ebm.dispatch.score_state_difference` (this module
imports that function; it does not modify ``aminx.ebm.dispatch`` at all --
E4/E5/E6 own that file and are landing in parallel).

**Sign convention.** Design spec §5: ``ΔE = E(x_1,s) - E(x_2,s)`` via
difference-``Fuse``; the default ``aminx.ebm.plan.DifferenceFuse`` computes
``per_item[0] - per_item[1]``. This module's public entry point
(:func:`score_conformational_bias`) follows the EPIC brief's convention of
stacking ``[closed, open]`` (state 0 = closed, state 1 = open), so a positive
``ΔE`` means the closed state is *higher* energy (less favorable) relative to
open for that sequence -- i.e. a sequence that stabilizes/prefers the open
state. Design spec line "sign/rank of ΔE on LplA open/closed -> positive
corr. w/ activity" (§8.2) is consistent with this convention if promiscuous
activity in the real biology is associated with populating the open state
(Cavanaugh et al.) -- callers who stack states in the opposite order simply
get ``-ΔE``; this module does not hard-code which biological state name
belongs at index 0, it only fixes index 0 = the first path passed to
:func:`load_conformational_states`.

**Structure alignment (the CRITICAL check this node's brief calls out).** Two
real crystal structures of "the same protein" almost never resolve the exact
same residue range (missing/disordered loops differ per crystal). Naively
stacking their CA-coordinate arrays index-by-index is silently wrong the
moment the residue counts differ -- an off-by-N frameshift with no crash.
:func:`align_conformational_states` fixes this by aligning on each parsed
structure's own **PDB residue numbering** (``residue_index``), verifying that
every state agrees on the amino-acid identity at every residue number they
share (raising ``ValueError`` listing every disagreement otherwise -- the
mismatch is never silently scored), and reducing to the **intersection** mask
(only residues resolved in *every* state contribute to any energy compared
here). This is the same discipline
:class:`aminx.ebm.checkpoint`'s weight-port gate applies to key names: every
residue is either aligned-and-verified or explicitly excluded, never silently
misaligned.

**Coordinate frame.** ``ProteinEBMModel`` is documented **non-equivariant**
(design spec §1) -- the reference's own real-structure scoring script
(``~/repos/ProteinEBM/protein_ebm/scripts/score_decoys.py``) always calls
``center_random_augmentation`` before scoring: center each structure at its
own centroid, then apply a *random* SO(3) rotation (the model was trained
with this augmentation so it is approximately, not exactly, rotation-
invariant). This module reproduces the **centering** half exactly (each
state's CA coordinates are re-centered at the centroid of the shared/
intersection-masked residues, independently per state, before the single
``coordinate_scaling`` application -- matching
``aminx.ebm.diffusion.DEFAULT_COORDINATE_SCALING``) but **deliberately omits
the random-rotation half** for reproducibility: a random rotation would make
every ``ΔE`` computation non-deterministic across runs (the model is not
proven rotation-invariant, so different rotations give different numeric
energies), which is unacceptable for a correlation analysis that must report
one real, reproducible number. This is a genuine, documented scope
limitation of this MVP wiring node, not an oversight -- see
``scripts/ebm/lpla_biasing_check.py``'s module docstring for the same note
applied to the real validation run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

from aminx.ebm.diffusion import DEFAULT_COORDINATE_SCALING
from aminx.ebm.dispatch import score_state_difference
from aminx.ebm.plan import difference_fuse as _default_difference_fuse
from aminx.utils.aa_convert import AF_ALPHABET

if TYPE_CHECKING:
  from collections.abc import Sequence
  from pathlib import Path

  from jaxtyping import Array, Float, Int

  from aminx.ebm.contracts import AAType, DiffusionTime, Energy, ResidueMask
  from aminx.ebm.model import ProteinEBMModel
  from aminx.ebm.plan import EnergyFusionFn

  # Module-level alias (assignment, not a bare-identifier annotation) so ruff's
  # UP037 "unnecessary quoted annotation" heuristic doesn't misfire on a
  # single-token jaxtyping shape string used directly as a class-field
  # annotation -- mirrors aminx.ebm.contracts's Coords/AAType/etc. pattern.
  ResidueNumbers = Int[Array, "N"]


class ConformationalStates(NamedTuple):
  """One shared, residue-aligned coordinate/mask frame across ``S`` conformational states.

  Built by :func:`align_conformational_states` (pure-array, unit-testable) or
  :func:`load_conformational_states` (the real-PDB-file convenience wrapper).
  """

  coords_states: Float[Array, "S N 3"]
  """Per-state CA coordinates, already centered (per-state, over ``mask``) and
  ``coordinate_scaling``-scaled -- ready to pass directly as
  ``ProteinEBMModel.energy``'s ``coords`` (per state) / as
  ``score_state_difference``'s ``coords_states``."""

  mask: ResidueMask
  """Shared residue mask, ``(N,)`` -- ``True`` only where *every* state
  resolved that residue (the intersection). Both states of a fixed sequence
  must use the same mask for ``score_state_difference`` (single ``mask``
  argument, design spec §5)."""

  reference_aatype: AAType
  """Amino-acid type per canonical residue slot (``(N,)``, AF-alphabet
  order -- see module docstring), taken from whichever input state resolved
  that residue; identity-verified consistent across all states that resolved
  it (see :func:`align_conformational_states`). Present only where at least
  one state resolved the residue; entries at wholly-unresolved slots are
  undefined (masked out everywhere) and best-effort zero-filled."""

  residue_numbers: ResidueNumbers
  """Canonical (1-indexed) PDB residue numbers backing each array slot, for
  debugging/cross-referencing against the source structures."""


def align_conformational_states(
  per_state_coords: Sequence[np.ndarray],
  per_state_residue_index: Sequence[np.ndarray],
  per_state_aatype: Sequence[np.ndarray],
  *,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> ConformationalStates:
  """Align ``S`` per-state CA-coordinate arrays onto one shared, verified frame.

  Pure-array core of :func:`load_conformational_states` (no file I/O, no
  ``proxide`` dependency) -- the part worth unit-testing directly with tiny
  synthetic inputs.

  Args:
      per_state_coords: ``S`` arrays, each ``(N_i, 3)`` -- CA coordinates in
          that state's own PDB residue order, in the SAME raw physical units
          across all states (e.g. angstroms; not yet ``coordinate_scaling``
          -scaled).
      per_state_residue_index: ``S`` arrays, each ``(N_i,)`` int -- that
          state's own PDB residue numbering (1-indexed, need not be
          contiguous), same order as its ``per_state_coords`` row.
      per_state_aatype: ``S`` arrays, each ``(N_i,)`` int -- that state's
          amino-acid type per residue, AF-alphabet order (``aminx.utils.
          aa_convert.AF_ALPHABET`` / ``aminx.io.parsing.parse_structure``'s
          own ``Protein.aatype`` convention), same order as
          ``per_state_residue_index``.
      coordinate_scaling: Applied once, after per-state centering (see module
          docstring's ``CoordinateScalingContract`` note -- matches
          ``aminx.ebm.diffusion.DEFAULT_COORDINATE_SCALING``).

  Returns:
      A :class:`ConformationalStates` with the intersection mask over all
      residue numbers seen by every state.

  Raises:
      ValueError: If fewer than 2 states are given, if any state's three
          per-state arrays disagree in length, or if two states disagree on
          the amino-acid identity at a residue number **both** resolved
          (the CRITICAL alignment check -- every offending residue number is
          listed, not just the first).

  """
  n_states = len(per_state_coords)
  if n_states < 2:
    msg = f"align_conformational_states: need >=2 states, got {n_states}"
    raise ValueError(msg)
  if not (len(per_state_residue_index) == n_states and len(per_state_aatype) == n_states):
    msg = (
      "align_conformational_states: per_state_coords/per_state_residue_index/"
      f"per_state_aatype must have equal length; got "
      f"{n_states}/{len(per_state_residue_index)}/{len(per_state_aatype)}"
    )
    raise ValueError(msg)

  for i, (coords, ridx, aatype) in enumerate(
    zip(per_state_coords, per_state_residue_index, per_state_aatype, strict=True),
  ):
    if not (coords.shape[0] == ridx.shape[0] == aatype.shape[0]):
      msg = (
        f"align_conformational_states: state {i} has mismatched per-residue "
        f"lengths -- coords={coords.shape[0]}, residue_index={ridx.shape[0]}, "
        f"aatype={aatype.shape[0]}"
      )
      raise ValueError(msg)

  # Canonical numbering = union of every residue number any state resolved.
  all_residue_numbers = sorted({int(r) for ridx in per_state_residue_index for r in ridx})
  canonical_index = {r: i for i, r in enumerate(all_residue_numbers)}
  n_canonical = len(all_residue_numbers)

  aatype_by_residue: dict[int, tuple[int, int]] = {}  # residue_number -> (aatype, first_state_idx)
  mismatches: list[str] = []

  coords_canonical = np.zeros((n_states, n_canonical, 3), dtype=np.float32)
  resolved = np.zeros((n_states, n_canonical), dtype=bool)
  reference_aatype = np.zeros((n_canonical,), dtype=np.int32)

  for s, (coords, ridx, aatype) in enumerate(
    zip(per_state_coords, per_state_residue_index, per_state_aatype, strict=True),
  ):
    for row, (r, aa) in enumerate(zip(ridx, aatype, strict=True)):
      r_int, aa_int = int(r), int(aa)
      c = canonical_index[r_int]
      coords_canonical[s, c] = coords[row]
      resolved[s, c] = True
      reference_aatype[c] = aa_int
      if r_int in aatype_by_residue:
        prev_aa, prev_state = aatype_by_residue[r_int]
        if prev_aa != aa_int:
          mismatches.append(
            f"residue {r_int}: state {prev_state} has aatype {prev_aa}, state {s} has {aa_int}",
          )
      else:
        aatype_by_residue[r_int] = (aa_int, s)

  if mismatches:
    msg = (
      "align_conformational_states: amino-acid identity mismatch between "
      f"states at {len(mismatches)} shared residue(s) -- refusing to score "
      "misaligned data. Details:\n  " + "\n  ".join(mismatches)
    )
    raise ValueError(msg)

  shared_mask = resolved.all(axis=0)  # (n_canonical,) -- intersection across all states

  centered_scaled = np.empty_like(coords_canonical)
  for s in range(n_states):
    centroid = coords_canonical[s, shared_mask].mean(axis=0)
    centered_scaled[s] = (coords_canonical[s] - centroid) * coordinate_scaling

  return ConformationalStates(
    coords_states=jnp.asarray(centered_scaled),
    mask=jnp.asarray(shared_mask),
    reference_aatype=jnp.asarray(reference_aatype),
    residue_numbers=jnp.asarray(np.asarray(all_residue_numbers, dtype=np.int32)),
  )


def load_conformational_states(
  structure_paths: Sequence[str | Path],
  *,
  coordinate_scaling: float = DEFAULT_COORDINATE_SCALING,
) -> ConformationalStates:
  """Parse ``S`` PDB/mmCIF structures of one protein's conformational states and align them.

  Convenience wrapper: parses each path via ``aminx.io.parsing.parse_structure``
  (Atom37 format; CA = atom index 1, per ``aminx.ebm.contracts.Atom37``'s
  docstring), extracts CA coordinates + PDB residue numbering + aatype, then
  delegates to :func:`align_conformational_states` for the real alignment/
  verification logic.

  Args:
      structure_paths: ``S`` paths to structure files (typically 2 -- e.g.
          a "closed" and an "open" conformer of the same protein, LplA
          1x2g.pdb (closed) / 3a7r.pdb (open) in the design spec's case
          study -- verified against ``~/repos/ProteinEBM/notebooks/
          confbiasing.ipynb`` and ``~/repos/ProteinEBM/README.md``). Order
          is preserved into ``coords_states``'s leading axis.
      coordinate_scaling: Forwarded to :func:`align_conformational_states`.

  Returns:
      A :class:`ConformationalStates`, ready for :func:`score_conformational_bias`.

  """
  from aminx.io.parsing import parse_structure  # noqa: PLC0415 (defer proxide import)

  per_state_coords: list[np.ndarray] = []
  per_state_residue_index: list[np.ndarray] = []
  per_state_aatype: list[np.ndarray] = []
  for path in structure_paths:
    protein = parse_structure(path)
    per_state_coords.append(np.asarray(protein.coordinates)[:, 1, :])  # CA = atom37 index 1
    per_state_residue_index.append(np.asarray(protein.residue_index))
    per_state_aatype.append(np.asarray(protein.aatype))

  return align_conformational_states(
    per_state_coords,
    per_state_residue_index,
    per_state_aatype,
    coordinate_scaling=coordinate_scaling,
  )


def sequence_to_af_aatype(sequence: str) -> AAType:
  """Convert a one-letter amino-acid string to AF-alphabet ``aatype`` indices.

  Uses ``aminx.utils.aa_convert.AF_ALPHABET`` directly (**not**
  ``aminx.utils.aa_convert.string_to_protein_sequence``, which converts to
  ProteinMPNN's alphabet order -- the wrong convention here).
  ``aminx.io.parsing.parse_structure``'s own ``Protein.aatype`` and
  ProteinEBM's reference ``restypes`` (``~/repos/ProteinEBM/protein_ebm/data/
  protein_utils.py``) both use this exact AF order -- verified by decoding a
  parsed LplA structure's ``aatype`` back to a string and confirming it
  matches ``eval_data/lpla.csv``'s ``Sequence`` column at every
  non-mutated position across 200 spot-checked rows (see
  ``scripts/ebm/lpla_biasing_check.py``).

  Args:
      sequence: A one-letter amino-acid string. Characters outside
          ``AF_ALPHABET`` (unknown/non-standard residues) map to the ``X``
          index (20), matching ``parse_structure``'s own unknown-residue
          convention.

  Returns:
      ``(len(sequence),)`` int32 array of AF-alphabet indices in ``[0, 20]``.

  Raises:
      ValueError: If ``sequence`` is empty.

  """
  if len(sequence) == 0:
    msg = "sequence_to_af_aatype: sequence must be non-empty"
    raise ValueError(msg)
  unk_index = len(AF_ALPHABET) - 1  # 'X'
  indices = [AF_ALPHABET.index(c) if c in AF_ALPHABET else unk_index for c in sequence]
  return jnp.asarray(indices, dtype=jnp.int32)


def score_conformational_bias(
  model: ProteinEBMModel,
  states: ConformationalStates,
  aatype: AAType,
  t: DiffusionTime,
  *,
  difference_fuse: EnergyFusionFn = _default_difference_fuse,
  default_batch_size: int | None = None,
) -> Energy:
  """Score the conformational-biasing energy gap for one fixed sequence (backlog node **E7**).

  Thin wrapper over E4's :func:`aminx.ebm.dispatch.score_state_difference`
  that (a) validates ``aatype`` is aligned to ``states`` (the CRITICAL
  sequence/structure length check this node's brief requires -- raises
  rather than silently scoring misaligned data) and (b) forwards ``states``'s
  pre-aligned/centered/scaled ``coords_states``/``mask``.

  Args:
      model: The assembled, checkpoint-ported ``ProteinEBMModel``.
      states: A :class:`ConformationalStates` from :func:`load_conformational_states`
          / :func:`align_conformational_states`.
      aatype: The fixed mutant/WT sequence to score across all states, AF-
          alphabet order (see :func:`sequence_to_af_aatype`) -- must have the
          same length as ``states.mask``.
      t: Diffusion time (design spec's MVP target ``t=0.05``).
      difference_fuse: Forwarded to ``score_state_difference``. Default
          ``aminx.ebm.plan.difference_fuse`` computes
          ``per_state_energy[0] - per_state_energy[1]`` -- see this module's
          docstring for the ``[closed, open]`` stacking convention.
      default_batch_size: Forwarded to ``score_state_difference`` (Vmap/
          SafeMap cutoff override for the state axis).

  Returns:
      The fused scalar energy gap (default: ``E(state_0) - E(state_1)``).

  Raises:
      ValueError: If ``aatype``'s length does not match ``states.mask``'s
          length -- the sequence/structure alignment guard.

  """
  n_residues = states.mask.shape[0]
  if aatype.shape[0] != n_residues:
    msg = (
      "score_conformational_bias: aatype length "
      f"{aatype.shape[0]} does not match the aligned structures' residue "
      f"count {n_residues} -- refusing to score misaligned sequence/structure "
      "data. Did you forget to align the sequence to the SAME canonical "
      "residue numbering load_conformational_states used?"
    )
    raise ValueError(msg)

  return score_state_difference(
    model,
    states.coords_states,
    aatype,
    t,
    states.mask,
    difference_fuse=difference_fuse,
    default_batch_size=default_batch_size,
  )


__all__ = [
  "ConformationalStates",
  "align_conformational_states",
  "load_conformational_states",
  "score_conformational_bias",
  "sequence_to_af_aatype",
]
