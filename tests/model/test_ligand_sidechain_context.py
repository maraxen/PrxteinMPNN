"""Tests for gated side-chain-conditioned ligand context in PrxteinLigandMPNN."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aminx.model.ligand_mpnn import PrxteinLigandMPNN
from scripts.convert_weights import resolve_ligand_side_chain_context


def _synthetic_inputs(*, seq_len: int = 8, ligand_atoms: int = 6) -> dict[str, jax.Array]:
  """Create deterministic synthetic inputs for ligand model tests."""
  rng = np.random.default_rng(7)
  sequence = rng.integers(0, 20, size=(seq_len,), dtype=np.int32)
  return {
    "structure_coordinates": jnp.asarray(rng.normal(size=(seq_len, 4, 3)).astype(np.float32)),
    "mask": jnp.ones((seq_len,), dtype=jnp.float32),
    "residue_index": jnp.arange(seq_len, dtype=jnp.int32),
    "chain_index": jnp.zeros((seq_len,), dtype=jnp.int32),
    "y": jnp.asarray(rng.normal(size=(seq_len, ligand_atoms, 3)).astype(np.float32)),
    "y_t": jnp.asarray(rng.integers(1, 30, size=(seq_len, ligand_atoms), dtype=np.int32)),
    "y_m": jnp.ones((seq_len, ligand_atoms), dtype=jnp.float32),
    "atom_37": jnp.asarray(rng.normal(size=(seq_len, 37, 3)).astype(np.float32)),
    "atom_37_mask": jnp.ones((seq_len, 37), dtype=jnp.float32),
    "chain_mask": jnp.zeros((seq_len,), dtype=jnp.float32),
    "ar_mask": jnp.zeros((seq_len, seq_len), dtype=jnp.float32),
    "one_hot_sequence": jax.nn.one_hot(jnp.asarray(sequence), 21),
  }


def _build_model(
  *,
  key: jax.Array,
  ligand_mpnn_use_side_chain_context: bool,
  k_neighbors: int = 8,
) -> PrxteinLigandMPNN:
  """Construct a small deterministic ligand model for context-lane tests."""
  return PrxteinLigandMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=k_neighbors,
    num_context_layers=2,
    dropout_rate=0.0,
    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    key=key,
  )


def _build_tie_group_map(seq_len: int, groups: list[list[int]]) -> jax.Array:
  """Build a dense tie-group map from grouped residue indices."""
  tie_group_map = np.arange(seq_len, dtype=np.int32)
  for positions in groups:
    representative = positions[0]
    for position in positions[1:]:
      tie_group_map[position] = representative
  _, compact_tie_group_map = np.unique(tie_group_map, return_inverse=True)
  return jnp.asarray(compact_tie_group_map.astype(np.int32, copy=False))


def _run_conditional(
  model: PrxteinLigandMPNN,
  inputs: dict[str, jax.Array],
  *,
  y_m: jax.Array,
  include_side_chain_inputs: bool,
) -> tuple[jax.Array, jax.Array]:
  """Run conditional decoding with optional side-chain context tensors."""
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.inference import score_conditional

  # Side-chain context (atom_37/atom_37_mask) is packaged onto the GeometryBundle via
  # build_inference_bundle (see #105), not passed as loose kernel kwargs.
  sc_atom_37 = inputs["atom_37"][None, ...] if include_side_chain_inputs else None
  sc_atom_37_mask = inputs["atom_37_mask"][None, ...] if include_side_chain_inputs else None

  # Build inference bundle with ligand coordinates and ligand mask
  bundle, config = build_inference_bundle(
    coords=inputs["structure_coordinates"][None, ...],  # Add batch dim
    mask=inputs["mask"][None, ...],
    residue_index=inputs["residue_index"][None, ...],
    chain_index=inputs["chain_index"][None, ...],
    ligand_coords=inputs["y"][None, ...],
    ligand_atom_types=inputs["y_t"][None, ...],
    ligand_mask=y_m[None, ...],
    atom_37=sc_atom_37,
    atom_37_mask=sc_atom_37_mask,
    ar_mask=inputs["ar_mask"][None, ...],
    sequence=inputs["one_hot_sequence"],
    mode="score_conditional",
  )
  stage_set = make_stage_set()

  # Run conditional scoring kernel
  logits = score_conditional.kernel(model, jax.random.PRNGKey(123), bundle, config, stage_set)

  # Return sequence (one-hot) and logits for compatibility with previous API
  return inputs["one_hot_sequence"], logits


@pytest.mark.parity_heavy
def test_ligand_side_chain_gate_off_preserves_default_path() -> None:
  """Ensure side-chain tensors do not affect outputs when gate is disabled."""
  inputs = _synthetic_inputs()
  model = _build_model(
    key=jax.random.PRNGKey(0),
    ligand_mpnn_use_side_chain_context=False,
  )
  y_m = jnp.zeros_like(inputs["y_m"])

  sequence_default, logits_default = _run_conditional(
    model,
    inputs,
    y_m=y_m,
    include_side_chain_inputs=False,
  )
  sequence_with_sidechain, logits_with_sidechain = _run_conditional(
    model,
    inputs,
    y_m=y_m,
    include_side_chain_inputs=True,
  )

  np.testing.assert_allclose(np.asarray(sequence_default), np.asarray(sequence_with_sidechain))
  np.testing.assert_allclose(np.asarray(logits_default), np.asarray(logits_with_sidechain))


@pytest.mark.parity_heavy
def test_ligand_side_chain_gate_on_executes_context_lane() -> None:
  """Ensure side-chain lane requires side-chain inputs and produces usable outputs."""
  inputs = _synthetic_inputs()
  model = _build_model(
    key=jax.random.PRNGKey(1),
    ligand_mpnn_use_side_chain_context=True,
  )
  y_m = jnp.zeros_like(inputs["y_m"])

  with pytest.raises(ValueError, match="atom_37 and atom_37_mask"):
    _run_conditional(
      model,
      inputs,
      y_m=y_m,
      include_side_chain_inputs=False,
    )

  _, _, _, _, _, y_m_out = model.features(
    jax.random.PRNGKey(5),
    inputs["structure_coordinates"],
    inputs["mask"],
    inputs["residue_index"],
    inputs["chain_index"],
    inputs["y"],
    inputs["y_t"],
    y_m,
    atom_37=inputs["atom_37"],
    atom_37_mask=inputs["atom_37_mask"],
    chain_mask=inputs["chain_mask"],
  )
  assert float(jnp.sum(y_m_out)) > 0.0

  sequence, logits = _run_conditional(
    model,
    inputs,
    y_m=y_m,
    include_side_chain_inputs=True,
  )
  assert sequence.shape == inputs["one_hot_sequence"].shape
  assert logits.shape == inputs["one_hot_sequence"].shape
  assert bool(jnp.all(jnp.isfinite(logits)))


@pytest.mark.parity_heavy
def test_ligand_tied_autoregressive_support_without_sidechain_context() -> None:
  """Ensure ligand autoregressive tied decoding enforces per-group token consistency."""
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.inference import sample_autoregressive

  inputs = _synthetic_inputs(seq_len=10, ligand_atoms=8)
  model = _build_model(
    key=jax.random.PRNGKey(11),
    ligand_mpnn_use_side_chain_context=False,
  )
  tie_groups = [[0, 1, 2], [3, 4]]
  tie_group_map = _build_tie_group_map(seq_len=10, groups=tie_groups)

  forced_tokens = np.arange(10, dtype=np.int32) % 20
  bias = np.zeros((10, 21), dtype=np.float32)
  bias[np.arange(10), forced_tokens] = 45.0

  def _sample_with_bundle(use_tie_groups: bool) -> np.ndarray:
    # Tying is wired through build_inference_bundle's tie_group_map; the AR
    # kernel/driver generates the wave schedule and ar_mask from it (see
    # bundle_builder: ar_mask/wave are placeholders for sampling mode).
    bundle, config = build_inference_bundle(
      coords=inputs["structure_coordinates"][None, ...],
      mask=inputs["mask"][None, ...],
      residue_index=inputs["residue_index"][None, ...],
      chain_index=inputs["chain_index"][None, ...],
      ligand_coords=inputs["y"][None, ...],
      ligand_atom_types=inputs["y_t"][None, ...],
      ligand_mask=inputs["y_m"][None, ...],
      tie_group_map=jnp.asarray(tie_group_map) if use_tie_groups else None,
      bias=jnp.asarray(bias),
      temperature=1.0,
      mode="sample_autoregressive",
    )
    stage_set = make_stage_set()
    result = sample_autoregressive.kernel(model, jax.random.PRNGKey(13), bundle, config, stage_set)
    # result.sequence is integer token ids (S, L) or (L,); collapse the single state.
    tokens = np.asarray(result.sequence)
    return tokens[0] if tokens.ndim > 1 else tokens

  no_tie_tokens = _sample_with_bundle(use_tie_groups=False)
  for group in tie_groups:
    assert len(np.unique(no_tie_tokens[group])) > 1

  tied_tokens = _sample_with_bundle(use_tie_groups=True)
  # When tied, positions in the same group should have the same token
  for group in tie_groups:
    unique_tokens = np.unique(tied_tokens[group])
    assert len(unique_tokens) == 1, f"Expected 1 unique token in group {group}, got {len(unique_tokens)}: {unique_tokens}"


@pytest.mark.parity_heavy
def test_ligand_tied_autoregressive_support_with_sidechain_context() -> None:
  """Ensure side-chain-conditioned ligand tied decoding remains group-consistent."""
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.inference import sample_autoregressive

  inputs = _synthetic_inputs(seq_len=10, ligand_atoms=8)
  model = _build_model(
    key=jax.random.PRNGKey(17),
    ligand_mpnn_use_side_chain_context=True,
  )
  tie_groups = [[0, 1, 2], [3, 4]]
  tie_group_map = _build_tie_group_map(seq_len=10, groups=tie_groups)
  forced_tokens = np.arange(10, dtype=np.int32) % 20
  bias = np.zeros((10, 21), dtype=np.float32)
  bias[np.arange(10), forced_tokens] = 45.0

  # Side-chain context (atom_37/atom_37_mask) is packaged onto the GeometryBundle via
  # build_inference_bundle, not injected as loose kernel kwargs; tying is wired
  # through tie_group_map (kernel/driver generates wave + ar_mask).
  bundle, config = build_inference_bundle(
    coords=inputs["structure_coordinates"][None, ...],
    mask=inputs["mask"][None, ...],
    residue_index=inputs["residue_index"][None, ...],
    chain_index=inputs["chain_index"][None, ...],
    ligand_coords=inputs["y"][None, ...],
    ligand_atom_types=inputs["y_t"][None, ...],
    ligand_mask=inputs["y_m"][None, ...],
    atom_37=inputs["atom_37"][None, ...],
    atom_37_mask=inputs["atom_37_mask"][None, ...],
    tie_group_map=jnp.asarray(tie_group_map),
    bias=jnp.asarray(bias),
    temperature=1.0,
    mode="sample_autoregressive",
  )
  stage_set = make_stage_set()
  result = sample_autoregressive.kernel(model, jax.random.PRNGKey(19), bundle, config, stage_set)
  tokens = np.asarray(result.sequence)
  tied_tokens = tokens[0] if tokens.ndim > 1 else tokens
  for group in tie_groups:
    assert np.all(tied_tokens[group] == tied_tokens[group[0]])
  assert bool(jnp.all(jnp.isfinite(result.logits)))


@pytest.mark.parity_heavy
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Conditional (teacher-forced) scoring does not fuse logits across tied "
        "positions: tie_group_fuse (TieGroupProductOfExperts) is wired only into the "
        "AR and STE decode paths (autoregressive.py, ste.py). ConditionalDecode fuses "
        "across conformational STATES via logit_transform, not across tied POSITIONS. "
        "So per-group-identical conditional logits is an unimplemented feature, not a "
        "passing invariant. If conditional tie-group fusion is added, this XPASSes and "
        "flags the test to assert it. See tech-debt: conditional-tie-group-fusion."
    ),
)
def test_ligand_conditional_multistate_logits_are_group_shared() -> None:
  """Conditional scoring would combine logits identically per tied group (unimplemented)."""
  from aminx.inference.bundle_builder import build_inference_bundle
  from aminx.inference.logits import make_stage_set
  from aminx.inference import score_conditional

  inputs = _synthetic_inputs(seq_len=10, ligand_atoms=8)
  model = _build_model(
    key=jax.random.PRNGKey(23),
    ligand_mpnn_use_side_chain_context=False,
  )
  tie_groups = [[0, 1, 2], [3, 4]]
  tie_group_map = _build_tie_group_map(seq_len=10, groups=tie_groups)

  bundle, config = build_inference_bundle(
    coords=inputs["structure_coordinates"][None, ...],
    mask=inputs["mask"][None, ...],
    residue_index=inputs["residue_index"][None, ...],
    chain_index=inputs["chain_index"][None, ...],
    ligand_coords=inputs["y"][None, ...],
    ligand_atom_types=inputs["y_t"][None, ...],
    ligand_mask=inputs["y_m"][None, ...],
    tie_group_map=jnp.asarray(tie_group_map),
    sequence=inputs["one_hot_sequence"],
    mode="score_conditional",
  )
  stage_set = make_stage_set()
  tied_logits = score_conditional.kernel(model, jax.random.PRNGKey(29), bundle, config, stage_set)
  tied_logits_np = np.asarray(tied_logits)
  for group in tie_groups:
    np.testing.assert_allclose(
      tied_logits_np[group],
      np.repeat(tied_logits_np[group[0]][None, :], repeats=len(group), axis=0),
      rtol=1e-5,
      atol=1e-5,
    )


def test_ligand_features_structure_mapping_masks_cross_state_neighbors() -> None:
  """Ensure ligand feature KNN never crosses structure boundaries when mapping is provided."""
  model = _build_model(
    key=jax.random.PRNGKey(31),
    ligand_mpnn_use_side_chain_context=False,
    k_neighbors=3,
  )
  seq_len = 8
  ligand_atoms = 6
  base_inputs = _synthetic_inputs(seq_len=seq_len, ligand_atoms=ligand_atoms)
  coords = np.array(base_inputs["structure_coordinates"], copy=True)
  coords[4:] = coords[:4]

  _, _, e_idx_nomap, *_ = model.features(
    jax.random.PRNGKey(37),
    jnp.asarray(coords),
    base_inputs["mask"],
    base_inputs["residue_index"],
    base_inputs["chain_index"],
    base_inputs["y"],
    base_inputs["y_t"],
    base_inputs["y_m"],
  )
  no_map_crossing = any(
    (int(i) < 4 and np.any(np.asarray(e_idx_nomap)[int(i)] >= 4))
    or (int(i) >= 4 and np.any(np.asarray(e_idx_nomap)[int(i)] < 4))
    for i in range(seq_len)
  )
  assert no_map_crossing

  structure_mapping = jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=jnp.int32)
  _, _, e_idx_mapped, *_ = model.features(
    jax.random.PRNGKey(37),
    jnp.asarray(coords),
    base_inputs["mask"],
    base_inputs["residue_index"],
    base_inputs["chain_index"],
    base_inputs["y"],
    base_inputs["y_t"],
    base_inputs["y_m"],
    structure_mapping=structure_mapping,
  )
  e_idx_mapped_np = np.asarray(e_idx_mapped)
  for i in range(seq_len):
    if i < 4:
      assert np.all(e_idx_mapped_np[i] < 4)
    else:
      assert np.all(e_idx_mapped_np[i] >= 4)


@pytest.mark.parametrize(
  ("mode", "checkpoint_payload", "input_path", "expected"),
  [
    ("on", None, "ligandmpnn_v_32_020_25.pt", True),
    ("off", {"ligand_mpnn_use_side_chain_context": True}, "ligandmpnn_v_32_020_25.pt", False),
    ("auto", {"ligand_mpnn_use_side_chain_context": True}, "ligandmpnn_v_32_020_25.pt", True),
    ("auto", None, "ligandmpnn_side_chain_context_v1.pt", True),
    ("auto", None, "ligandmpnn_sc_v_32_020_25.pt", True),
    ("auto", None, "ligandmpnn_v_32_020_25.pt", False),
  ],
)
def test_resolve_ligand_side_chain_context_mode(
  mode: str,
  checkpoint_payload: dict[str, object] | None,
  input_path: str,
  expected: bool,
) -> None:
  """Validate conversion-time side-chain context mode resolution."""
  observed = resolve_ligand_side_chain_context(
    mode,
    checkpoint_payload=checkpoint_payload,
    input_path=input_path,
  )
  assert observed is expected
