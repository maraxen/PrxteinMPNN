"""Tests for gated side-chain-conditioned ligand context in PrxteinLigandMPNN."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.mpnn import PrxteinLigandMPNN
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
    "xyz_37": jnp.asarray(rng.normal(size=(seq_len, 37, 3)).astype(np.float32)),
    "xyz_37_m": jnp.ones((seq_len, 37), dtype=jnp.float32),
    "chain_mask": jnp.zeros((seq_len,), dtype=jnp.float32),
    "ar_mask": jnp.zeros((seq_len, seq_len), dtype=jnp.float32),
    "one_hot_sequence": jax.nn.one_hot(jnp.asarray(sequence), 21),
  }


def _build_model(
  *,
  key: jax.Array,
  ligand_mpnn_use_side_chain_context: bool,
) -> PrxteinLigandMPNN:
  """Construct a small deterministic ligand model for context-lane tests."""
  return PrxteinLigandMPNN(
    node_features=32,
    edge_features=32,
    hidden_features=32,
    num_encoder_layers=2,
    num_decoder_layers=2,
    k_neighbors=8,
    num_context_layers=2,
    dropout_rate=0.0,
    ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    key=key,
  )


def _run_conditional(
  model: PrxteinLigandMPNN,
  inputs: dict[str, jax.Array],
  *,
  y_m: jax.Array,
  include_side_chain_inputs: bool,
) -> tuple[jax.Array, jax.Array]:
  """Run conditional decoding with optional side-chain context tensors."""
  kwargs: dict[str, jax.Array] = {}
  if include_side_chain_inputs:
    kwargs = {
      "xyz_37": inputs["xyz_37"],
      "xyz_37_m": inputs["xyz_37_m"],
      "chain_mask": inputs["chain_mask"],
    }

  return model(
    inputs["structure_coordinates"],
    inputs["mask"],
    inputs["residue_index"],
    inputs["chain_index"],
    inputs["y"],
    inputs["y_t"],
    y_m,
    "conditional",
    prng_key=jax.random.PRNGKey(123),
    ar_mask=inputs["ar_mask"],
    one_hot_sequence=inputs["one_hot_sequence"],
    inference=True,
    **kwargs,
  )


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


def test_ligand_side_chain_gate_on_executes_context_lane() -> None:
  """Ensure side-chain lane requires side-chain inputs and produces usable outputs."""
  inputs = _synthetic_inputs()
  model = _build_model(
    key=jax.random.PRNGKey(1),
    ligand_mpnn_use_side_chain_context=True,
  )
  y_m = jnp.zeros_like(inputs["y_m"])

  with pytest.raises(ValueError, match="xyz_37 and xyz_37_m"):
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
    xyz_37=inputs["xyz_37"],
    xyz_37_m=inputs["xyz_37_m"],
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
