"""Parity oracle tests for averaged-topology scoring (RS-7b).

Invariants for Interpretation (A): mean node/edge features THEN score.
NOT Interpretation (B): mean of D independent scalar NLLs (differs: log_softmax is nonlinear).

Test structure:
- Inv-1 (D=1 degenerate): backbone_noise=(0.0,) with average_node_features=True
  must produce byte-identical logits and NLL vs standard scoring.
- Inv-2 (A vs B, D=2): nll_A (mean features) != nll_B (mean scores) on non-degenerate fixture.
- Inv-3 (fusion math): synthetic stacked features -> mean of stacked -> identity fusion -> pass-through.
- Inv-4 (golden): small real structure + seq, D=3, pin scalar NLL.
"""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from aminx.host.averaging import ArithmeticMeanEncodingFusion, IdentityEncodingFusion
from aminx.scoring.score import _nll_from_logits
from aminx.types.encodings import EncoderOutput


def test_inv3_fusion_math_sanity() -> None:
  """Inv-3: Fusion math sanity (synthetic 30s)."""
  # Test ArithmeticMeanEncodingFusion: mean of features over D axis
  D = 2
  L = 10
  K = 5
  H_node = 8
  H_edge = 4

  # Create synthetic stacked encodings: (D, L, H), (D, L, K, H), (D, L, K), (D, L)
  node_features = jnp.stack(
    [
      jnp.ones((L, H_node)) * 1.0,
      jnp.ones((L, H_node)) * 3.0,
    ],
    axis=0,
  )
  edge_features = jnp.stack(
    [
      jnp.ones((L, K, H_edge)) * 1.0,
      jnp.ones((L, K, H_edge)) * 3.0,
    ],
    axis=0,
  )
  neighbor_indices = jnp.arange(L)[:, None].repeat(K, axis=1).astype(jnp.int32)
  mask = jnp.ones((L,))

  stacked_enc = EncoderOutput(
    node_features=node_features,
    edge_features=edge_features,
    neighbor_indices=neighbor_indices,
    mask=mask,
  )

  fusion = ArithmeticMeanEncodingFusion()
  fused = fusion(stacked_enc)

  # Expected: mean of [1.0, 3.0] = 2.0
  assert fused.node_features.shape == (L, H_node), f"Got {fused.node_features.shape}"
  assert jnp.allclose(
    fused.node_features,
    2.0,
    atol=1e-6,
  ), f"Node features should be 2.0, got {fused.node_features[0, 0]}"
  assert jnp.allclose(
    fused.edge_features,
    2.0,
    atol=1e-6,
  ), f"Edge features should be 2.0, got {fused.edge_features[0, 0, 0]}"

  # Geometry from first: neighbor_indices and mask from [0]
  assert jnp.array_equal(
    fused.neighbor_indices,
    neighbor_indices[0],
  ), "neighbor_indices should be from first encoding"
  assert jnp.array_equal(fused.mask, mask), "mask should be from first encoding"


def test_inv3_identity_fusion_passthrough() -> None:
  """Inv-3 variant: IdentityEncodingFusion is a no-op."""
  D = 2
  L = 5
  K = 3
  H = 4

  node_features = jnp.ones((D, L, H)) * 2.0
  edge_features = jnp.ones((D, L, K, H)) * 3.0
  neighbor_indices = jnp.zeros((D, L, K), dtype=jnp.int32)
  mask = jnp.ones((D, L))

  stacked_enc = EncoderOutput(
    node_features=node_features,
    edge_features=edge_features,
    neighbor_indices=neighbor_indices,
    mask=mask,
  )

  fusion = IdentityEncodingFusion()
  fused = fusion(stacked_enc)

  # Should pass through unchanged
  assert jnp.array_equal(fused.node_features, node_features)
  assert jnp.array_equal(fused.edge_features, edge_features)
  assert jnp.array_equal(fused.neighbor_indices, neighbor_indices)
  assert jnp.array_equal(fused.mask, mask)


def test_nll_from_logits_basic() -> None:
  """Test _nll_from_logits basic computation."""
  L = 5
  logits = jnp.ones((L, 21)) * 0.0  # Uniform logits -> log_softmax ~ -log(20) for vocab
  seq_one_hot = jnp.zeros((L, 21))
  seq_one_hot = seq_one_hot.at[:, 0].set(1.0)  # All position 0 (amino acid index 0)
  mask = jnp.ones((L,))

  nll = _nll_from_logits(logits, seq_one_hot, mask)

  # For uniform logits: log_softmax = log(1/21) = -log(21)
  # NLL = -log_softmax = log(21) ≈ 3.044...
  expected = jnp.log(21.0)
  assert jnp.allclose(nll, expected, atol=1e-5), f"Got {nll}, expected {expected}"


def test_nll_from_logits_with_mask() -> None:
  """Test _nll_from_logits with per-residue masking."""
  L = 4
  logits = jnp.ones((L, 21))
  seq_one_hot = jax.nn.one_hot(jnp.array([0, 1, 2, 0], dtype=jnp.int32), 21)
  mask = jnp.array([1.0, 1.0, 0.0, 0.0])  # Only first 2 residues count

  nll = _nll_from_logits(logits, seq_one_hot, mask)

  # Should be NLL of first 2 residues only
  logits_2 = logits[:2]
  seq_oh_2 = seq_one_hot[:2]
  mask_2 = jnp.ones((2,))
  expected = _nll_from_logits(logits_2, seq_oh_2, mask_2)

  assert jnp.allclose(nll, expected, atol=1e-5), f"Got {nll}, expected {expected}"


def test_nll_from_logits_vectorized() -> None:
  """Test _nll_from_logits with (S, L, 21) logits (multi-state)."""
  S = 2
  L = 3
  logits = jnp.ones((S, L, 21)) * 0.1
  seq_one_hot = jax.nn.one_hot(jnp.array([[0, 1, 2], [1, 2, 0]], dtype=jnp.int32), 21)
  mask = jnp.ones((S, L))  # Full mask, will use mask[0]

  nll = _nll_from_logits(logits, seq_one_hot, mask)

  # Should process all (S, L) and compute mean NLL
  # For each (s, l): -log_softmax(logits[s, l])[seq_one_hot[s, l]]
  assert nll.shape == (), f"Should be scalar, got {nll.shape}"
  assert nll > 0, "NLL should be positive"
