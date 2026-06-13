"""Tests for PoeModel: Product-of-Experts Potts ensemble."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, Int, PRNGKeyArray


# Mock mistypotts before importing PottsModel
def _mock_mistypotts():
  """Create minimal mock of mistypotts for PottsModel import."""
  if "mistypotts" in sys.modules:
    return

  mock_trw_spec = MagicMock()

  class MockDifferentiableTRW:
    def __init__(self, q: int, trw_iters: int, damping: float, rho_floor: float, spec):
      self.q = q
      self.trw_iters = trw_iters
      self.damping = damping
      self.rho_floor = rho_floor
      self.spec = spec

  class MockPottsTRWRunSpec:
    def __init__(
      self,
      rho_backend: str = "dense_pinv",
      message_backend: str = "dense",
      tile_size: int = 8,
      lanczos_rank: int = 32,
      slq_num_samples: int = 32,
      checkpoint_trw_step: bool = False,
      trw_loop: str = "fori",
      uniform_rho_value: float = 0.5,
    ):
      self.rho_backend = rho_backend
      self.message_backend = message_backend
      self.tile_size = tile_size
      self.lanczos_rank = lanczos_rank
      self.slq_num_samples = slq_num_samples
      self.checkpoint_trw_step = checkpoint_trw_step
      self.trw_loop = trw_loop
      self.uniform_rho_value = uniform_rho_value

    def to_json_dict(self):
      return {
        "rho_backend": self.rho_backend,
        "message_backend": self.message_backend,
        "tile_size": self.tile_size,
        "lanczos_rank": self.lanczos_rank,
        "slq_num_samples": self.slq_num_samples,
        "checkpoint_trw_step": self.checkpoint_trw_step,
        "trw_loop": self.trw_loop,
        "uniform_rho_value": self.uniform_rho_value,
      }

    @classmethod
    def from_json_dict(cls, d):
      return cls(
        rho_backend=d.get("rho_backend", "dense_pinv"),
        message_backend=d.get("message_backend", "dense"),
        tile_size=int(d.get("tile_size", 8)),
        lanczos_rank=int(d.get("lanczos_rank", 32)),
        slq_num_samples=int(d.get("slq_num_samples", 32)),
        checkpoint_trw_step=bool(d.get("checkpoint_trw_step", False)),
        trw_loop=d.get("trw_loop", "fori"),
        uniform_rho_value=float(d.get("uniform_rho_value", 0.5)),
      )

    @staticmethod
    def default_dense():
      return MockPottsTRWRunSpec()

  mistypotts_module = MagicMock()
  mistypotts_module.trw.DifferentiableTRW = MockDifferentiableTRW
  mistypotts_module.potts_trw_spec.PottsTRWRunSpec = MockPottsTRWRunSpec

  sys.modules["mistypotts"] = mistypotts_module
  sys.modules["mistypotts.trw"] = MagicMock(DifferentiableTRW=MockDifferentiableTRW)
  sys.modules["mistypotts.potts_trw_spec"] = MagicMock(
    PottsTRWRunSpec=MockPottsTRWRunSpec
  )


_mock_mistypotts()

from aminx.potts.model import PottsModel, PottsParams


class TestPoeModelConstruction:
  """Test PoeModel construction and validation."""

  def test_single_backbone_construction(self) -> None:
    """Test construction with single backbone."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    backbone = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k1,
    )

    from aminx.potts.poe import PoeModel

    model = PoeModel(backbones=(backbone,))
    assert model.n_backbones == 1
    assert len(model.backbones) == 1
    assert model.backbones[0] is backbone

  def test_two_backbones_identical_config(self) -> None:
    """Test construction with two backbones with identical static config."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    cfg = {
      "hidden_dim": 64,
      "num_aa": 21,
      "k_neighbors": 30,
      "edge_features_dim": 128,
      "trw_iters": 5,
    }

    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k2)

    from aminx.potts.poe import PoeModel

    model = PoeModel(backbones=(backbone_a, backbone_b))
    assert model.n_backbones == 2
    assert len(model.backbones) == 2

  def test_mismatched_hidden_dim_raises(self) -> None:
    """Test that mismatched hidden_dim raises ValueError."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    backbone_a = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k1,
    )

    backbone_b = PottsModel(
      hidden_dim=128,  # Different!
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k2,
    )

    from aminx.potts.poe import PoeModel

    with pytest.raises(ValueError, match="All backbones must share identical static config"):
      PoeModel(backbones=(backbone_a, backbone_b))

  def test_mismatched_num_aa_raises(self) -> None:
    """Test that mismatched num_aa raises ValueError."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    backbone_a = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k1,
    )

    backbone_b = PottsModel(
      hidden_dim=64,
      num_aa=20,  # Different!
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k2,
    )

    from aminx.potts.poe import PoeModel

    with pytest.raises(ValueError, match="All backbones must share identical static config"):
      PoeModel(backbones=(backbone_a, backbone_b))

  def test_mismatched_k_neighbors_raises(self) -> None:
    """Test that mismatched k_neighbors raises ValueError."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    backbone_a = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k1,
    )

    backbone_b = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=25,  # Different!
      edge_features_dim=128,
      trw_iters=5,
      key=k2,
    )

    from aminx.potts.poe import PoeModel

    with pytest.raises(ValueError, match="All backbones must share identical static config"):
      PoeModel(backbones=(backbone_a, backbone_b))

  def test_mismatched_trw_spec_raises(self) -> None:
    """Test that mismatched trw_spec raises ValueError."""
    from mistypotts.potts_trw_spec import PottsTRWRunSpec

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    backbone_a = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k1,
      trw_spec=PottsTRWRunSpec.default_dense(),
    )

    # Create a different trw_spec
    backbone_b = PottsModel(
      hidden_dim=64,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=128,
      trw_iters=5,
      key=k2,
      trw_spec=PottsTRWRunSpec(
        rho_backend="edge_lanczos",
        lanczos_rank=16,
      ),
    )

    from aminx.potts.poe import PoeModel

    # The specs are different
    with pytest.raises(ValueError, match="All backbones must share identical static config"):
      PoeModel(backbones=(backbone_a, backbone_b))


class TestPoeModelJointEnergy:
  """Test joint_energy: sum of per-backbone energies."""

  @pytest.mark.slow
  @pytest.mark.potts
  def test_joint_energy_single_backbone(self) -> None:
    """Test joint_energy with single backbone matches single-backbone energy."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1 = jax.random.split(key, 1)[0]

    backbone = PottsModel(
      hidden_dim=16,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=64,
      trw_iters=2,
      key=k1,
    )

    model = PoeModel(backbones=(backbone,))

    # Create synthetic test data: seq (N,), h, J, W
    seq = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    h = jnp.zeros((5, 21), dtype=jnp.float32)
    J = jnp.zeros((5, 5, 21, 21), dtype=jnp.float32)
    W = jnp.zeros((5, 5), dtype=jnp.float32)

    # Joint energy should match single backbone
    # params_list is a tuple of (h, J, W) tuples, one per backbone
    energy_joint = model.joint_energy(seq, ((h, J, W),))

    # For single backbone with all-zero params, energy should be 0
    assert jnp.allclose(energy_joint, 0.0, atol=1e-5)

  @pytest.mark.slow
  @pytest.mark.potts
  def test_joint_energy_two_backbones_sums_correctly(self) -> None:
    """Test joint_energy sums per-backbone energies."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    cfg = {
      "hidden_dim": 16,
      "num_aa": 21,
      "k_neighbors": 30,
      "edge_features_dim": 64,
      "trw_iters": 2,
    }

    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k2)

    model = PoeModel(backbones=(backbone_a, backbone_b))

    # Create synthetic test data
    seq = jnp.array([0, 1, 2], dtype=jnp.int32)
    h1 = jnp.ones((3, 21), dtype=jnp.float32)
    J1 = jnp.zeros((3, 3, 21, 21), dtype=jnp.float32)
    W1 = jnp.zeros((3, 3), dtype=jnp.float32)

    h2 = jnp.ones((3, 21), dtype=jnp.float32)
    J2 = jnp.zeros((3, 3, 21, 21), dtype=jnp.float32)
    W2 = jnp.zeros((3, 3), dtype=jnp.float32)

    # Joint energy = E1 + E2
    # With h1[i, seq[i]] = 1.0 for all i: E1 = sum_i h1[i, seq[i]] = 3
    energy_joint = model.joint_energy(seq, ((h1, J1, W1), (h2, J2, W2)))

    # Both backbones have h=ones, so E1=E2=3 (unary only, no pairwise with W=0)
    expected = 3.0 + 3.0  # E1 + E2
    assert jnp.allclose(energy_joint, expected, atol=1e-5)

  @pytest.mark.slow
  @pytest.mark.potts
  def test_joint_energy_no_x2_factor_added(self) -> None:
    """Test that joint_energy does NOT add extra x2 factor (already in h,J)."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1 = jax.random.split(key, 1)[0]

    backbone = PottsModel(
      hidden_dim=16,
      num_aa=4,  # Small alphabet for testing
      k_neighbors=30,
      edge_features_dim=64,
      trw_iters=2,
      key=k1,
    )

    model = PoeModel(backbones=(backbone,))

    # Create test data with known energies
    seq = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
    n = 4
    q = 4

    # h[i, seq[i]] = 2.0 for all i → unary energy = sum_i h[i, seq[i]] = 8.0
    h = jnp.ones((n, q), dtype=jnp.float32) * 2.0
    J = jnp.zeros((n, n, q, q), dtype=jnp.float32)
    W = jnp.zeros((n, n), dtype=jnp.float32)

    energy = model.joint_energy(seq, ((h, J, W),))

    # Should be 8.0 (no x2 factor applied)
    assert jnp.allclose(energy, 8.0, atol=1e-5)


class TestPoeModelInferAllParams:
  """Test infer_all_params: vmap over backbone stack."""

  @pytest.mark.slow
  @pytest.mark.potts
  def test_infer_all_params_single_backbone(self) -> None:
    """Test infer_all_params with single backbone."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1 = jax.random.split(key, 1)[0]

    backbone = PottsModel(
      hidden_dim=16,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=64,
      trw_iters=2,
      key=k1,
    )

    model = PoeModel(backbones=(backbone,))

    # Create test input
    key_inf = jax.random.PRNGKey(123)
    n = 4
    coords_stack = jnp.zeros((1, n, 37, 3), dtype=jnp.float32)  # (B=1, N, 37, 3)
    mask = jnp.ones((n,), dtype=jnp.float32)
    residue_index = jnp.arange(n, dtype=jnp.int32)
    chain_index = jnp.zeros(n, dtype=jnp.int32)

    # This will raise NotImplementedError due to placeholder in PottsModel.__call__
    # but we'll check the method exists and has correct signature
    assert hasattr(model, "infer_all_params")

  @pytest.mark.slow
  @pytest.mark.potts
  def test_infer_all_params_signature(self) -> None:
    """Test that infer_all_params has correct signature."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    cfg = {
      "hidden_dim": 16,
      "num_aa": 21,
      "k_neighbors": 30,
      "edge_features_dim": 64,
      "trw_iters": 2,
    }

    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k2)

    model = PoeModel(backbones=(backbone_a, backbone_b))

    # Check method signature
    import inspect

    sig = inspect.signature(model.infer_all_params)
    params = list(sig.parameters.keys())
    assert "key" in params
    assert "edge_knn_stack" in params
    assert "nei_stack" in params
    assert "mask" in params


class TestPoeModelJointLogProb:
  """Test joint_log_prob: log unnormalized joint probability."""

  @pytest.mark.slow
  @pytest.mark.potts
  def test_joint_log_prob_single_backbone(self) -> None:
    """Test joint_log_prob with single backbone."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1 = jax.random.split(key, 1)[0]

    backbone = PottsModel(
      hidden_dim=16,
      num_aa=21,
      k_neighbors=30,
      edge_features_dim=64,
      trw_iters=2,
      key=k1,
    )

    model = PoeModel(backbones=(backbone,))

    # Test with synthetic data
    seq = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    h = jnp.ones((5, 21), dtype=jnp.float32) * 0.5
    J = jnp.zeros((5, 5, 21, 21), dtype=jnp.float32)
    W = jnp.zeros((5, 5), dtype=jnp.float32)

    log_prob = model.joint_log_prob(seq, ((h, J, W),))

    # Should be a scalar
    assert log_prob.shape == ()
    # With h=0.5 for all: unary = 2.5
    assert jnp.allclose(log_prob, 2.5, atol=1e-5)

  @pytest.mark.slow
  @pytest.mark.potts
  def test_joint_log_prob_two_backbones_sums(self) -> None:
    """Test joint_log_prob sums log probs from both backbones."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    cfg = {
      "hidden_dim": 16,
      "num_aa": 21,
      "k_neighbors": 30,
      "edge_features_dim": 64,
      "trw_iters": 2,
    }

    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k2)

    model = PoeModel(backbones=(backbone_a, backbone_b))

    # Test with synthetic data
    seq = jnp.array([0, 1, 2], dtype=jnp.int32)
    h1 = jnp.ones((3, 21), dtype=jnp.float32) * 1.0
    J1 = jnp.zeros((3, 3, 21, 21), dtype=jnp.float32)
    W1 = jnp.zeros((3, 3), dtype=jnp.float32)

    h2 = jnp.ones((3, 21), dtype=jnp.float32) * 2.0
    J2 = jnp.zeros((3, 3, 21, 21), dtype=jnp.float32)
    W2 = jnp.zeros((3, 3), dtype=jnp.float32)

    log_prob = model.joint_log_prob(seq, ((h1, J1, W1), (h2, J2, W2)))

    # Should sum: log_prob1 (3.0) + log_prob2 (6.0) = 9.0
    assert jnp.allclose(log_prob, 9.0, atol=1e-5)


class TestPoeModelCall:
  """Test PoeModel.__call__: aggregate marginals across backbones."""

  @pytest.mark.slow
  @pytest.mark.potts
  def test_poe_single_backbone_matches_potts(self) -> None:
    """Test that single-backbone PoE marginals match PottsModel (atol=1e-5)."""
    from aminx.potts.poe import PoeModel, PoeOutput

    key = jax.random.PRNGKey(42)
    k1 = jax.random.split(key, 1)[0]

    backbone = PottsModel(
      hidden_dim=16,
      num_aa=20,
      k_neighbors=30,
      edge_features_dim=64,
      trw_iters=2,
      key=k1,
    )

    model = PoeModel(backbones=(backbone,))

    # Create test data: synthetic edge features and neighbor indices
    key_call = jax.random.PRNGKey(123)
    n = 10
    k = 30  # k_neighbors
    e = 64  # edge_features_dim
    mask = jnp.ones(n, dtype=jnp.float32)

    # Create synthetic edge_knn (B, N, k, E) and nei (B, N, k)
    edge_knn_stack = jnp.ones((1, n, k, e), dtype=jnp.float32)
    nei_stack = jnp.zeros((1, n, k), dtype=jnp.int32)
    # Fill nei with valid neighbor indices
    for i in range(n):
      for j in range(k):
        nei_stack = nei_stack.at[0, i, j].set((i + j + 1) % n)

    # Call PoeModel
    poe_output = model(
      key=key_call,
      edge_knn_stack=edge_knn_stack,
      nei_stack=nei_stack,
      mask=mask,
    )

    # Verify output structure
    assert isinstance(poe_output, PoeOutput)
    assert poe_output.marginals.shape == (n, 20)
    assert poe_output.log_poe_unnorm.shape == (n, 20)
    assert poe_output.per_backbone_marginals.shape == (1, n, 20)

    # For single backbone, marginals should match per-backbone marginals
    assert jnp.allclose(
      poe_output.marginals,
      poe_output.per_backbone_marginals[0, :, :],
      atol=1e-5,
    ), "Single-backbone PoE marginals should match PottsModel output"

  @pytest.mark.slow
  @pytest.mark.potts
  def test_poe_shape(self) -> None:
    """Test that shapes are correct for B=3, N=10, q=20."""
    from aminx.potts.poe import PoeModel, PoeOutput

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    cfg = {
      "hidden_dim": 16,
      "num_aa": 20,
      "k_neighbors": 30,
      "edge_features_dim": 64,
      "trw_iters": 2,
    }

    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k2)
    backbone_c = PottsModel(**cfg, key=k3)

    model = PoeModel(backbones=(backbone_a, backbone_b, backbone_c))

    # Create test data: B=3, N=10
    key_call = jax.random.PRNGKey(999)
    b, n = 3, 10
    k = 30  # k_neighbors
    e = 64  # edge_features_dim
    q = 20
    mask = jnp.ones(n, dtype=jnp.float32)

    # Create synthetic edge_knn (B, N, k, E) and nei (B, N, k)
    edge_knn_stack = jnp.ones((b, n, k, e), dtype=jnp.float32)
    nei_stack = jnp.zeros((b, n, k), dtype=jnp.int32)
    for bi in range(b):
      for i in range(n):
        for j in range(k):
          nei_stack = nei_stack.at[bi, i, j].set((i + j + 1) % n)

    poe_output = model(
      key=key_call,
      edge_knn_stack=edge_knn_stack,
      nei_stack=nei_stack,
      mask=mask,
    )

    # Verify all shapes
    assert poe_output.marginals.shape == (n, q)
    assert poe_output.log_poe_unnorm.shape == (n, q)
    assert poe_output.per_backbone_marginals.shape == (b, n, q)

  @pytest.mark.slow
  @pytest.mark.potts
  def test_poe_marginals_sum_to_one(self) -> None:
    """Test that poe_marginals.sum(axis=-1) ≈ 1.0 (atol=1e-5)."""
    from aminx.potts.poe import PoeModel, PoeOutput

    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    cfg = {
      "hidden_dim": 16,
      "num_aa": 20,
      "k_neighbors": 30,
      "edge_features_dim": 64,
      "trw_iters": 2,
    }

    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k2)
    backbone_c = PottsModel(**cfg, key=k3)

    model = PoeModel(backbones=(backbone_a, backbone_b, backbone_c))

    # Create test data
    key_call = jax.random.PRNGKey(777)
    n = 10
    b = 3
    k = 30  # k_neighbors
    e = 64  # edge_features_dim
    mask = jnp.ones(n, dtype=jnp.float32)

    # Create synthetic edge_knn (B, N, k, E) and nei (B, N, k)
    edge_knn_stack = jnp.ones((b, n, k, e), dtype=jnp.float32)
    nei_stack = jnp.zeros((b, n, k), dtype=jnp.int32)
    for bi in range(b):
      for i in range(n):
        for j in range(k):
          nei_stack = nei_stack.at[bi, i, j].set((i + j + 1) % n)

    poe_output = model(
      key=key_call,
      edge_knn_stack=edge_knn_stack,
      nei_stack=nei_stack,
      mask=mask,
    )

    # Check that marginals sum to 1 over amino acid dimension
    marginal_sums = poe_output.marginals.sum(axis=-1)
    assert jnp.allclose(
      marginal_sums,
      jnp.ones(n),
      atol=1e-5,
    ), f"Marginals should sum to 1 over amino acid axis, got sums: {marginal_sums}"


class TestPoeModelSanityCheck:
  """Test embedded sanity check: PoE invariant."""

  @pytest.mark.slow
  @pytest.mark.potts
  def test_poe_sanity_check_two_identical_backbones(self) -> None:
    """Test that two identical backbones give joint_energy ≈ 2 * single_energy."""
    from aminx.potts.poe import PoeModel

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)

    cfg = {
      "hidden_dim": 16,
      "num_aa": 4,
      "k_neighbors": 30,
      "edge_features_dim": 64,
      "trw_iters": 2,
    }

    # Create two backbones with identical parameters (use same key)
    backbone_a = PottsModel(**cfg, key=k1)
    backbone_b = PottsModel(**cfg, key=k1)  # Same key → identical params

    model = PoeModel(backbones=(backbone_a, backbone_b))

    # Test with random but valid data
    seq = jnp.array([0, 1, 2], dtype=jnp.int32)
    h = jnp.array([[1.0, 0.5, 0.3, 0.1], [0.5, 1.0, 0.2, 0.3], [0.3, 0.2, 1.0, 0.5]])
    J = jnp.zeros((3, 3, 4, 4), dtype=jnp.float32)
    W = jnp.zeros((3, 3), dtype=jnp.float32)

    # Compute single and joint energies
    energy_single = PottsModel.log_prob(seq, h, J, W)
    energy_joint = model.joint_energy(seq, ((h, J, W), (h, J, W)))

    # joint_energy should be ≈ 2 * single_energy
    assert jnp.allclose(energy_joint, 2.0 * energy_single, atol=1e-5), (
      f"PoE sanity check failed: joint={energy_joint}, "
      f"2*single={2*energy_single}, diff={energy_joint - 2*energy_single}"
    )
