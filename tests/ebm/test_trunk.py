"""Shape + numerical-sanity tests for the ProteinEBM trunk port (backlog node E1).

Covers every module in ``aminx.ebm.trunk``: construction with small dims,
output-shape contracts, and a handful of *provable* structural invariants
(not "close to" hand-waves) that fall directly out of the ported math:

* :class:`~aminx.ebm.trunk.AdaLN` -- with the conditioning-facing linear
  weights zeroed, the output is provably independent of ``s`` (the linear
  algebra makes ``s`` unreachable, not merely "small").
* :class:`~aminx.ebm.trunk.AttentionPairBias` -- (i) at init, ``proj_o`` is
  zero-weight/no-bias (Boltz-1's residual-friendly zero-init), so the block's
  output is *exactly* zero for any input; (ii) with ``pair_proj`` zeroed, the
  pairwise bias becomes a per-head constant added uniformly across the
  softmax (key) axis, so by softmax shift-invariance the output is *exactly*
  independent of ``z``; (iii) masked-out key positions cannot influence any
  valid query's output.
* SO(3) rotation utilities -- genuine orthogonality (``R^T R = I``) and
  ``det = +1`` (proper rotations, not just O(3)) on a batch.
* :func:`~aminx.ebm.trunk.center_random_augmentation` -- ``rotate=False`` is
  deterministic-up-to-translation (pairwise relative vectors unchanged);
  ``rotate=True`` is reproducible under a fixed key but actually rotates
  (relative vectors change).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from aminx.ebm import trunk


# ---------------------------------------------------------------------------
# FourierEmbedding
# ---------------------------------------------------------------------------


class TestFourierEmbedding:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(0)
    embed = trunk.FourierEmbedding(dim=6, key=key)
    out = embed(jnp.array(0.3))
    assert out.shape == (6,)

  def test_output_is_bounded_cosine_range(self) -> None:
    key = jax.random.PRNGKey(1)
    embed = trunk.FourierEmbedding(dim=16, key=key)
    out = embed(jnp.array(0.7))
    assert jnp.all(out >= -1.0 - 1e-6)
    assert jnp.all(out <= 1.0 + 1e-6)

  def test_frozen_params_are_deterministic_given_key(self) -> None:
    key = jax.random.PRNGKey(2)
    embed_a = trunk.FourierEmbedding(dim=8, key=key)
    embed_b = trunk.FourierEmbedding(dim=8, key=key)
    assert jnp.array_equal(embed_a.weight, embed_b.weight)
    assert jnp.array_equal(embed_a.bias, embed_b.bias)

  def test_different_times_give_different_embeddings(self) -> None:
    key = jax.random.PRNGKey(3)
    embed = trunk.FourierEmbedding(dim=8, key=key)
    out_a = embed(jnp.array(0.1))
    out_b = embed(jnp.array(0.9))
    assert not jnp.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# RelativePositionEncoder
# ---------------------------------------------------------------------------


class TestRelativePositionEncoder:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(4)
    encoder = trunk.RelativePositionEncoder(token_z=10, r_max=4, s_max=1, key=key)
    residue_index = jnp.arange(6)
    chain_id = jnp.zeros(6, dtype=jnp.int32)
    out = encoder(residue_index, chain_id)
    assert out.shape == (6, 6, 10)

  def test_multi_chain_shape_and_finiteness(self) -> None:
    key = jax.random.PRNGKey(5)
    encoder = trunk.RelativePositionEncoder(token_z=8, key=key)
    residue_index = jnp.concatenate([jnp.arange(3), jnp.arange(4)])
    chain_id = jnp.array([0, 0, 0, 1, 1, 1, 1])
    out = encoder(residue_index, chain_id)
    assert out.shape == (7, 7, 8)
    assert jnp.all(jnp.isfinite(out))

  def test_diagonal_is_self_relative_position(self) -> None:
    # i == j -> rel_pos == 0, same chain -> a deterministic feature
    # independent of which residue, so the diagonal is constant across i.
    key = jax.random.PRNGKey(6)
    encoder = trunk.RelativePositionEncoder(token_z=5, key=key)
    residue_index = jnp.arange(5)
    chain_id = jnp.zeros(5, dtype=jnp.int32)
    out = encoder(residue_index, chain_id)
    diag = jnp.diagonal(out, axis1=0, axis2=1).T  # (5, token_z)
    assert jnp.allclose(diag, diag[0], atol=1e-5)


# ---------------------------------------------------------------------------
# AdaLN
# ---------------------------------------------------------------------------


def _zero_adaln_conditioning_weights(adaln: trunk.AdaLN) -> trunk.AdaLN:
  """Zero ``s_scale``/``s_bias`` weights so ``s`` becomes unreachable (see module docstring)."""
  adaln = eqx.tree_at(lambda m: m.s_scale.weight, adaln, jnp.zeros_like(adaln.s_scale.weight))
  return eqx.tree_at(lambda m: m.s_bias.weight, adaln, jnp.zeros_like(adaln.s_bias.weight))


class TestAdaLN:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(7)
    adaln = trunk.AdaLN(dim=8, dim_single_cond=6, key=key)
    a = jax.random.normal(jax.random.PRNGKey(8), (5, 8))
    s = jax.random.normal(jax.random.PRNGKey(9), (5, 6))
    out = adaln(a, s)
    assert out.shape == (5, 8)

  def test_conditioning_becomes_unreachable_when_weights_zeroed(self) -> None:
    """With s_scale/s_bias weights zeroed, output cannot depend on s (real invariant)."""
    key = jax.random.PRNGKey(10)
    adaln = _zero_adaln_conditioning_weights(trunk.AdaLN(dim=6, dim_single_cond=4, key=key))
    a = jax.random.normal(jax.random.PRNGKey(11), (4, 6))
    s_one = jax.random.normal(jax.random.PRNGKey(12), (4, 4))
    s_two = jax.random.normal(jax.random.PRNGKey(13), (4, 4)) * 100.0  # wildly different

    out_one = adaln(a, s_one)
    out_two = adaln(a, s_two)
    assert jnp.allclose(out_one, out_two, atol=1e-5)


# ---------------------------------------------------------------------------
# Transition (reuses model.diffusion_mpnn.SwiGLU)
# ---------------------------------------------------------------------------


class TestTransition:
  def test_output_shape_default_out_dim(self) -> None:
    key = jax.random.PRNGKey(14)
    transition = trunk.Transition(dim=8, hidden=16, key=key)
    x = jax.random.normal(jax.random.PRNGKey(15), (8,))
    out = transition(x)
    assert out.shape == (8,)

  def test_output_shape_explicit_out_dim(self) -> None:
    key = jax.random.PRNGKey(16)
    transition = trunk.Transition(dim=8, hidden=16, out_dim=5, key=key)
    x = jax.random.normal(jax.random.PRNGKey(17), (8,))
    out = transition(x)
    assert out.shape == (5,)

  def test_vmappable_over_single_and_pair_reps(self) -> None:
    key = jax.random.PRNGKey(18)
    transition = trunk.Transition(dim=4, hidden=8, key=key)
    single = jax.random.normal(jax.random.PRNGKey(19), (6, 4))
    pair = jax.random.normal(jax.random.PRNGKey(20), (6, 6, 4))

    out_single = jax.vmap(transition)(single)
    out_pair = jax.vmap(jax.vmap(transition))(pair)
    assert out_single.shape == (6, 4)
    assert out_pair.shape == (6, 6, 4)


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock
# ---------------------------------------------------------------------------


class TestConditionedTransitionBlock:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(21)
    block = trunk.ConditionedTransitionBlock(dim_single=8, dim_single_cond=6, key=key)
    a = jax.random.normal(jax.random.PRNGKey(22), (5, 8))
    s = jax.random.normal(jax.random.PRNGKey(23), (5, 6))
    out = block(a, s)
    assert out.shape == (5, 8)

  def test_output_is_finite(self) -> None:
    key = jax.random.PRNGKey(24)
    block = trunk.ConditionedTransitionBlock(dim_single=4, dim_single_cond=4, key=key)
    a = jax.random.normal(jax.random.PRNGKey(25), (3, 4)) * 10.0
    s = jax.random.normal(jax.random.PRNGKey(26), (3, 4)) * 10.0
    out = block(a, s)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# AttentionPairBias
# ---------------------------------------------------------------------------


class TestAttentionPairBias:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(27)
    attn = trunk.AttentionPairBias(c_s=8, c_z=6, num_heads=2, key=key)
    s = jax.random.normal(jax.random.PRNGKey(28), (5, 8))
    z = jax.random.normal(jax.random.PRNGKey(29), (5, 5, 6))
    mask = jnp.ones((5,), dtype=bool)
    out = attn(s, z, mask)
    assert out.shape == (5, 8)

  def test_rejects_non_divisible_heads(self) -> None:
    with pytest.raises(ValueError, match="divisible"):
      trunk.AttentionPairBias(c_s=7, c_z=6, num_heads=2, key=jax.random.PRNGKey(30))

  def test_output_is_exactly_zero_at_init(self) -> None:
    """proj_o is zero-weight + no-bias at init (final_init_) -> output ≡ 0 for any input."""
    key = jax.random.PRNGKey(31)
    attn = trunk.AttentionPairBias(c_s=6, c_z=4, num_heads=3, key=key)
    s = jax.random.normal(jax.random.PRNGKey(32), (4, 6)) * 5.0
    z = jax.random.normal(jax.random.PRNGKey(33), (4, 4, 4)) * 5.0
    mask = jnp.ones((4,), dtype=bool)
    out = attn(s, z, mask)
    assert jnp.allclose(out, 0.0, atol=1e-6)

  def test_output_independent_of_z_when_pair_proj_zeroed(self) -> None:
    """With pair_proj zeroed, z's contribution is a per-head constant -> softmax shift-invariant."""
    key = jax.random.PRNGKey(34)
    attn = trunk.AttentionPairBias(c_s=8, c_z=5, num_heads=2, key=key)
    # Also give proj_o nonzero weight so the exactly-zero-at-init invariant
    # above doesn't mask this one out.
    attn = eqx.tree_at(
      lambda m: m.proj_o.weight,
      attn,
      jax.random.normal(jax.random.PRNGKey(35), attn.proj_o.weight.shape),
    )
    attn = eqx.tree_at(lambda m: m.pair_proj.weight, attn, jnp.zeros_like(attn.pair_proj.weight))

    s = jax.random.normal(jax.random.PRNGKey(36), (5, 8))
    mask = jnp.ones((5,), dtype=bool)
    z_one = jax.random.normal(jax.random.PRNGKey(37), (5, 5, 5))
    z_two = jax.random.normal(jax.random.PRNGKey(38), (5, 5, 5)) * 50.0

    out_one = attn(s, z_one, mask)
    out_two = attn(s, z_two, mask)
    assert jnp.allclose(out_one, out_two, atol=1e-4)

  def test_masked_key_position_does_not_affect_valid_queries(self) -> None:
    key = jax.random.PRNGKey(39)
    attn = trunk.AttentionPairBias(c_s=6, c_z=4, num_heads=2, key=key)
    attn = eqx.tree_at(
      lambda m: m.proj_o.weight,
      attn,
      jax.random.normal(jax.random.PRNGKey(40), attn.proj_o.weight.shape),
    )

    s = jax.random.normal(jax.random.PRNGKey(41), (5, 6))
    z = jax.random.normal(jax.random.PRNGKey(42), (5, 5, 4))
    mask = jnp.array([True, True, True, True, False])

    out_one = attn(s, z, mask)

    s_perturbed = s.at[-1].set(s[-1] * 1000.0 + 37.0)
    out_two = attn(s_perturbed, z, mask)

    # Valid (non-masked) query rows must be unaffected by the masked
    # position's (excluded) key/value content.
    assert jnp.allclose(out_one[:-1], out_two[:-1], atol=1e-4)

  def test_initial_norm_false_skips_norm_s(self) -> None:
    attn = trunk.AttentionPairBias(c_s=4, c_z=3, num_heads=2, initial_norm=False, key=jax.random.PRNGKey(43))
    assert attn.norm_s is None


# ---------------------------------------------------------------------------
# DiffusionTransformerLayer / DiffusionTransformer (end-to-end)
# ---------------------------------------------------------------------------


class TestDiffusionTransformerLayer:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(44)
    layer = trunk.DiffusionTransformerLayer(heads=2, dim=8, dim_pairwise=4, key=key)
    a = jax.random.normal(jax.random.PRNGKey(45), (5, 8))
    s = jax.random.normal(jax.random.PRNGKey(46), (5, 8))
    z = jax.random.normal(jax.random.PRNGKey(47), (5, 5, 4))
    mask = jnp.ones((5,), dtype=bool)
    out = layer(a, s, z, mask)
    assert out.shape == (5, 8)


class TestDiffusionTransformer:
  def test_end_to_end_output_shape_matches_dim(self) -> None:
    """Single-residue-set input -> output shape matches `dim` (task spec (b))."""
    key = jax.random.PRNGKey(48)
    dim, dim_pairwise, n = 8, 6, 5
    model = trunk.DiffusionTransformer(depth=2, heads=2, dim=dim, dim_pairwise=dim_pairwise, key=key)

    a = jax.random.normal(jax.random.PRNGKey(49), (n, dim))
    s = jax.random.normal(jax.random.PRNGKey(50), (n, dim))
    z = jax.random.normal(jax.random.PRNGKey(51), (n, n, dim_pairwise))
    mask = jnp.ones((n,), dtype=bool)

    out = model(a, s, z, mask)
    assert out.shape == (n, dim)
    assert jnp.all(jnp.isfinite(out))

  def test_default_construction_uses_proteinebm_x_defaults(self) -> None:
    key = jax.random.PRNGKey(52)
    model = trunk.DiffusionTransformer(depth=1, key=key)
    assert len(model.layers) == 1

  def test_jit_compatible(self) -> None:
    key = jax.random.PRNGKey(53)
    dim, dim_pairwise, n = 6, 4, 4
    model = trunk.DiffusionTransformer(depth=1, heads=2, dim=dim, dim_pairwise=dim_pairwise, key=key)

    a = jax.random.normal(jax.random.PRNGKey(54), (n, dim))
    s = jax.random.normal(jax.random.PRNGKey(55), (n, dim))
    z = jax.random.normal(jax.random.PRNGKey(56), (n, n, dim_pairwise))
    mask = jnp.ones((n,), dtype=bool)

    eager = model(a, s, z, mask)
    jitted = eqx.filter_jit(lambda m, a_, s_, z_, mask_: m(a_, s_, z_, mask_))(model, a, s, z, mask)
    assert jnp.allclose(eager, jitted, atol=1e-5)


# ---------------------------------------------------------------------------
# SingleConditioning / PairwiseConditioning
# ---------------------------------------------------------------------------


class TestSingleConditioning:
  def test_output_shape_with_times(self) -> None:
    key = jax.random.PRNGKey(57)
    cond = trunk.SingleConditioning(input_dim=6, token_s=4, dim_fourier=8, key=key)
    s = jax.random.normal(jax.random.PRNGKey(58), (5, 6))
    out, fourier = cond(s, times=jnp.array(0.4))
    assert out.shape == (5, 8)  # 2 * token_s
    assert fourier.shape == (8,)

  def test_output_shape_with_direct_embedding(self) -> None:
    key = jax.random.PRNGKey(59)
    cond = trunk.SingleConditioning(input_dim=6, token_s=4, dim_fourier=8, key=key)
    s = jax.random.normal(jax.random.PRNGKey(60), (5, 6))
    direct = jax.random.normal(jax.random.PRNGKey(61), (8,))
    out, fourier = cond(s, direct_embedding=direct)
    assert out.shape == (5, 8)
    assert jnp.array_equal(fourier, direct)

  def test_raises_when_neither_provided(self) -> None:
    key = jax.random.PRNGKey(62)
    cond = trunk.SingleConditioning(input_dim=4, token_s=4, dim_fourier=4, key=key)
    s = jax.random.normal(jax.random.PRNGKey(63), (3, 4))
    with pytest.raises(ValueError, match="Either times or direct_embedding"):
      cond(s)

  def test_raises_when_both_provided(self) -> None:
    key = jax.random.PRNGKey(64)
    cond = trunk.SingleConditioning(input_dim=4, token_s=4, dim_fourier=4, key=key)
    s = jax.random.normal(jax.random.PRNGKey(65), (3, 4))
    direct = jax.random.normal(jax.random.PRNGKey(66), (4,))
    with pytest.raises(ValueError, match="Cannot provide both"):
      cond(s, times=jnp.array(0.1), direct_embedding=direct)


class TestPairwiseConditioning:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(67)
    cond = trunk.PairwiseConditioning(input_dim=4, token_z=6, dim_token_rel_pos_feats=3, key=key)
    z_trunk = jax.random.normal(jax.random.PRNGKey(68), (5, 5, 4))
    rel_pos_feats = jax.random.normal(jax.random.PRNGKey(69), (5, 5, 3))
    out = cond(z_trunk, rel_pos_feats)
    assert out.shape == (5, 5, 6)


# ---------------------------------------------------------------------------
# Quaternion / SO(3) rotation utilities
# ---------------------------------------------------------------------------


class TestQuaternionRotationUtils:
  def test_random_rotations_are_orthogonal_with_det_plus_one(self) -> None:
    """(e) task spec: genuinely SO(3), not just O(3), on a batch."""
    key = jax.random.PRNGKey(70)
    rotations = trunk.random_rotations(key, 64)

    identity = jnp.eye(3, dtype=rotations.dtype)
    rt_r = jnp.einsum("bij,bik->bjk", rotations, rotations)
    assert jnp.allclose(rt_r, identity[None], atol=1e-5)

    dets = jnp.linalg.det(rotations)
    assert jnp.allclose(dets, 1.0, atol=1e-5)

  def test_random_quaternions_are_unit_norm_with_nonnegative_real_part(self) -> None:
    key = jax.random.PRNGKey(71)
    quats = trunk.random_quaternions(key, 32)
    norms = jnp.linalg.norm(quats, axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)
    assert jnp.all(quats[:, 0] >= -1e-6)

  def test_quaternion_to_matrix_identity_quaternion_gives_identity_matrix(self) -> None:
    identity_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
    mat = trunk.quaternion_to_matrix(identity_quat)
    assert jnp.allclose(mat, jnp.eye(3), atol=1e-6)


# ---------------------------------------------------------------------------
# center_random_augmentation / batched_center_random_augmentation
# ---------------------------------------------------------------------------


def _pairwise_relative_vectors(coords: jax.Array) -> jax.Array:
  return coords[:, None, :] - coords[None, :, :]


class TestCenterRandomAugmentation:
  def test_no_augmentation_just_centers(self) -> None:
    coords = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]])
    mask = jnp.ones((3,), dtype=bool)
    out = trunk.center_random_augmentation(
      coords,
      mask,
      jax.random.PRNGKey(0),
      augmentation=False,
    )
    expected = coords - jnp.mean(coords, axis=0, keepdims=True)
    assert jnp.allclose(out, expected, atol=1e-6)

  def test_rotate_false_is_identity_up_to_translation(self) -> None:
    """rotate=False: pairwise relative vectors are unchanged (only centering + translation)."""
    key = jax.random.PRNGKey(1)
    coords = jax.random.normal(jax.random.PRNGKey(2), (6, 3)) * 3.0
    mask = jnp.ones((6,), dtype=bool)

    out = trunk.center_random_augmentation(coords, mask, key, rotate=False)

    centered = coords - jnp.mean(coords, axis=0, keepdims=True)
    rel_before = _pairwise_relative_vectors(centered)
    rel_after = _pairwise_relative_vectors(out)
    assert jnp.allclose(rel_before, rel_after, atol=1e-5)

  def test_rotate_true_changes_relative_vectors(self) -> None:
    key = jax.random.PRNGKey(3)
    coords = jax.random.normal(jax.random.PRNGKey(4), (6, 3)) * 3.0
    mask = jnp.ones((6,), dtype=bool)

    out = trunk.center_random_augmentation(coords, mask, key, rotate=True)

    centered = coords - jnp.mean(coords, axis=0, keepdims=True)
    rel_before = _pairwise_relative_vectors(centered)
    rel_after = _pairwise_relative_vectors(out)
    assert not jnp.allclose(rel_before, rel_after, atol=1e-3)

  def test_rotate_true_preserves_pairwise_distances(self) -> None:
    """Rotation + translation are isometries: pairwise distances must be exactly preserved."""
    key = jax.random.PRNGKey(5)
    coords = jax.random.normal(jax.random.PRNGKey(6), (7, 3)) * 2.0
    mask = jnp.ones((7,), dtype=bool)

    out = trunk.center_random_augmentation(coords, mask, key, rotate=True)

    dist_before = jnp.linalg.norm(_pairwise_relative_vectors(coords), axis=-1)
    dist_after = jnp.linalg.norm(_pairwise_relative_vectors(out), axis=-1)
    assert jnp.allclose(dist_before, dist_after, atol=1e-4)

  def test_reproducible_under_fixed_key(self) -> None:
    key = jax.random.PRNGKey(7)
    coords = jax.random.normal(jax.random.PRNGKey(8), (5, 3))
    mask = jnp.ones((5,), dtype=bool)

    out_a = trunk.center_random_augmentation(coords, mask, key, rotate=True)
    out_b = trunk.center_random_augmentation(coords, mask, key, rotate=True)
    assert jnp.array_equal(out_a, out_b)

  def test_different_keys_give_different_augmentations(self) -> None:
    coords = jax.random.normal(jax.random.PRNGKey(9), (5, 3))
    mask = jnp.ones((5,), dtype=bool)
    key_a, key_b = jax.random.split(jax.random.PRNGKey(10))

    out_a = trunk.center_random_augmentation(coords, mask, key_a, rotate=True)
    out_b = trunk.center_random_augmentation(coords, mask, key_b, rotate=True)
    assert not jnp.allclose(out_a, out_b, atol=1e-4)


class TestBatchedCenterRandomAugmentation:
  def test_matches_fold_in_per_element(self) -> None:
    key = jax.random.PRNGKey(11)
    coords = jax.random.normal(jax.random.PRNGKey(12), (3, 5, 3))
    mask = jnp.ones((3, 5), dtype=bool)

    batched = trunk.batched_center_random_augmentation(coords, mask, key, rotate=True)

    for idx in range(3):
      sub_key = jax.random.fold_in(key, idx)
      expected = trunk.center_random_augmentation(coords[idx], mask[idx], sub_key, rotate=True)
      assert jnp.allclose(batched[idx], expected, atol=1e-5)

  def test_batch_elements_get_independent_augmentations(self) -> None:
    key = jax.random.PRNGKey(13)
    # Identical coords/mask across the batch; augmentations must still differ per element.
    single = jax.random.normal(jax.random.PRNGKey(14), (4, 3))
    coords = jnp.stack([single, single, single])
    mask = jnp.ones((3, 4), dtype=bool)

    out = trunk.batched_center_random_augmentation(coords, mask, key, rotate=True)
    assert not jnp.allclose(out[0], out[1], atol=1e-4)
    assert not jnp.allclose(out[1], out[2], atol=1e-4)
