"""End-to-end assembly tests for the ProteinEBM-equivalent model (backlog node E3.6).

Covers ``aminx.ebm.model``: the first genuinely end-to-end
``score = -jax.grad(energy w.r.t. coords)`` check for the *assembled* model
(input embeddings -> ``SingleConditioning`` -> pairwise conditioning ->
``DiffusionTransformer`` -> readouts), not just the toy/readout-only cases
E0/E3 already covered. Also checks masked-residue exclusion, jit
compatibility, and the ``sc_coords=None`` vs. explicit-zeros self-conditioning
fallback (mirrors ``ebm.py:194``'s
``sc_coords if sc_coords is not None else torch.zeros_like(r_noisy)``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.ebm.model import InputEmbeddings, ProteinEBMModel

TOKEN_S = 16
TOKEN_Z = 8
DEPTH = 2
HEADS = 2
N = 6


def _make_model(key: jax.Array) -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=TOKEN_S,
    token_z=TOKEN_Z,
    dim_fourier=12,
    conditioning_transition_layers=1,
    transformer_depth=DEPTH,
    transformer_heads=HEADS,
    key=key,
  )


def _synthetic_inputs(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  k_coords, k_aatype = jax.random.split(key)
  coords = jax.random.normal(k_coords, (N, 3)) * 0.1
  aatype = jax.random.randint(k_aatype, (N,), 0, 21)
  mask = jnp.ones((N,), dtype=bool)
  t = jnp.array(0.3)
  return coords, aatype, mask, t


class TestInputEmbeddings:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(0)
    embed = InputEmbeddings(token_s=TOKEN_S, key=key)
    aatype = jnp.zeros((N,), dtype=jnp.int32)
    coords = jnp.zeros((N, 3))
    contacts = jnp.zeros((N,), dtype=jnp.int32)
    sc_coords = jnp.zeros((N, 3))
    out = embed(aatype, coords, contacts, sc_coords)
    assert out.shape == (N, TOKEN_S * 4)

  def test_output_is_finite(self) -> None:
    key = jax.random.PRNGKey(1)
    embed = InputEmbeddings(token_s=TOKEN_S, key=key)
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    aatype = jax.random.randint(k1, (N,), 0, 21)
    coords = jax.random.normal(k2, (N, 3))
    contacts = jnp.zeros((N,), dtype=jnp.int32)
    sc_coords = jnp.zeros((N, 3))
    out = embed(aatype, coords, contacts, sc_coords)
    assert jnp.all(jnp.isfinite(out))


class TestProteinEBMModelConstruction:
  def test_field_dims_are_consistent(self) -> None:
    key = jax.random.PRNGKey(3)
    model = _make_model(key)
    assert model.token_s == TOKEN_S
    assert model.token_z == TOKEN_Z
    assert len(model.token_transformer.layers) == DEPTH


class TestTrunkFeatures:
  def test_output_shape(self) -> None:
    key = jax.random.PRNGKey(4)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(5))

    out = model.trunk_features(coords, aatype, t, mask)
    assert out.shape == (N, 2 * TOKEN_S)
    assert jnp.all(jnp.isfinite(out))


class TestEnergy:
  def test_output_shape_and_finite_and_nonnegative(self) -> None:
    key = jax.random.PRNGKey(6)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(7))

    e = model.energy(coords, aatype, t, mask)
    assert e.shape == ()
    assert jnp.isfinite(e)
    assert e >= 0.0

  def test_masked_residues_do_not_change_energy(self) -> None:
    """Perturbing a masked-out residue's coords/aatype must not change the energy."""
    key = jax.random.PRNGKey(8)
    model = _make_model(key)
    coords, aatype, _mask, t = _synthetic_inputs(jax.random.PRNGKey(9))
    mask = jnp.array([True, True, True, True, False, False])

    e_before = model.energy(coords, aatype, t, mask)

    coords_perturbed = coords.at[4:].set(coords[4:] * 1000.0 + 37.0)
    aatype_perturbed = aatype.at[4:].set((aatype[4:] + 5) % 21)
    e_after = model.energy(coords_perturbed, aatype_perturbed, t, mask)

    assert jnp.allclose(e_before, e_after, atol=1e-4)

  def test_masked_residues_zero_out_entirely(self) -> None:
    """An all-False mask forces the readout's masked sum to exactly zero."""
    key = jax.random.PRNGKey(10)
    model = _make_model(key)
    coords, aatype, _mask, t = _synthetic_inputs(jax.random.PRNGKey(11))
    mask = jnp.zeros((N,), dtype=bool)

    e = model.energy(coords, aatype, t, mask)
    assert jnp.allclose(e, 0.0)


class TestScoreIsNegativeGradOfEnergy:
  def test_score_matches_negative_grad_of_energy(self) -> None:
    """The central EBM identity, end to end through the FULL assembled model."""
    key = jax.random.PRNGKey(12)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(13))

    score = model.score(coords, aatype, t, mask)

    def energy_of_coords(c: jax.Array) -> jax.Array:
      return model.energy(c, aatype, t, mask)

    expected = -jax.grad(energy_of_coords)(coords)
    assert score.shape == (N, 3)
    assert jnp.all(jnp.isfinite(score))
    assert jnp.allclose(score, expected, atol=1e-5, rtol=1e-5)

  def test_masked_residues_do_not_contribute_to_score(self) -> None:
    key = jax.random.PRNGKey(14)
    model = _make_model(key)
    coords, aatype, _mask, t = _synthetic_inputs(jax.random.PRNGKey(15))
    mask = jnp.array([True, True, True, True, False, False])

    score = model.score(coords, aatype, t, mask)
    assert jnp.allclose(score[4:], 0.0, atol=1e-6)


class TestAuxScore:
  def test_output_shape_and_finite(self) -> None:
    key = jax.random.PRNGKey(16)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(17))

    aux = model.aux_score(coords, aatype, t, mask)
    assert aux.shape == (N, 3)
    assert jnp.all(jnp.isfinite(aux))

  def test_aux_score_does_not_require_grad(self) -> None:
    """Plain forward call -- contrast score(), which wraps jax.grad."""
    key = jax.random.PRNGKey(18)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(19))

    # No jax.grad anywhere in this call path.
    aux = model.aux_score(coords, aatype, t, mask)
    assert aux.shape == (N, 3)


class TestSelfConditioningDefaultFallback:
  def test_sc_coords_none_matches_explicit_zeros(self) -> None:
    """``sc_coords=None`` must match passing explicit zeros (ebm.py:194's fallback)."""
    key = jax.random.PRNGKey(20)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(21))

    out_default = model.trunk_features(coords, aatype, t, mask, sc_coords=None)
    out_explicit_zeros = model.trunk_features(coords, aatype, t, mask, sc_coords=jnp.zeros_like(coords))

    assert jnp.array_equal(out_default, out_explicit_zeros)

  def test_nonzero_sc_coords_changes_trunk_features(self) -> None:
    """Sanity: self-conditioning is actually wired in (not silently dropped)."""
    key = jax.random.PRNGKey(22)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(23))
    sc_coords = jax.random.normal(jax.random.PRNGKey(24), (N, 3))

    out_zero_sc = model.trunk_features(coords, aatype, t, mask, sc_coords=jnp.zeros_like(coords))
    out_nonzero_sc = model.trunk_features(coords, aatype, t, mask, sc_coords=sc_coords)

    assert not jnp.allclose(out_zero_sc, out_nonzero_sc, atol=1e-4)


class TestContactsDefaultFallback:
  def test_contacts_none_matches_explicit_zeros(self) -> None:
    key = jax.random.PRNGKey(25)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(26))

    out_default = model.trunk_features(coords, aatype, t, mask, contacts=None)
    out_explicit_zeros = model.trunk_features(
      coords, aatype, t, mask, contacts=jnp.zeros((N,), dtype=jnp.int32)
    )
    assert jnp.array_equal(out_default, out_explicit_zeros)


class TestJitCompatibility:
  def test_energy_is_jit_compatible(self) -> None:
    key = jax.random.PRNGKey(27)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(28))

    eager = model.energy(coords, aatype, t, mask)
    jitted_fn = eqx.filter_jit(lambda m, c, a, tt, mm: m.energy(c, a, tt, mm))
    jitted = jitted_fn(model, coords, aatype, t, mask)
    assert jnp.allclose(eager, jitted, atol=1e-5)

  def test_score_is_jit_compatible(self) -> None:
    key = jax.random.PRNGKey(29)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(30))

    eager = model.score(coords, aatype, t, mask)
    jitted_fn = eqx.filter_jit(lambda m, c, a, tt, mm: m.score(c, a, tt, mm))
    jitted = jitted_fn(model, coords, aatype, t, mask)
    assert jnp.allclose(eager, jitted, atol=1e-5)

  def test_aux_score_is_jit_compatible(self) -> None:
    key = jax.random.PRNGKey(31)
    model = _make_model(key)
    coords, aatype, mask, t = _synthetic_inputs(jax.random.PRNGKey(32))

    eager = model.aux_score(coords, aatype, t, mask)
    jitted_fn = eqx.filter_jit(lambda m, c, a, tt, mm: m.aux_score(c, a, tt, mm))
    jitted = jitted_fn(model, coords, aatype, t, mask)
    assert jnp.allclose(eager, jitted, atol=1e-5)
