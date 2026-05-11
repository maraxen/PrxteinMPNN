"""Sprint MODELINPUTS + EncoderHooks regression tests."""
import jax.numpy as jnp
from prxteinmpnn.pipeline_registry import make_geometric_mean_transform, register_hook


def test_make_geometric_mean_transform_factory():
    T = 0.5
    fn = make_geometric_mean_transform(T)
    state_logits = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # (S=2, 3)
    result = fn(state_logits, state_index=None, state_weights=None)
    # Closed-form: mean(state_logits, axis=0) / T
    expected = jnp.mean(state_logits, axis=0) / T
    assert jnp.allclose(result, expected, atol=1e-5)


def test_make_geometric_mean_transform_registerable():
    fn = make_geometric_mean_transform(0.1)
    uid = register_hook(fn, name="test_geom_mean")
    assert isinstance(uid, str) and len(uid) == 16


def test_make_geometric_mean_transform_cache_idempotent():
    """Same temperature must return the same closure object (no registry leak)."""
    fn_a = make_geometric_mean_transform(0.5)
    fn_b = make_geometric_mean_transform(0.5)
    assert fn_a is fn_b, "Same temperature must return the same cached closure"


def test_geometric_mean_transform_temperature_effect():
    state_logits = jnp.array([[0.0, 1.0, -1.0], [0.0, -1.0, 1.0]])
    fn_hot = make_geometric_mean_transform(2.0)
    fn_cold = make_geometric_mean_transform(0.5)
    out_hot = fn_hot(state_logits, None, None)
    out_cold = fn_cold(state_logits, None, None)
    # cold (T=0.5) should produce 4x the magnitude of hot (T=2.0)
    assert jnp.allclose(out_cold, 4.0 * out_hot, atol=1e-5)


def test_mpnn_score_unconditional_no_temperature_param():
    import inspect
    import jax
    from prxteinmpnn.model.mpnn import PrxteinMPNN

    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=jax.random.PRNGKey(0))
    sig = inspect.signature(m.score_unconditional)
    assert "multi_state_temperature" not in sig.parameters, (
        "multi_state_temperature must not appear in score_unconditional"
    )


def test_score_py_temperature_semantics_preserved():
    """After D.5 migration, scoring via _make_score_fn_state_vmap_exact with two
    different PipelineFns temperatures must yield different logits."""
    import jax
    import jax.numpy as jnp
    from prxteinmpnn.model.mpnn import PrxteinMPNN
    from prxteinmpnn.scoring.score import _make_score_fn_state_vmap_exact
    from prxteinmpnn.pipeline_fns import PipelineFns
    from prxteinmpnn.pipeline_registry import make_geometric_mean_transform

    S, L = 1, 6
    key = jax.random.PRNGKey(0)
    m = PrxteinMPNN(16, 16, 16, 1, 1, 6, key=key)

    score_fn = _make_score_fn_state_vmap_exact(m, inference=False)

    # state_flat_rows: shape (S, L) where row i lists the flat indices for state i
    state_flat_rows = jnp.array([[i * L + j for j in range(L)] for i in range(S)])
    fns_T01 = PipelineFns.from_callables(logit_transform=make_geometric_mean_transform(0.1))
    fns_T20 = PipelineFns.from_callables(logit_transform=make_geometric_mean_transform(2.0))

    seq_flat = jnp.zeros(S * L, dtype=jnp.int32)
    # score_sequence takes (prng_key, sequence, structure_coords, mask, ri, ci, ...)
    # structure_coords/mask/ri/ci are positional but deleted immediately; pass None.
    # Real data comes from coords_stack etc. keyword-only args.
    common_args = (key, seq_flat, None, None, None, None)
    common_kw = dict(
        coords_stack=jnp.zeros((S, L, 4, 3)),
        mask_stack=jnp.ones((S, L)),
        residue_index_stack=jnp.tile(jnp.arange(L, dtype=jnp.int32), (S, 1)),
        chain_index_stack=jnp.zeros((S, L), dtype=jnp.int32),
        state_flat_rows=state_flat_rows,
        n_flat=int(S * L),
        structure_mapping=None,
        tie_group_map=jnp.zeros(S * L, dtype=jnp.int32),
        multi_state_strategy="geometric_mean",
        multi_state_temperature=1.0,
        state_weights=jnp.ones(S, dtype=jnp.float32),
        y_stack=None,
        y_t_stack=None,
        y_m_stack=None,
        ar_mask_stack=None,
        bias_flat=None,
    )
    _, logits_T01, _ = score_fn(*common_args, fns=fns_T01, **common_kw)
    _, logits_T20, _ = score_fn(*common_args, fns=fns_T20, **common_kw)

    assert not jnp.allclose(logits_T01, logits_T20, atol=1e-5), (
        "T=0.1 and T=2.0 must produce different logits. "
        "If temperature is silently discarded, this assertion fails."
    )
