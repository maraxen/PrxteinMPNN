"""Regression coverage for the ProteinEBM checkpoint weight-port (backlog node E3.5).

This is the fast, CI-safe, torch-free complement to
``scripts/ebm/checkpoint_parity_check.py`` (the actual independent parity
gate against the real PyTorch reference model + downloaded checkpoint --
requires ``torch``, ``~/repos/ProteinEBM``, and a 430MB checkpoint file, none
of which belong in the pytest suite). These tests exercise
``aminx.ebm.checkpoint.load_pytorch_checkpoint``'s remap/skip/shape-validation
logic against a small *synthetic* state dict built with the exact reference
key structure (verified against the real checkpoint's key names/shapes --
see ``checkpoint.py``'s module docstring), so a future refactor of the
mapping table cannot silently regress without a fast, always-run test
catching it.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from aminx.ebm.checkpoint import load_pytorch_checkpoint
from aminx.ebm.model import ProteinEBMModel

_TOKEN_S = 8
_TOKEN_Z = 6
_DIM_FOURIER = 4
_NUM_TRANSITIONS = 1
_DEPTH = 2
_HEADS = 2
_NUM_CONTACT_EMBEDDINGS = 3


def _leaf(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
  return rng.standard_normal(shape).astype(np.float32)


def _build_model() -> ProteinEBMModel:
  return ProteinEBMModel(
    token_s=_TOKEN_S,
    token_z=_TOKEN_Z,
    dim_fourier=_DIM_FOURIER,
    conditioning_transition_layers=_NUM_TRANSITIONS,
    transformer_depth=_DEPTH,
    transformer_heads=_HEADS,
    num_contact_embeddings=_NUM_CONTACT_EMBEDDINGS,
    key=jax.random.PRNGKey(0),
  )


def _build_synthetic_state_dict(rng: np.random.Generator) -> dict[str, np.ndarray]:
  """Build a synthetic ``model.*``-prefixed state dict matching the real key structure.

  Shapes are consistent with a tiny (``token_s=8, token_z=6, depth=2,
  heads=2, dim_fourier=4, conditioning_transition_layers=1,
  num_contact_embeddings=3``) instance -- not the real ProteinEBM-x
  dimensions, but the exact same *key names* verified against the real
  checkpoint (see ``checkpoint.py`` module docstring).
  """
  two_token_s = 2 * _TOKEN_S  # 16
  four_token_s = 4 * _TOKEN_S  # 32
  concat_dim = four_token_s + _TOKEN_Z  # pairwise_conditioner input: 32 + 6 = 38
  sd: dict[str, np.ndarray] = {
    "model.sequence_embedding.weight": _leaf(rng, (21, _TOKEN_S)),
    "model.noisy_coord_embedding.weight": _leaf(rng, (_TOKEN_S, 3)),
    "model.contact_embedding.weight": _leaf(rng, (_NUM_CONTACT_EMBEDDINGS, _TOKEN_S)),
    "model.self_conditioning_embedding.weight": _leaf(rng, (_TOKEN_S, 3)),
    "model.single_conditioner.norm_single.weight": _leaf(rng, (four_token_s,)),
    "model.single_conditioner.norm_single.bias": _leaf(rng, (four_token_s,)),
    "model.single_conditioner.single_embed.weight": _leaf(rng, (two_token_s, four_token_s)),
    "model.single_conditioner.single_embed.bias": _leaf(rng, (two_token_s,)),
    "model.single_conditioner.fourier_embed.proj.weight": _leaf(rng, (_DIM_FOURIER, 1)),
    "model.single_conditioner.fourier_embed.proj.bias": _leaf(rng, (_DIM_FOURIER,)),
    "model.single_conditioner.norm_fourier.weight": _leaf(rng, (_DIM_FOURIER,)),
    "model.single_conditioner.norm_fourier.bias": _leaf(rng, (_DIM_FOURIER,)),
    "model.single_conditioner.fourier_to_single.weight": _leaf(rng, (two_token_s, _DIM_FOURIER)),
    "model.rel_pos.linear_layer.weight": _leaf(rng, (_TOKEN_Z, 68)),
    "model.pairwise_conditioner.dim_pairwise_init_proj.0.weight": _leaf(rng, (concat_dim,)),
    "model.pairwise_conditioner.dim_pairwise_init_proj.0.bias": _leaf(rng, (concat_dim,)),
    "model.pairwise_conditioner.dim_pairwise_init_proj.1.weight": _leaf(rng, (_TOKEN_Z, concat_dim)),
    "model.s_to_a_linear.0.weight": _leaf(rng, (two_token_s,)),
    "model.s_to_a_linear.0.bias": _leaf(rng, (two_token_s,)),
    "model.s_to_a_linear.1.weight": _leaf(rng, (two_token_s, two_token_s)),
    "model.a_norm.weight": _leaf(rng, (two_token_s,)),
    "model.a_norm.bias": _leaf(rng, (two_token_s,)),
    "model.r_update_proj.weight": _leaf(rng, (3, two_token_s)),
    "model.r_update_proj_aux.weight": _leaf(rng, (3, two_token_s)),
    "model.sidechain_proj.weight": _leaf(rng, (108, two_token_s)),
  }
  for conditioner, dim in (("single_conditioner", two_token_s), ("pairwise_conditioner", _TOKEN_Z)):
    hidden = 2 * dim
    for t_idx in range(_NUM_TRANSITIONS):
      p = f"model.{conditioner}.transitions.{t_idx}"
      sd[f"{p}.norm.weight"] = _leaf(rng, (dim,))
      sd[f"{p}.norm.bias"] = _leaf(rng, (dim,))
      sd[f"{p}.fc1.weight"] = _leaf(rng, (hidden, dim))
      sd[f"{p}.fc2.weight"] = _leaf(rng, (hidden, dim))
      sd[f"{p}.fc3.weight"] = _leaf(rng, (dim, hidden))

  for i in range(_DEPTH):
    p = f"model.token_transformer.layers.{i}"
    sd[f"{p}.adaln.s_norm.weight"] = _leaf(rng, (two_token_s,))
    sd[f"{p}.adaln.s_scale.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.adaln.s_scale.bias"] = _leaf(rng, (two_token_s,))
    sd[f"{p}.adaln.s_bias.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.pair_bias_attn.proj_q.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.pair_bias_attn.proj_q.bias"] = _leaf(rng, (two_token_s,))
    sd[f"{p}.pair_bias_attn.proj_k.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.pair_bias_attn.proj_v.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.pair_bias_attn.proj_g.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.pair_bias_attn.proj_z.0.weight"] = _leaf(rng, (_TOKEN_Z,))
    sd[f"{p}.pair_bias_attn.proj_z.0.bias"] = _leaf(rng, (_TOKEN_Z,))
    sd[f"{p}.pair_bias_attn.proj_z.1.weight"] = _leaf(rng, (_HEADS, _TOKEN_Z))
    sd[f"{p}.pair_bias_attn.proj_o.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.output_projection_linear.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.output_projection_linear.bias"] = _leaf(rng, (two_token_s,))
    # Reference-side duplicate: same nn.Parameter under a second attribute
    # name (verified bit-identical against the real checkpoint) -- mirror
    # that here so the skip-as-redundant path is exercised too.
    sd[f"{p}.output_projection.0.weight"] = sd[f"{p}.output_projection_linear.weight"]
    sd[f"{p}.output_projection.0.bias"] = sd[f"{p}.output_projection_linear.bias"]
    sd[f"{p}.transition.adaln.s_norm.weight"] = _leaf(rng, (two_token_s,))
    sd[f"{p}.transition.adaln.s_scale.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.transition.adaln.s_scale.bias"] = _leaf(rng, (two_token_s,))
    sd[f"{p}.transition.adaln.s_bias.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.transition.swish_gate.0.weight"] = _leaf(rng, (4 * two_token_s, two_token_s))
    sd[f"{p}.transition.a_to_b.weight"] = _leaf(rng, (2 * two_token_s, two_token_s))
    sd[f"{p}.transition.b_to_a.weight"] = _leaf(rng, (two_token_s, 2 * two_token_s))
    sd[f"{p}.transition.output_projection.0.weight"] = _leaf(rng, (two_token_s, two_token_s))
    sd[f"{p}.transition.output_projection.0.bias"] = _leaf(rng, (two_token_s,))

  return sd


@pytest.fixture
def synthetic_state_dict() -> dict[str, np.ndarray]:
  return _build_synthetic_state_dict(np.random.default_rng(0))


def test_full_remap_loads_every_non_skipped_key(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  model = _build_model()
  ported, report = load_pytorch_checkpoint(model, synthetic_state_dict)

  assert len(report.loaded_keys) + len(report.skipped_keys) <= len(synthetic_state_dict)
  # Every key must be accounted for: loaded, or skipped-with-reason (the
  # a_norm.{weight,bias} keys count once each in loaded_keys but populate two
  # destinations -- see checkpoint.py's _special_case_entries).
  accounted = set(report.loaded_keys) | {k for k, _ in report.skipped_keys}
  stripped_keys = {k.removeprefix("model.") for k in synthetic_state_dict}
  assert accounted == stripped_keys

  skipped_names = {k for k, _ in report.skipped_keys}
  assert "s_to_a_linear.0.weight" in skipped_names
  assert "s_to_a_linear.1.weight" in skipped_names
  assert "sidechain_proj.weight" in skipped_names
  assert "token_transformer.layers.0.output_projection.0.weight" in skipped_names
  assert "token_transformer.layers.1.output_projection.0.bias" in skipped_names
  # The *_linear form (the one actually loaded) must NOT be in the skip list.
  assert "token_transformer.layers.0.output_projection_linear.weight" not in skipped_names


def test_a_norm_duplicated_into_both_readout_norms(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  model = _build_model()
  ported, _report = load_pytorch_checkpoint(model, synthetic_state_dict)

  expected_weight = jnp.asarray(synthetic_state_dict["model.a_norm.weight"])
  expected_bias = jnp.asarray(synthetic_state_dict["model.a_norm.bias"])
  assert jnp.allclose(ported.energy_readout.norm.weight, expected_weight)
  assert jnp.allclose(ported.aux_score_readout.norm.weight, expected_weight)
  assert jnp.allclose(ported.energy_readout.norm.bias, expected_bias)
  assert jnp.allclose(ported.aux_score_readout.norm.bias, expected_bias)


def test_fourier_embed_weight_is_squeezed(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  model = _build_model()
  ported, _report = load_pytorch_checkpoint(model, synthetic_state_dict)

  ref_weight = synthetic_state_dict["model.single_conditioner.fourier_embed.proj.weight"]
  assert ref_weight.shape == (_DIM_FOURIER, 1)
  assert ported.single_conditioner.fourier_embed.weight.shape == (_DIM_FOURIER,)
  assert jnp.allclose(ported.single_conditioner.fourier_embed.weight, jnp.asarray(ref_weight).squeeze(-1))


def test_extra_swiglu_biases_are_zeroed_not_left_random(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  model = _build_model()
  ported, _report = load_pytorch_checkpoint(model, synthetic_state_dict)

  for transition in ported.single_conditioner.transitions:
    assert jnp.allclose(transition.swiglu.w_gate.bias, 0.0)
    assert jnp.allclose(transition.swiglu.w_val.bias, 0.0)
    assert jnp.allclose(transition.swiglu.w_out.bias, 0.0)
  for transition in ported.pairwise_conditioner.transitions:
    assert jnp.allclose(transition.swiglu.w_gate.bias, 0.0)
    assert jnp.allclose(transition.swiglu.w_val.bias, 0.0)
    assert jnp.allclose(transition.swiglu.w_out.bias, 0.0)


def test_ported_model_forward_and_score_run(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  """Sanity: the ported model isn't just weight-shaped correctly -- it actually runs."""
  model = _build_model()
  ported, _report = load_pytorch_checkpoint(model, synthetic_state_dict)

  n = 5
  coords = jax.random.normal(jax.random.PRNGKey(1), (n, 3)) * 0.1
  aatype = jnp.zeros((n,), dtype=jnp.int32)
  mask = jnp.ones((n,), dtype=bool)

  energy = ported.energy(coords, aatype, jnp.asarray(0.05), mask)
  assert energy.shape == ()
  assert energy >= 0.0  # sum-of-squares parameterization

  score = ported.score(coords, aatype, jnp.asarray(0.05), mask)
  assert score.shape == (n, 3)


def test_unrecognized_key_raises_instead_of_silently_dropping(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  model = _build_model()
  bad_state_dict = dict(synthetic_state_dict)
  bad_state_dict["model.totally_unknown_future_field.weight"] = np.zeros((3, 3), dtype=np.float32)

  with pytest.raises(ValueError, match="totally_unknown_future_field"):
    load_pytorch_checkpoint(model, bad_state_dict)


def test_shape_mismatch_raises_with_informative_message(synthetic_state_dict: dict[str, np.ndarray]) -> None:
  model = _build_model()
  bad_state_dict = dict(synthetic_state_dict)
  # Simulate a real failure mode: checkpoint's contact_embedding table has a
  # different num_contact_embeddings than the constructed model expects.
  bad_state_dict["model.contact_embedding.weight"] = np.zeros((99, _TOKEN_S), dtype=np.float32)

  with pytest.raises(ValueError, match="contact_embedding.weight"):
    load_pytorch_checkpoint(model, bad_state_dict)
