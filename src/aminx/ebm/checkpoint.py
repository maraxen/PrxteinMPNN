"""ProteinEBM PyTorch-Lightning checkpoint -> Equinox ``ProteinEBMModel`` weight port.

Backlog node **E3.5** (GATE) -- see
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §2 and design
spec ``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §9. This
is the highest-risk node in the EPIC: a silently-wrong key remap produces
plausible-but-wrong energies with no crash. Every reference key in the
checkpoint's ``state_dict`` is either **explicitly mapped** to a destination
leaf on :class:`aminx.ebm.model.ProteinEBMModel`, or **explicitly matched**
against :data:`SKIPPED_KEY_PATTERNS` with a human-readable reason. Any key
that is neither raises :class:`ValueError` listing every offending key --
this module never silently drops an unrecognized tensor.

**Verified against the actual checkpoint** (``model_6_expert_frozen_1m_md.pt``,
a Lightning ``.pt`` with a ``state_dict`` key prefix of ``"model."`` --
``LightningModule.model = ProteinEBM(...)``; the sibling ``self.diffuser =
R3Diffuser(...)`` is a plain Python object, not an ``nn.Module``, so it
contributes zero ``state_dict`` keys), not guessed from reading
``~/repos/ProteinEBM`` source alone:

* ``eqx.nn.Linear.weight`` is ``(out, in)`` -- **the same convention** as
  ``torch.nn.Linear.weight``. No transpose is needed (confirmed by
  constructing both and comparing shapes: ``eqx.nn.Linear(3, 256).weight.shape
  == (256, 3)``, identical to the reference's ``noisy_coord_embedding.weight``
  shape in the checkpoint).
* ``eqx.nn.Embedding.weight`` is ``(num_embeddings, dim)`` -- also the same
  convention as ``torch.nn.Embedding.weight``. No transpose needed.
* ``eqx.nn.LayerNorm.weight``/``.bias`` are both ``(dim,)``, matching
  ``torch.nn.LayerNorm`` directly.

**Checkpoint-driven construction, not spec-default construction.** This
checkpoint's ``hyper_parameters.config.model`` has
``num_contact_embeddings=3`` (**not** :data:`aminx.ebm.model.
DEFAULT_NUM_CONTACT_EMBEDDINGS` `== 2`) -- callers must construct their
``ProteinEBMModel`` with ``num_contact_embeddings=3`` to match this
checkpoint's ``contact_embedding.weight`` shape ``(3, 256)``, or the shape
check in :func:`load_pytorch_checkpoint` raises before any silent corruption.
``token_s=256, token_z=128, token_transformer_depth=16,
token_transformer_heads=8, dim_fourier=256, conditioning_transition_layers=2``
all match :mod:`aminx.ebm.trunk`/:mod:`aminx.ebm.model`'s defaults exactly for
*this* checkpoint, but a caller porting a different ProteinEBM checkpoint must
re-derive these from that checkpoint's own config, not assume the defaults
carry over.

**Structural deviations requiring explicit remap handling (design spec §10,
``aminx.ebm.model`` module docstring):**

1. ``s_to_a_linear.*`` -- dead code in the reference (constructed,
   never called in ``forward``). Skipped, see :data:`SKIPPED_KEY_PATTERNS`.
2. A single reference ``a_norm.{weight,bias}`` is loaded into **both**
   ``energy_readout.norm`` and ``aux_score_readout.norm`` (our port holds two
   independent norms where the reference shares one) -- handled specially in
   :func:`load_pytorch_checkpoint`, not via the generic 1:1 mapping table.
3. ``DiffusionTransformerLayer.output_projection_linear.*`` and
   ``...output_projection.0.*`` are **the same learned parameter registered
   under two attribute names** in the reference (``self.output_projection =
   nn.Sequential(self.output_projection_linear, nn.Sigmoid())`` aliases the
   already-registered ``self.output_projection_linear``). Verified
   bit-identical against the live checkpoint (``torch.equal(...) == True`` for
   both weight and bias, all 16 layers checked). Only ``.output_projection_linear.*``
   is loaded; ``...output_projection.0.*`` is skipped as redundant (**not**
   missing/dropped functionality).
4. ``ConditionedTransitionBlock.transition.output_projection.0.*`` has **no**
   such duplicate (that class never separately names the linear) -- loaded
   directly, distinct code path from (3).
5. The reference's ``Transition.{fc1,fc2,fc3}`` are bias-free
   (``nn.Linear(..., bias=False)``), but this port reuses
   ``aminx.model.diffusion_mpnn.SwiGLU`` for :class:`aminx.ebm.trunk.Transition`,
   whose three ``eqx.nn.Linear`` sublayers all carry biases (``trunk.py``
   module docstring flags this and defers resolution to this gate). Those 12
   extra bias leaves (``w_gate``/``w_val``/``w_out`` x 2 transitions x 2
   conditioners) have **no reference counterpart to load** -- they are
   explicitly zeroed post-remap (see ``_zero_extra_swiglu_biases``) so the
   ported model's forward pass exactly reproduces
   ``silu(fc1(x)) * fc2(x)`` -> ``fc3(x)`` with no bias terms, matching the
   reference bit-for-bit rather than leaving stale random-init biases in
   place.
6. ``sidechain_proj.*`` -- this checkpoint has ``predict_sidechain=True``
   (all-atom aux head), out of MVP scope for E3-E7 (design spec Fork 5).
   Skipped, see :data:`SKIPPED_KEY_PATTERNS`.

Any OTHER key present in a checkpoint's ``state_dict`` that this module does
not recognize (e.g. ``present_embedding.*``/``atom_mask_embedding.*`` -- both
absent from *this* checkpoint's config, ``use_present_embedding``/
``diffuse_sidechain`` both false/unset -- would appear if a different
checkpoint enabled them) is **not** silently dropped: :func:`load_pytorch_checkpoint`
raises, listing every such key, so a future checkpoint with a different
config shape fails loudly instead of porting a subset of its weights.
"""

from __future__ import annotations

import dataclasses
import re
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
  from collections.abc import Callable, Mapping

  import jax

  from aminx.ebm.model import ProteinEBMModel

_MODEL_PREFIX = "model."

# ---------------------------------------------------------------------------
# Intentionally-skipped reference keys -- regexes over the *stripped*
# (prefix-removed) checkpoint key. Every entry here is a deliberate,
# documented non-load, not an oversight. See the module docstring for the
# full itemized rationale.
# ---------------------------------------------------------------------------

SKIPPED_KEY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
  (
    re.compile(r"^s_to_a_linear\."),
    "dead code in the reference: ProteinEBM.__init__ constructs "
    "s_to_a_linear (ebm.py:84-88) but ProteinEBM.forward never calls it "
    "(verified by reading forward in full, ebm.py:114-243) -- see "
    "aminx.ebm.model module docstring.",
  ),
  (
    re.compile(r"^sidechain_proj\."),
    "this checkpoint has predict_sidechain=True (all-atom/sidechain aux "
    "head), which is out of MVP scope for E3-E7 (design spec Fork 5: "
    "all-atom aux head dropped until E8).",
  ),
  (
    # Deliberately anchored right after "layers.<i>." (no ".transition" in
    # between) so this does NOT also match
    # "token_transformer.layers.<i>.transition.output_projection.0.*" --
    # that ConditionedTransitionBlock key has no such duplicate and IS loaded
    # (see _add_conditioned_transition_block_keys).
    re.compile(r"^token_transformer\.layers\.\d+\.output_projection\.0\."),
    "redundant alias: verified bit-identical (torch.equal) against the "
    "same layer's '...output_projection_linear.*' key -- the reference "
    "registers self.output_projection_linear AND self.output_projection = "
    "nn.Sequential(self.output_projection_linear, nn.Sigmoid()), so both "
    "attribute names point at the same nn.Parameter. The "
    "'...output_projection_linear.*' form is the one actually loaded (see "
    "_build_mapping); this key is reference-side duplication, not "
    "missing/dropped functionality.",
  ),
)


def _match_skip_reason(key: str) -> str | None:
  """Return the documented skip reason for ``key``, or ``None`` if unmatched."""
  for pattern, reason in SKIPPED_KEY_PATTERNS:
    if pattern.match(key):
      return reason
  return None


def _to_jax(tensor: Any) -> jax.Array:  # noqa: ANN401 -- duck-typed torch.Tensor, torch is dev-only
  """Convert a (duck-typed) ``torch.Tensor`` to a ``float32`` JAX array.

  Deliberately does not import ``torch`` (an optional/dev-only aminx
  dependency) -- any object exposing ``.detach().cpu().numpy()`` (a real
  ``torch.Tensor``) or a plain array-like (for test fixtures) works via this
  duck-typed conversion.
  """
  array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
  return jnp.asarray(np.asarray(array, dtype=np.float32))


def _add_plain_transition_keys(
  add: Callable[[str, Callable[[ProteinEBMModel], jax.Array | None]], None],
  ref_prefix: str,
  select_transition: Callable[[ProteinEBMModel], Any],
) -> None:
  """Wire one reference ``Transition`` (``norm``/``fc1``/``fc2``/``fc3``) onto ours.

  Ours (``aminx.ebm.trunk.Transition``) holds ``norm`` + ``swiglu``
  (``aminx.model.diffusion_mpnn.SwiGLU``'s ``w_gate``/``w_val``/``w_out``);
  the reference's ``x = silu(fc1(x)) * fc2(x); x = fc3(x)`` maps
  ``fc1 -> w_gate`` (silu'd), ``fc2 -> w_val``, ``fc3 -> w_out`` (module
  docstring point 5 -- the three extra biases this creates are zeroed
  separately, not loaded here).
  """
  add(f"{ref_prefix}.norm.weight", lambda mdl: select_transition(mdl).norm.weight)
  add(f"{ref_prefix}.norm.bias", lambda mdl: select_transition(mdl).norm.bias)
  add(f"{ref_prefix}.fc1.weight", lambda mdl: select_transition(mdl).swiglu.w_gate.weight)
  add(f"{ref_prefix}.fc2.weight", lambda mdl: select_transition(mdl).swiglu.w_val.weight)
  add(f"{ref_prefix}.fc3.weight", lambda mdl: select_transition(mdl).swiglu.w_out.weight)


def _add_conditioned_transition_block_keys(
  add: Callable[[str, Callable[[ProteinEBMModel], jax.Array | None]], None],
  ref_prefix: str,
  select_block: Callable[[ProteinEBMModel], Any],
) -> None:
  """Wire one reference ``ConditionedTransitionBlock`` onto ours (module docstring point 4)."""
  add(f"{ref_prefix}.adaln.s_norm.weight", lambda mdl: select_block(mdl).adaln.s_norm.weight)
  add(f"{ref_prefix}.adaln.s_scale.weight", lambda mdl: select_block(mdl).adaln.s_scale.weight)
  add(f"{ref_prefix}.adaln.s_scale.bias", lambda mdl: select_block(mdl).adaln.s_scale.bias)
  add(f"{ref_prefix}.adaln.s_bias.weight", lambda mdl: select_block(mdl).adaln.s_bias.weight)
  add(f"{ref_prefix}.swish_gate.0.weight", lambda mdl: select_block(mdl).swish_gate_proj.weight)
  add(f"{ref_prefix}.a_to_b.weight", lambda mdl: select_block(mdl).a_to_b.weight)
  add(f"{ref_prefix}.b_to_a.weight", lambda mdl: select_block(mdl).b_to_a.weight)
  add(
    f"{ref_prefix}.output_projection.0.weight",
    lambda mdl: select_block(mdl).output_projection.weight,
  )
  add(
    f"{ref_prefix}.output_projection.0.bias",
    lambda mdl: select_block(mdl).output_projection.bias,
  )


def _not_none(
  key: str,
  selector: Callable[[ProteinEBMModel], jax.Array | None],
) -> Callable[[ProteinEBMModel], jax.Array]:
  """Narrow a leaf selector's statically-Optional return type (``ty``, not a runtime concern).

  ``eqx.nn.Linear.bias``/``eqx.nn.LayerNorm.weight``/``.bias`` are typed
  ``Array | None`` because those modules *support* a bias-free/weight-free
  construction -- but every submodule ``_build_mapping``/``_special_case_entries``
  selects into here was constructed (in ``aminx.ebm.trunk``/``readout``/``model``)
  with that parameter present, so the leaf is never actually ``None`` at
  runtime. A ``None`` here would mean the destination module was constructed
  wrong (a real port bug), so this raises rather than silently coercing.
  """

  def wrapped(model: ProteinEBMModel) -> jax.Array:
    leaf = selector(model)
    if leaf is None:
      msg = f"{key}: destination leaf is None -- the destination sublayer was constructed without this weight/bias (real port bug, not a checkpoint issue)."
      raise ValueError(msg)
    return leaf

  return wrapped


def _build_mapping(model: ProteinEBMModel) -> dict[str, Callable[[ProteinEBMModel], jax.Array]]:
  """Build ``{stripped_reference_key: leaf_selector}`` for every 1:1-mappable key.

  Excludes the two special-cased, non-1:1 keys (``a_norm.{weight,bias}`` --
  one source, two destinations; ``single_conditioner.fourier_embed.proj.
  {weight,bias}`` -- the ``.weight`` needs a shape squeeze), which
  :func:`load_pytorch_checkpoint` handles directly.
  """
  mapping: dict[str, Callable[[ProteinEBMModel], jax.Array]] = {}

  def add(key: str, selector: Callable[[ProteinEBMModel], jax.Array | None]) -> None:
    mapping[key] = _not_none(key, selector)

  add("sequence_embedding.weight", lambda mdl: mdl.input_embeddings.sequence_embedding.weight)
  add(
    "noisy_coord_embedding.weight",
    lambda mdl: mdl.input_embeddings.noisy_coord_embedding.weight,
  )
  add("contact_embedding.weight", lambda mdl: mdl.input_embeddings.contact_embedding.weight)
  add(
    "self_conditioning_embedding.weight",
    lambda mdl: mdl.input_embeddings.self_conditioning_embedding.weight,
  )

  add(
    "single_conditioner.norm_single.weight",
    lambda mdl: mdl.single_conditioner.norm_single.weight,
  )
  add("single_conditioner.norm_single.bias", lambda mdl: mdl.single_conditioner.norm_single.bias)
  add(
    "single_conditioner.single_embed.weight",
    lambda mdl: mdl.single_conditioner.single_embed.weight,
  )
  add("single_conditioner.single_embed.bias", lambda mdl: mdl.single_conditioner.single_embed.bias)
  add(
    "single_conditioner.norm_fourier.weight",
    lambda mdl: mdl.single_conditioner.norm_fourier.weight,
  )
  add("single_conditioner.norm_fourier.bias", lambda mdl: mdl.single_conditioner.norm_fourier.bias)
  add(
    "single_conditioner.fourier_to_single.weight",
    lambda mdl: mdl.single_conditioner.fourier_to_single.weight,
  )
  for i in range(len(model.single_conditioner.transitions)):
    _add_plain_transition_keys(
      add,
      f"single_conditioner.transitions.{i}",
      lambda mdl, i=i: mdl.single_conditioner.transitions[i],
    )

  add("rel_pos.linear_layer.weight", lambda mdl: mdl.rel_pos.linear.weight)

  add(
    "pairwise_conditioner.dim_pairwise_init_proj.0.weight",
    lambda mdl: mdl.pairwise_conditioner.init_norm.weight,
  )
  add(
    "pairwise_conditioner.dim_pairwise_init_proj.0.bias",
    lambda mdl: mdl.pairwise_conditioner.init_norm.bias,
  )
  add(
    "pairwise_conditioner.dim_pairwise_init_proj.1.weight",
    lambda mdl: mdl.pairwise_conditioner.init_proj.weight,
  )
  for i in range(len(model.pairwise_conditioner.transitions)):
    _add_plain_transition_keys(
      add,
      f"pairwise_conditioner.transitions.{i}",
      lambda mdl, i=i: mdl.pairwise_conditioner.transitions[i],
    )

  for i in range(len(model.token_transformer.layers)):
    prefix = f"token_transformer.layers.{i}"
    add(
      f"{prefix}.adaln.s_norm.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].adaln.s_norm.weight,
    )
    add(
      f"{prefix}.adaln.s_scale.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].adaln.s_scale.weight,
    )
    add(
      f"{prefix}.adaln.s_scale.bias",
      lambda mdl, i=i: mdl.token_transformer.layers[i].adaln.s_scale.bias,
    )
    add(
      f"{prefix}.adaln.s_bias.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].adaln.s_bias.weight,
    )

    add(
      f"{prefix}.pair_bias_attn.proj_q.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.proj_q.weight,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_q.bias",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.proj_q.bias,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_k.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.proj_k.weight,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_v.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.proj_v.weight,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_g.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.proj_g.weight,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_z.0.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.pair_norm.weight,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_z.0.bias",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.pair_norm.bias,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_z.1.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.pair_proj.weight,
    )
    add(
      f"{prefix}.pair_bias_attn.proj_o.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].pair_bias_attn.proj_o.weight,
    )

    add(
      f"{prefix}.output_projection_linear.weight",
      lambda mdl, i=i: mdl.token_transformer.layers[i].output_projection.weight,
    )
    add(
      f"{prefix}.output_projection_linear.bias",
      lambda mdl, i=i: mdl.token_transformer.layers[i].output_projection.bias,
    )

    _add_conditioned_transition_block_keys(
      add,
      f"{prefix}.transition",
      lambda mdl, i=i: mdl.token_transformer.layers[i].transition,
    )

  add("r_update_proj.weight", lambda mdl: mdl.energy_readout.r_proj.weight)
  add("r_update_proj_aux.weight", lambda mdl: mdl.aux_score_readout.r_proj.weight)

  return mapping


def _extra_swiglu_bias_selectors(
  model: ProteinEBMModel,
) -> list[Callable[[ProteinEBMModel], jax.Array]]:
  """Selectors for the 12 ``SwiGLU`` biases with no reference counterpart (docstring point 5)."""
  selectors: list[Callable[[ProteinEBMModel], jax.Array]] = []
  for i in range(len(model.single_conditioner.transitions)):
    selectors += [
      _not_none(
        f"single_conditioner.transitions[{i}].swiglu.w_gate.bias",
        lambda mdl, i=i: mdl.single_conditioner.transitions[i].swiglu.w_gate.bias,
      ),
      _not_none(
        f"single_conditioner.transitions[{i}].swiglu.w_val.bias",
        lambda mdl, i=i: mdl.single_conditioner.transitions[i].swiglu.w_val.bias,
      ),
      _not_none(
        f"single_conditioner.transitions[{i}].swiglu.w_out.bias",
        lambda mdl, i=i: mdl.single_conditioner.transitions[i].swiglu.w_out.bias,
      ),
    ]
  for i in range(len(model.pairwise_conditioner.transitions)):
    selectors += [
      _not_none(
        f"pairwise_conditioner.transitions[{i}].swiglu.w_gate.bias",
        lambda mdl, i=i: mdl.pairwise_conditioner.transitions[i].swiglu.w_gate.bias,
      ),
      _not_none(
        f"pairwise_conditioner.transitions[{i}].swiglu.w_val.bias",
        lambda mdl, i=i: mdl.pairwise_conditioner.transitions[i].swiglu.w_val.bias,
      ),
      _not_none(
        f"pairwise_conditioner.transitions[{i}].swiglu.w_out.bias",
        lambda mdl, i=i: mdl.pairwise_conditioner.transitions[i].swiglu.w_out.bias,
      ),
    ]
  return selectors


@dataclasses.dataclass(frozen=True)
class CheckpointPortReport:
  """Itemized record of what :func:`load_pytorch_checkpoint` did with every checkpoint key.

  Plain Python metadata (not a JAX pytree) -- purely for reporting/audit, per
  the E3.5 gate's requirement to make every skipped key explicit rather than
  silently swallowed.
  """

  loaded_keys: tuple[str, ...]
  """Reference keys (post ``"model."`` strip) actually loaded into the model."""

  skipped_keys: tuple[tuple[str, str], ...]
  """``(key, reason)`` pairs for reference keys intentionally not loaded."""

  zeroed_bias_paths: tuple[str, ...]
  """Destination paths explicitly zeroed (no reference counterpart to load)."""


# A (label, selector, value) triple destined for one eqx.tree_at replace pass.
_Entry = tuple[str, "Callable[[ProteinEBMModel], jax.Array]", "jax.Array"]


def _special_case_entries(key: str, tensor: Any) -> list[_Entry] | None:  # noqa: ANN401
  """Handle the non-1:1 keys (one source -> two destinations, or a shape squeeze).

  Returns ``None`` if ``key`` is not one of these special cases (caller then
  falls back to the generic 1:1 ``_build_mapping`` table).
  """
  if key == "a_norm.weight":
    value = _to_jax(tensor)
    return [
      (
        "a_norm.weight -> energy_readout.norm.weight",
        _not_none("energy_readout.norm.weight", lambda mdl: mdl.energy_readout.norm.weight),
        value,
      ),
      (
        "a_norm.weight -> aux_score_readout.norm.weight",
        _not_none("aux_score_readout.norm.weight", lambda mdl: mdl.aux_score_readout.norm.weight),
        value,
      ),
    ]
  if key == "a_norm.bias":
    value = _to_jax(tensor)
    return [
      (
        "a_norm.bias -> energy_readout.norm.bias",
        _not_none("energy_readout.norm.bias", lambda mdl: mdl.energy_readout.norm.bias),
        value,
      ),
      (
        "a_norm.bias -> aux_score_readout.norm.bias",
        _not_none("aux_score_readout.norm.bias", lambda mdl: mdl.aux_score_readout.norm.bias),
        value,
      ),
    ]
  if key == "single_conditioner.fourier_embed.proj.weight":
    # Reference nn.Linear(1, dim).weight is (dim, 1); ours stores the
    # squeezed (dim,) vector directly (see aminx.ebm.trunk.FourierEmbedding).
    value = _to_jax(tensor).squeeze(-1)
    return [(key, lambda mdl: mdl.single_conditioner.fourier_embed.weight, value)]
  if key == "single_conditioner.fourier_embed.proj.bias":
    value = _to_jax(tensor)
    return [(key, lambda mdl: mdl.single_conditioner.fourier_embed.bias, value)]
  return None


def _classify_checkpoint_keys(
  model: ProteinEBMModel,
  stripped: Mapping[str, Any],
) -> tuple[list[_Entry], list[str], list[tuple[str, str]]]:
  """Split every stripped checkpoint key into (entries, loaded_keys, skipped_keys).

  Raises ``ValueError`` listing every key that is neither mapped nor
  explicitly skipped -- see the module docstring's non-silent-drop contract.
  """
  mapping = _build_mapping(model)
  entries: list[_Entry] = []
  loaded_keys: list[str] = []
  skipped_keys: list[tuple[str, str]] = []
  unmapped_keys: list[str] = []

  for key, tensor in stripped.items():
    special = _special_case_entries(key, tensor)
    if special is not None:
      entries.extend(special)
      loaded_keys.append(key)
      continue
    if key in mapping:
      entries.append((key, mapping[key], _to_jax(tensor)))
      loaded_keys.append(key)
      continue
    reason = _match_skip_reason(key)
    if reason is not None:
      skipped_keys.append((key, reason))
      continue
    unmapped_keys.append(key)

  if unmapped_keys:
    msg = (
      f"load_pytorch_checkpoint: refusing to silently drop {len(unmapped_keys)} "
      "unrecognized reference key(s) -- this is exactly the "
      "silently-wrong-remap failure mode the E3.5 gate exists to prevent. "
      "Add an explicit mapping (_build_mapping) or an explicit documented "
      "skip pattern (SKIPPED_KEY_PATTERNS) for:\n  " + "\n  ".join(sorted(unmapped_keys))
    )
    raise ValueError(msg)

  return entries, loaded_keys, skipped_keys


def _validate_shapes(model: ProteinEBMModel, entries: list[_Entry]) -> None:
  """Raise ``ValueError`` if any entry's checkpoint tensor shape mismatches its destination leaf."""
  shape_errors = [
    f"{label}: model leaf shape {selector(model).shape} != checkpoint tensor shape {value.shape}"
    for label, selector, value in entries
    if selector(model).shape != value.shape
  ]
  if shape_errors:
    msg = (
      "load_pytorch_checkpoint: shape mismatch(es) -- construct `model` with "
      "dimensions matching this checkpoint:\n  " + "\n  ".join(shape_errors)
    )
    raise ValueError(msg)


def _apply_entries(model: ProteinEBMModel, entries: list[_Entry]) -> ProteinEBMModel:
  """Replace every ``entries`` leaf on ``model`` in a single ``eqx.tree_at`` pass."""

  def where(m: ProteinEBMModel) -> list[jax.Array]:
    return [selector(m) for _, selector, _ in entries]

  replace = [value for _, _, value in entries]
  return eqx.tree_at(where, model, replace=replace)


def _zero_extra_swiglu_biases(model: ProteinEBMModel) -> ProteinEBMModel:
  """Zero the 12 ``SwiGLU`` biases with no reference counterpart (docstring point 5)."""
  bias_selectors = _extra_swiglu_bias_selectors(model)

  def where_bias(m: ProteinEBMModel) -> list[jax.Array]:
    return [selector(m) for selector in bias_selectors]

  zero_values = [jnp.zeros_like(selector(model)) for selector in bias_selectors]
  return eqx.tree_at(where_bias, model, replace=zero_values)


def load_pytorch_checkpoint(
  model: ProteinEBMModel,
  torch_state_dict: Mapping[str, Any],
  *,
  prefix: str = _MODEL_PREFIX,
) -> tuple[ProteinEBMModel, CheckpointPortReport]:
  """Port a ProteinEBM PyTorch-Lightning ``state_dict`` onto a fresh ``ProteinEBMModel``.

  Args:
      model: A freshly-constructed ``ProteinEBMModel`` whose dimensions
        (``token_s``, ``token_z``, ``transformer_depth``, ``transformer_heads``,
        ``num_contact_embeddings``, ...) already match the checkpoint's own
        config -- construct this by reading the checkpoint's
        ``hyper_parameters.config.model`` (or by inspecting tensor shapes)
        *before* calling this function. This function only replaces leaf
        arrays; it never changes ``model``'s static shape/depth structure, so
        a dimension mismatch surfaces as an explicit shape-mismatch
        ``ValueError`` here rather than a silent partial load.
      torch_state_dict: The checkpoint's flat ``state_dict`` mapping (e.g.
        ``torch.load(path, ...)["state_dict"]``), with or without the
        Lightning ``"model."`` key prefix.
      prefix: The submodule prefix to strip from every key before matching
        (default ``"model."``, matching ``LightningModule.model = ProteinEBM(...)``).

  Returns:
      ``(ported_model, report)`` -- the weight-replaced model, plus an
      itemized :class:`CheckpointPortReport` of every key's disposition.

  Raises:
      ValueError: If any checkpoint key is neither explicitly mapped nor
        explicitly skipped (see :data:`SKIPPED_KEY_PATTERNS`), or if a
        mapped key's checkpoint tensor shape does not match the
        corresponding destination leaf's shape on ``model``.

  """
  stripped = {key.removeprefix(prefix): tensor for key, tensor in torch_state_dict.items()}

  entries, loaded_keys, skipped_keys = _classify_checkpoint_keys(model, stripped)
  _validate_shapes(model, entries)

  ported = _apply_entries(model, entries)
  ported = _zero_extra_swiglu_biases(ported)

  report = CheckpointPortReport(
    loaded_keys=tuple(loaded_keys),
    skipped_keys=tuple(skipped_keys),
    zeroed_bias_paths=(
      "single_conditioner.transitions[*].swiglu.{w_gate,w_val,w_out}.bias",
      "pairwise_conditioner.transitions[*].swiglu.{w_gate,w_val,w_out}.bias",
    ),
  )
  return ported, report


__all__ = [
  "SKIPPED_KEY_PATTERNS",
  "CheckpointPortReport",
  "load_pytorch_checkpoint",
]
