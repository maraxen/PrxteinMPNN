"""Top-level ProteinEBM-equivalent input embeddings + full forward pass (backlog node **E3.6**).

This is a newly-discovered gap node, inserted between E3 (readout heads) and
E3.5/E8 (weight-port gate / training): the top-level assembly that composes
E1's trunk (``aminx.ebm.trunk``) with E3's readout heads
(``aminx.ebm.readout``) into the actual ``ProteinEBM``-equivalent forward
pass. Ported from ``~/repos/ProteinEBM/protein_ebm/model/ebm.py``
(``ProteinEBM.__init__``, lines ~19-111; ``ProteinEBM.forward``, lines
~114-243).

**Scope (MVP, design spec §9/CA-only).** ``data_dimension = 3`` (CA-only;
``diffuse_sidechain = False``) throughout -- there is no sidechain-diffusion
variant here. Three simplifications relative to the full reference config,
each individually gated/no-op in the reference too:

* ``present_embedding`` (``use_present_embedding``, default ``False`` in
  every shipped config) is **dropped entirely** -- not merely defaulted, the
  field does not exist here. Reintroducing it later is additive (one more
  concatenated embedding + a wider ``single_cond_input_dim``), not a
  breaking change.
* ``atom_mask_embedding`` / all sidechain-diffusion machinery is **dropped**
  (``diffuse_sidechain = False`` in both shipped configs; out of scope for
  E0-E7 per design spec §1 Fork 5).
* ``external_contacts`` always defaults to the all-zeros "no external
  contact" category (design spec §1) -- the reference's
  ``num_contact_embeddings == 3 -> default ones`` special case (a 3-way
  contact-type variant, ``ebm.py:169``) is not reproduced; only the 2-way
  (no-contact / has-contact) default-zero behavior is, matching the
  documented MVP contract.

``use_self_conditioning`` (default ``True`` in the shipped configs) **is**
included: :class:`InputEmbeddings` always builds a
``self_conditioning_embedding`` head, with the reference's
``sc_coords if sc_coords is not None else torch.zeros_like(r_noisy)``
fallback preserved exactly (see :meth:`ProteinEBMModel.trunk_features`).

**CoordinateScalingContract.** This module never applies, re-applies, or
otherwise knows about ``coordinate_scaling`` (design spec §10 MINOR --
the reference's triple-application footgun; same rule as
``aminx.ebm.readout``). The reference's ``forward`` rescales ``r_noisy``/
``sc_coords`` by ``self.diffuser.config.coordinate_scaling`` on entry
(``ebm.py:178-181``, gated behind ``rescale_input_coords=True``) because its
callers pass *raw* physical coordinates. Here, every ``coords``/``sc_coords``
argument is already an ``aminx.ebm.contracts.Coords`` -- i.e. **already
scaled**, per ``aminx.ebm.diffusion.forward_marginal``'s single-application
boundary (E0). So this module's equivalent of ``rescale_input_coords`` is
always "off": callers must pass already-scaled coordinates, and this module
performs no scaling of its own. Do not scale again here.

**Dead reference field: ``s_to_a_linear`` (deliberately NOT ported).**
``ProteinEBM.__init__`` constructs ``self.s_to_a_linear`` (``ebm.py:84-88``,
a zero-init ``LayerNorm -> LinearNoBias`` pair), but ``ProteinEBM.forward``
**never calls it** -- verified by reading ``forward`` in full (``ebm.py:189-
228``): after ``single_conditioner`` produces ``s``, the trunk input is set
directly via ``a = s`` (``ebm.py:219``), with no ``self.s_to_a_linear(s)``
call anywhere in the method. This is dead/unused weight in the upstream
reference itself, not a port artifact. Reproducing it here would add a field
with zero effect on any output (a strict no-op), so it is **omitted**.
Flagged for the future E3.5 weight-port gate: the reference's flat
``state_dict`` will contain extra ``s_to_a_linear.{0,1}.{weight,bias}`` keys
that have no counterpart on :class:`ProteinEBMModel` and should be **dropped**
during the orbax remap, not treated as a missing-key error.

**Resolved: single shared ``a_norm`` vs. two independent readout norms
(a real correctness detail, not stylistic).** The reference applies **one**
shared ``self.a_norm`` (``nn.LayerNorm(2*token_s)``, ``ebm.py:100,226``) to
the raw trunk output ``a``, then feeds that *same* normalized tensor to
**both** ``r_update_proj`` (energy head) and ``r_update_proj_aux`` (aux-score
head) -- one LayerNorm, two linear heads reading its output.
``aminx.ebm.readout.EnergyReadout``/``AuxScoreReadout`` (E3, untouched here)
each already apply their **own**, independently-parameterized internal
``LayerNorm`` before their respective ``r_proj`` (see
``EnergyReadout.residual_vectors``/``AuxScoreReadout.__call__``). Therefore
:meth:`ProteinEBMModel.trunk_features` passes the **raw** (un-normalized)
``token_transformer`` output straight through as ``trunk_out`` -- adding a
third, separate ``a_norm`` step here would double-normalize (LayerNorm is
not idempotent) and does not correspond to anything in the reference. Each
readout's own internal norm plays the ``a_norm`` role for that head. This is
a **documented structural deviation** from the reference (two
independently-learned norms instead of one shared norm feeding two heads --
strictly more free parameters, same shape contract) with a real consequence
for the future E3.5 weight-port gate: the single ``a_norm.{weight,bias}``
tensor in the reference ``state_dict`` has no single counterpart here and
must be broadcast/duplicated (or the two heads' norms independently
fine-tuned) during remap. Not a blocker for this node -- E3's readout
contract already committed to per-head norms; this module simply avoids
compounding that decision with a redundant extra norm.

**Composition order (mirrors ``ebm.py:189-227`` exactly, module names per
E1/E3):** input embeddings -> concat -> ``SingleConditioning`` (produces
``s``, dim ``2*token_s``) -> tile ``s`` into pairwise features -> ``rel_pos``
(``RelativePositionEncoder``) + ``PairwiseConditioning`` -> ``z`` -> Diffusion
``Transformer(a=s, s=s, z=z, mask)`` -> raw trunk features (dim
``2*token_s``) -> ``EnergyReadout``/``ScoreReadout``/``AuxScoreReadout``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from aminx.ebm.readout import AuxScoreReadout, EnergyReadout, ScoreReadout
from aminx.ebm.trunk import (
  DEFAULT_TOKEN_S,
  DEFAULT_TOKEN_Z,
  DEFAULT_TRANSFORMER_DEPTH,
  DEFAULT_TRANSFORMER_HEADS,
  DiffusionTransformer,
  PairwiseConditioning,
  RelativePositionEncoder,
  SingleConditioning,
)

if TYPE_CHECKING:
  from jaxtyping import Array, Float, PRNGKeyArray

  from aminx.ebm.contracts import AAType, Coords, DiffusionTime, Energy, ResidueMask, Score

# CA-only MVP (design spec §9): the diffused variable is always a 3-vector
# per residue; ``diffuse_sidechain``/``data_dimension=37*3`` is out of scope
# (design spec §1 Fork 5).
DATA_DIMENSION = 3

# Default number of contact-embedding categories (the reference's 2-way
# no-contact/has-contact variant; the 3-way variant's ones-default special
# case is not reproduced -- see module docstring).
DEFAULT_NUM_CONTACT_EMBEDDINGS = 2


class InputEmbeddings(eqx.Module):
  """Fuse aatype/coords/contacts/self-conditioning into one per-residue embedding.

  Mirrors ``ProteinEBM.__init__``'s four always-present embedding heads
  (``sequence_embedding``, ``noisy_coord_embedding``, ``contact_embedding``,
  ``self_conditioning_embedding``; ``ebm.py:34-59``) and ``forward``'s
  ``residue_embed = torch.cat([...], dim=-1)`` (``ebm.py:189-196``), with
  ``present_embedding``/``atom_mask_embedding`` dropped (see module
  docstring). Folded into one ``eqx.Module`` (rather than four separate
  top-level classes) since all four heads share the same call-time shape
  contract (``(N,) or (N, data_dimension) -> (N, token_s)``, concatenated
  once) and none is independently reused elsewhere in this epic.
  """

  sequence_embedding: eqx.nn.Embedding
  noisy_coord_embedding: eqx.nn.Linear
  contact_embedding: eqx.nn.Embedding
  self_conditioning_embedding: eqx.nn.Linear

  def __init__(
    self,
    token_s: int,
    num_contact_embeddings: int = DEFAULT_NUM_CONTACT_EMBEDDINGS,
    data_dimension: int = DATA_DIMENSION,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the four input-embedding heads.

    Args:
        token_s: Per-head embedding width (matches ``ProteinEBM``'s
          ``config.token_s``).
        num_contact_embeddings: Number of contact-flag categories (reference
          default 2: no-contact / has-contact).
        data_dimension: Diffused-coordinate dimensionality (3, CA-only).
        key: PRNG key, split across the four embedding heads.

    """
    k_seq, k_coord, k_contact, k_selfcond = jax.random.split(key, 4)
    self.sequence_embedding = eqx.nn.Embedding(21, token_s, key=k_seq)
    self.noisy_coord_embedding = eqx.nn.Linear(data_dimension, token_s, use_bias=False, key=k_coord)
    self.contact_embedding = eqx.nn.Embedding(num_contact_embeddings, token_s, key=k_contact)
    self.self_conditioning_embedding = eqx.nn.Linear(
      data_dimension, token_s, use_bias=False, key=k_selfcond,
    )

  def __call__(
    self,
    aatype: AAType,
    r_noisy: Coords,
    contacts: jax.Array,  # int, shape (N,); see module docstring re: bare-identifier jaxtyping shapes
    sc_coords: Coords,
  ) -> Float[Array, "N four_token_s"]:
    """Concatenate the four embedding heads into one per-residue vector, dim ``4*token_s``."""
    sequence_emb = jax.vmap(self.sequence_embedding)(aatype)
    coord_emb = jax.vmap(self.noisy_coord_embedding)(r_noisy)
    contact_emb = jax.vmap(self.contact_embedding)(contacts)
    self_cond_emb = jax.vmap(self.self_conditioning_embedding)(sc_coords)
    return jnp.concatenate([sequence_emb, coord_emb, contact_emb, self_cond_emb], axis=-1)


class ProteinEBMModel(eqx.Module):
  """Full ProteinEBM-equivalent forward pass: input embeddings -> trunk -> readouts.

  Composes E1's trunk (``SingleConditioning``, ``RelativePositionEncoder``,
  ``PairwiseConditioning``, ``DiffusionTransformer``) with E3's readout heads
  (``EnergyReadout``, ``AuxScoreReadout``; ``ScoreReadout`` is constructed
  on demand in :meth:`score`, not stored -- see below) behind the input
  embeddings defined above. Mirrors ``ProteinEBM.__init__``/``forward``
  exactly except for the documented simplifications/resolutions in the
  module docstring (dropped ``present_embedding``/``atom_mask_embedding``,
  omitted dead ``s_to_a_linear``, no extra ``a_norm`` beyond each readout's
  own).

  ``ScoreReadout`` is deliberately **not** a stored field: it is a thin,
  stateless wrapper around ``energy_readout`` (see
  ``aminx.ebm.readout.ScoreReadout``'s single field). Storing a second,
  independently-constructed ``ScoreReadout(energy_readout=...)`` instance
  alongside ``self.energy_readout`` would alias the same parameter arrays
  under two different pytree paths -- harmless for a forward pass, but a
  footgun for any training loop that later maps a gradient/optimizer state
  over pytree leaves (the aliased weight would appear twice, risking a
  double update). :meth:`score` builds ``ScoreReadout(self.energy_readout)``
  fresh on each call instead; this is cheap (a bare ``eqx.Module`` wrapper
  around an existing instance, no new arrays) and keeps ``energy_readout``
  the single source of truth for those parameters.
  """

  input_embeddings: InputEmbeddings
  single_conditioner: SingleConditioning
  rel_pos: RelativePositionEncoder
  pairwise_conditioner: PairwiseConditioning
  token_transformer: DiffusionTransformer
  energy_readout: EnergyReadout
  aux_score_readout: AuxScoreReadout
  token_s: int = eqx.field(static=True)
  token_z: int = eqx.field(static=True)

  def __init__(
    self,
    token_s: int = DEFAULT_TOKEN_S,
    token_z: int = DEFAULT_TOKEN_Z,
    *,
    dim_fourier: int = 256,
    conditioning_transition_layers: int = 2,
    transformer_depth: int = DEFAULT_TRANSFORMER_DEPTH,
    transformer_heads: int = DEFAULT_TRANSFORMER_HEADS,
    num_contact_embeddings: int = DEFAULT_NUM_CONTACT_EMBEDDINGS,
    r_max: int = 32,
    s_max: int = 1,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the full ProteinEBM-equivalent model.

    Args:
        token_s: Single-representation embedding half-width (``config.
          token_s``; the input-embedding heads and ``SingleConditioning``
          output ``2*token_s``).
        token_z: Pairwise-representation dimension (``config.token_z``).
        dim_fourier: Fourier time-embedding dimension.
        conditioning_transition_layers: ``num_transitions`` shared by
          ``SingleConditioning``/``PairwiseConditioning`` (``config.
          conditioning_transition_layers``).
        transformer_depth: ``DiffusionTransformer`` depth (``config.
          token_transformer_depth``).
        transformer_heads: ``DiffusionTransformer`` heads (``config.
          token_transformer_heads``).
        num_contact_embeddings: Contact-flag category count (default 2).
        r_max: ``RelativePositionEncoder`` clipped residue-distance bound.
        s_max: ``RelativePositionEncoder`` clipped chain-distance bound.
        key: PRNG key, split across every learned sub-module.

    """
    k_embed, k_single, k_relpos, k_pairwise, k_transformer, k_readouts = jax.random.split(key, 6)

    self.input_embeddings = InputEmbeddings(
      token_s, num_contact_embeddings, DATA_DIMENSION, key=k_embed,
    )

    # Always sequence_emb + coord_emb + contact_emb + self_cond_emb == 4 heads
    # (present_embedding/atom_mask_embedding dropped -- module docstring).
    single_cond_input_dim = token_s * 4

    self.single_conditioner = SingleConditioning(
      input_dim=single_cond_input_dim,
      token_s=token_s,
      dim_fourier=dim_fourier,
      num_transitions=conditioning_transition_layers,
      key=k_single,
    )
    self.rel_pos = RelativePositionEncoder(token_z, r_max=r_max, s_max=s_max, key=k_relpos)

    # s (post-SingleConditioning) has dim 2*token_s; tiling+concat over the
    # pairwise axis doubles that again -> 4*token_s (matches ebm.py:78
    # `input_dim=config.token_s*4`).
    self.pairwise_conditioner = PairwiseConditioning(
      input_dim=token_s * 4,
      token_z=token_z,
      dim_token_rel_pos_feats=token_z,
      num_transitions=conditioning_transition_layers,
      key=k_pairwise,
    )
    self.token_transformer = DiffusionTransformer(
      depth=transformer_depth,
      heads=transformer_heads,
      dim=2 * token_s,
      dim_single_cond=2 * token_s,
      dim_pairwise=token_z,
      key=k_transformer,
    )

    k_energy, k_aux = jax.random.split(k_readouts)
    self.energy_readout = EnergyReadout(trunk_dim=2 * token_s, key=k_energy)
    self.aux_score_readout = AuxScoreReadout(trunk_dim=2 * token_s, key=k_aux)

    self.token_s = token_s
    self.token_z = token_z

  def trunk_features(
    self,
    coords: Coords,
    aatype: AAType,
    t: DiffusionTime,
    mask: ResidueMask,
    contacts: jax.Array | None = None,  # int, shape (N,)
    sc_coords: Coords | None = None,
    residue_index: jax.Array | None = None,  # int, shape (N,)
    chain_id: jax.Array | None = None,  # int, shape (N,)
  ) -> Float[Array, "N two_token_s"]:
    """Encode inputs into raw (un-normalized) trunk features, dim ``2*token_s``.

    This is the "encode" half of the composition -- the natural
    ``trunk_fn: Callable[[Coords, ResidueMask], features]`` for
    ``aminx.ebm.readout.ScoreReadout`` (see :meth:`score`, which builds
    exactly that closure from this method).

    Defaults mirror ``ProteinEBM.forward`` (``ebm.py:163-196``):

    * ``contacts=None`` -> all-zeros ("no external contact"; design spec §1).
    * ``sc_coords=None`` -> ``jnp.zeros_like(coords)`` (``ebm.py:194``'s
      ``sc_coords if sc_coords is not None else torch.zeros_like(r_noisy)``).
    * ``residue_index=None`` -> ``jnp.arange(N)`` (sequential residue
      numbering, single fragment).
    * ``chain_id=None`` -> all-zeros (``ebm.py:164-165``, single chain).

    ``coords``/``sc_coords`` are consumed as-is (already ``coordinate_scaling``
    -scaled) -- see the module docstring's ``CoordinateScalingContract``.
    """
    n = coords.shape[0]
    if contacts is None:
      contacts = jnp.zeros((n,), dtype=jnp.int32)
    if sc_coords is None:
      sc_coords = jnp.zeros_like(coords)
    if residue_index is None:
      residue_index = jnp.arange(n)
    if chain_id is None:
      chain_id = jnp.zeros((n,), dtype=jnp.int32)

    residue_embed = self.input_embeddings(aatype, coords, contacts, sc_coords)
    s, _normed_fourier = self.single_conditioner(residue_embed, times=t)

    s_dim = s.shape[-1]
    s_tiled = jnp.concatenate(
      [
        jnp.broadcast_to(s[:, None, :], (n, n, s_dim)),
        jnp.broadcast_to(s[None, :, :], (n, n, s_dim)),
      ],
      axis=-1,
    )
    token_rel_pos_feats = self.rel_pos(residue_index, chain_id)
    z = self.pairwise_conditioner(s_tiled, token_rel_pos_feats)

    # a = s (ebm.py:219), fed through the trunk conditioned on itself.
    return self.token_transformer(s, s, z, mask)

  def energy(
    self,
    coords: Coords,
    aatype: AAType,
    t: DiffusionTime,
    mask: ResidueMask,
    contacts: jax.Array | None = None,  # int, shape (N,)
    sc_coords: Coords | None = None,
    residue_index: jax.Array | None = None,  # int, shape (N,)
    chain_id: jax.Array | None = None,  # int, shape (N,)
  ) -> Energy:
    """Full forward pass to a scalar energy (encode + ``EnergyReadout``)."""
    trunk_out = self.trunk_features(
      coords,
      aatype,
      t,
      mask,
      contacts=contacts,
      sc_coords=sc_coords,
      residue_index=residue_index,
      chain_id=chain_id,
    )
    return self.energy_readout(trunk_out, mask)

  def score(
    self,
    coords: Coords,
    aatype: AAType,
    t: DiffusionTime,
    mask: ResidueMask,
    contacts: jax.Array | None = None,  # int, shape (N,)
    sc_coords: Coords | None = None,
    residue_index: jax.Array | None = None,  # int, shape (N,)
    chain_id: jax.Array | None = None,  # int, shape (N,)
  ) -> Score:
    """Conservative score ``-jax.grad(energy)(coords)`` via ``ScoreReadout``.

    Builds the ``trunk_fn`` closure (holding ``aatype``/``t``/``contacts``/
    ``sc_coords``/``residue_index``/``chain_id`` fixed) and composes it with
    a fresh ``ScoreReadout(self.energy_readout)`` (see the class docstring
    for why this is not a stored field). Only ``coords`` is differentiated.
    """

    def trunk_fn(c: Coords, m: ResidueMask) -> Float[Array, "N two_token_s"]:
      return self.trunk_features(
        c,
        aatype,
        t,
        m,
        contacts=contacts,
        sc_coords=sc_coords,
        residue_index=residue_index,
        chain_id=chain_id,
      )

    score_readout = ScoreReadout(self.energy_readout)
    return score_readout(coords, mask, trunk_fn)

  def aux_score(
    self,
    coords: Coords,
    aatype: AAType,
    t: DiffusionTime,
    mask: ResidueMask,
    contacts: jax.Array | None = None,  # int, shape (N,)
    sc_coords: Coords | None = None,
    residue_index: jax.Array | None = None,  # int, shape (N,)
    chain_id: jax.Array | None = None,  # int, shape (N,)
  ) -> Score:
    """Cheap non-conservative score surrogate via ``AuxScoreReadout`` (no ``jax.grad``)."""
    trunk_out = self.trunk_features(
      coords,
      aatype,
      t,
      mask,
      contacts=contacts,
      sc_coords=sc_coords,
      residue_index=residue_index,
      chain_id=chain_id,
    )
    return self.aux_score_readout(trunk_out)


__all__ = ["DATA_DIMENSION", "DEFAULT_NUM_CONTACT_EMBEDDINGS", "InputEmbeddings", "ProteinEBMModel"]
