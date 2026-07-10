"""Energy, score, and auxiliary-score readout heads for ProteinEBM (backlog node E3).

This is the central EBM mechanism: a scalar energy `E_theta` over per-residue
trunk features, and a *conservative* score derived from it by autograd. Ported
from ``~/repos/ProteinEBM/protein_ebm/model/ebm.py``:

* ``ProteinEBM.forward``'s ``a_norm``/``r_update_proj`` tail (``a_norm`` =
  ``nn.LayerNorm(2*token_s)``, ``r_update_proj`` = ``LinearNoBias(2*token_s,
  3)``) -> :class:`EnergyReadout`.
* ``ProteinEBM.compute_energy``'s ``energy = sum_i mask_i * ||r_update_i||**2``
  -> :meth:`EnergyReadout.__call__` / :meth:`EnergyReadout.per_residue_energy`.
* ``ProteinEBM.compute_score``'s conservative branch (``torch.autograd.grad(
  energy.sum(), r_noisy, create_graph=True)``) -> :class:`ScoreReadout`, using
  ``jax.grad`` (JAX's reverse-mode AD is natively higher-order -- there is no
  ``create_graph`` flag to set; nesting ``jax.grad`` again at training time,
  as E8 will, "just works").
* ``ProteinEBM.forward``'s ``r_update_proj_aux`` (the ``aux_score`` branch) ->
  :class:`AuxScoreReadout`: a second, independent linear head from the same
  trunk features, used as a cheap non-conservative score surrogate during
  Langevin simulation (E9, follow-on epic) to avoid backprop-through-gradient
  at every step. **No** ``jax.grad`` is involved in this head.

**Scope boundary (backlog node E1, ``aminx.ebm.trunk``).** This module never
imports or instantiates ``aminx.ebm.trunk.DiffusionTransformer``. The trunk is
a *sibling* concern (composed at backlog node E4, ``host/plan.py``); here, the
map from coordinates to trunk features is always an externally supplied
``trunk_fn: Callable[[Coords, ResidueMask], Float[Array, "N trunk_dim"]]``,
passed at call time to :meth:`ScoreReadout.__call__`/:meth:`ScoreReadout.energy`
(not stored as a field). This keeps :class:`EnergyReadout` the only
learned/differentiable state this module owns, so ``eqx.filter_grad``
naturally reaches its parameters through the full nested-grad path (E8's
training loss differentiates *through* the conservative score, i.e.
reverse-over-reverse AD w.r.t. the trunk's own parameters lives in whatever
outer closure supplies ``trunk_fn`` -- it must **not** be hidden behind a
``static``, non-differentiable field on this module, or the trunk's weights
would be invisible to the outer training gradient).

**``coordinate_scaling`` (design spec §10 MINOR -- the reference's
triple-application footgun).** This module does **not** apply, re-apply, or
otherwise know about ``coordinate_scaling``. It operates purely in whatever
coordinate space its ``coords``/``trunk_out`` inputs already live in. The
single application point is ``aminx.ebm.diffusion.forward_marginal`` (E0,
already implemented and documented there). Any caller composing
:class:`ScoreReadout` with a real trunk must ensure ``coords`` passed here are
already in that scaled space -- do not scale again inside ``trunk_fn``.

**Non-equivariance / random augmentation (design spec §10 MINOR).** The
energy/score readouts here are **not** SO(3)-invariant by construction (the
learned trunk is non-equivariant, per design spec §1). This module has *no
opinion* on augmentation -- it consumes whatever (possibly pre-augmented)
coordinates/trunk output it is given -- but states the contract callers must
honor:

* For **deterministic** scoring (decoy ranking, ΔE biasing -- backlog nodes
  E5/E7): callers must invoke ``aminx.ebm.trunk.center_random_augmentation``
  with ``rotate=False`` *upstream* of ``trunk_fn``/this readout, so repeated
  calls on the same input are bit-for-bit reproducible.
* If augmentation rotation **is** applied somewhere inside a ``trunk_fn`` that
  is differentiated via :class:`ScoreReadout` (i.e. inside the ``jax.grad``
  closure), the rotation must be sampled **outside** that closure and held
  fixed (a constant w.r.t. the gradient) -- never re-sampled per
  ``jax.grad`` evaluation. Re-sampling per evaluation would make the "energy"
  being differentiated a different function on every call, silently
  corrupting the conservative-score identity ``score = -grad_x E``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Array, Float, PRNGKeyArray

  from aminx.ebm.contracts import Coords, Energy, PerResidueEnergy, ResidueMask, Score


class EnergyReadout(eqx.Module):
  """Per-residue vector head + masked sum-of-squares energy.

  Mirrors ``ebm.py``'s ``a_norm`` (``nn.LayerNorm(2*token_s)``) ->
  ``r_update_proj`` (``LinearNoBias(2*token_s, 3)``) tail, then
  ``energy = sum_i mask_i * ||r_i||**2`` (``compute_energy``). Non-negative by
  construction (a sum of squared norms).
  """

  norm: eqx.nn.LayerNorm
  r_proj: eqx.nn.Linear
  trunk_dim: int = eqx.field(static=True)

  def __init__(self, trunk_dim: int, *, key: PRNGKeyArray) -> None:
    """Initialize the energy readout head.

    Args:
        trunk_dim: Feature dimension of the trunk output this head reads
          from (the reference's ``2*token_s``; e.g. 512 for ProteinEBM-x's
          ``token_s=256``).
        key: PRNG key for the (bias-free) ``r_proj`` linear.

    """
    self.trunk_dim = trunk_dim
    self.norm = eqx.nn.LayerNorm(trunk_dim)
    self.r_proj = eqx.nn.Linear(trunk_dim, 3, use_bias=False, key=key)

  def residual_vectors(self, trunk_out: Float[Array, "N trunk_dim"]) -> Float[Array, "N 3"]:
    """Per-residue 3-vector ``r_i = r_proj(LayerNorm(trunk_out_i))``."""
    normed = jax.vmap(self.norm)(trunk_out)
    return jax.vmap(self.r_proj)(normed)

  def per_residue_energy(
    self,
    trunk_out: Float[Array, "N trunk_dim"],
    mask: ResidueMask,
  ) -> PerResidueEnergy:
    """Masked per-residue energy ``mask_i * ||r_i||**2``, prior to the sum reduction."""
    r = self.residual_vectors(trunk_out)
    sq_norm = jnp.sum(r * r, axis=-1)
    return mask.astype(sq_norm.dtype) * sq_norm

  def __call__(self, trunk_out: Float[Array, "N trunk_dim"], mask: ResidueMask) -> Energy:
    """Total energy ``E = sum_i mask_i * ||r_i||**2`` (masked sum over residues)."""
    return jnp.sum(self.per_residue_energy(trunk_out, mask))


class ScoreReadout(eqx.Module):
  """Conservative score ``s(x) = -grad_x E_theta(x)`` -- the central EBM mechanism.

  Wraps an :class:`EnergyReadout`. Composition with the trunk (backlog node
  E1) is deliberately *not* owned by this module: ``trunk_fn`` is supplied at
  call time (see the module docstring's scope-boundary note on why it must
  not be a stored, ``static`` field). This is the exact JAX analog of the
  reference's ``torch.autograd.grad(energy.sum(), r_noisy,
  create_graph=True)[0]`` (negated) in ``ebm.py::compute_score``.
  """

  energy_readout: EnergyReadout

  def energy(
    self,
    coords: Coords,
    mask: ResidueMask,
    trunk_fn: Callable[[Coords, ResidueMask], Float[Array, "N trunk_dim"]],
  ) -> Energy:
    """Energy as a function of coordinates, composed through an external ``trunk_fn``.

    ``trunk_fn(coords, mask) -> trunk_out`` stands in for "encode via E1's
    ``DiffusionTransformer`` (+ conditioning), external to this module".
    """
    trunk_out = trunk_fn(coords, mask)
    return self.energy_readout(trunk_out, mask)

  def __call__(
    self,
    coords: Coords,
    mask: ResidueMask,
    trunk_fn: Callable[[Coords, ResidueMask], Float[Array, "N trunk_dim"]],
  ) -> Score:
    """Conservative score ``-jax.grad(energy)(coords)``, holding ``trunk_fn`` fixed.

    ``trunk_fn`` (and anything it closes over -- e.g. a fixed sampled
    augmentation rotation, per the module docstring) is held constant across
    the gradient evaluation; only ``coords`` is differentiated.
    """
    return -jax.grad(lambda x: self.energy(x, mask, trunk_fn))(coords)


class AuxScoreReadout(eqx.Module):
  """Cheap non-conservative score surrogate -- no ``jax.grad`` involved.

  Mirrors ``ebm.py``'s ``r_update_proj_aux`` (``aux_score`` branch): a second,
  independent ``LayerNorm`` + bias-free linear head reading the *same* trunk
  output as :class:`EnergyReadout`, producing ``r_aux_i`` directly as a plain
  forward pass. Used during Langevin simulation (E9, follow-on epic) to avoid
  backprop-through-gradient at every step.
  """

  norm: eqx.nn.LayerNorm
  r_proj: eqx.nn.Linear
  trunk_dim: int = eqx.field(static=True)

  def __init__(self, trunk_dim: int, *, key: PRNGKeyArray) -> None:
    """Initialize the auxiliary score head.

    Args:
        trunk_dim: Feature dimension of the trunk output this head reads
          from (same ``trunk_out`` as :class:`EnergyReadout`).
        key: PRNG key for the (bias-free) ``r_proj`` linear.

    """
    self.trunk_dim = trunk_dim
    self.norm = eqx.nn.LayerNorm(trunk_dim)
    self.r_proj = eqx.nn.Linear(trunk_dim, 3, use_bias=False, key=key)

  def __call__(self, trunk_out: Float[Array, "N trunk_dim"]) -> Score:
    """Direct (non-conservative) per-residue score surrogate, no autodiff involved."""
    normed = jax.vmap(self.norm)(trunk_out)
    return jax.vmap(self.r_proj)(normed)


__all__ = ["AuxScoreReadout", "EnergyReadout", "ScoreReadout"]
