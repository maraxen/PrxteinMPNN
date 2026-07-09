"""Learned DiffusionTransformer trunk for the ProteinEBM energy/score path.

Mechanical PyTorch -> Equinox port of the Boltz-1-style diffusion transformer
used by ProteinEBM (backlog node **E1**; see
``.praxia/docs/plans/260709_proteinebm-epic-backlog-dag.md`` §2 and design
spec ``.praxia/docs/specs/260709_proteinebm-aminx-decomposition.md`` §3.5).
Ported from:

* ``~/repos/ProteinEBM/protein_ebm/model/layers.py`` -- ``AdaLN``,
  ``ConditionedTransitionBlock``, ``AttentionPairBias``,
  ``DiffusionTransformerLayer``, ``DiffusionTransformer``, ``Transition``,
  ``FourierEmbedding``, ``RelativePositionEncoder``, ``SingleConditioning``,
  ``PairwiseConditioning``.
* ``~/repos/ProteinEBM/protein_ebm/model/boltz_utils.py`` -- the quaternion /
  ``center_random_augmentation`` SO(3) utilities.

**Scope.** This module is the LEARNED transformer trunk only -- the pure
VP-SDE math lives in ``aminx.ebm.diffusion`` (E0, untouched). The energy/score
readout heads (``EnergyReadout``/``ScoreReadout``/``AuxScoreReadout``) are
backlog node **E3** and are *not* implemented here.

**Port conventions (design spec §3.5, "mechanical, no new math"):**

* ``torch.einsum`` -> ``jnp.einsum``; einops works over JAX arrays unchanged.
* In-place PyTorch inits (``nn.init.zeros_``, ``nn.init.constant_``) become
  Equinox init-time construction via ``eqx.tree_at`` (see
  :func:`_zero_init_linear`).
* ``torch.autocast("cuda", enabled=False)`` (the numerics-sensitive attention
  block) becomes an explicit ``.astype(jnp.float32)`` upcast around the
  softmax/einsum in :class:`AttentionPairBias`.
* No custom CUDA kernels exist in the reference core path (confirmed in the
  design spec's port-surface recon, §4) -- nothing here needed a bespoke
  kernel.

**No batch dimension.** Matching the existing aminx convention
(``model/encoder.py``, ``model/decoder.py``: a single structure, ``N``
residues, batching happens via ``jax.vmap`` at call sites, not inside the
module), every module here operates on a *single* structure: single reps are
``(N, D)``, pair reps are ``(N, N, D)``. The reference's leading batch axis
``B`` is dropped; callers ``vmap`` over structures exactly as
``host/kernel_dispatch.py`` already does for the logit path (``Vmap``/
``SafeMap`` strategies, per design spec §3.3).

**Dropped/simplified vs. the reference (documented, not silent):**

* ``AttentionPairBias``'s ``to_keys`` windowed-attention chunking optimization
  is dropped -- irrelevant at the bucketed lengths this epic targets
  (``{64,128,256,512}``, spec Fork 6) and orthogonal to E1's job (trunk
  correctness, not a memory optimization for very large structures).
* ``mask`` is a **required** argument throughout (the reference's
  ``mask=None`` default on ``AttentionPairBias``/``DiffusionTransformerLayer``
  would itself crash on the ``(1 - mask...)`` arithmetic if actually
  exercised with ``None`` -- tightening the contract here doesn't change any
  working reference behavior).
* ``DiffusionTransformer``'s ``activation_checkpointing`` flag is dropped: in
  the reference it is a genuine no-op (``layers.py:409-426`` -- both branches
  of the ``if activation_checkpointing`` append the *identical*
  ``DiffusionTransformerLayer``, so the flag never changes forward behavior).
  Real ``jax.checkpoint`` wrapping of the 85M trunk is an E8 training-time
  concern (design spec Fork 2 / MAJOR-6), not this inference-shape port.

**``FourierEmbedding`` vs. ``model/diffusion_mpnn.py``'s ``SinusoidalEmbedding``
-- NOT equivalent, both kept.** ``SinusoidalEmbedding`` is a deterministic,
multi-frequency embedding: ``concat(cos(t*f_k), sin(t*f_k))`` over a
log-spaced frequency ladder ``f_k``, no learned/random parameters.
``FourierEmbedding`` (ported here) is a **fixed random Fourier feature** map:
``cos(2*pi*(w*t + b))`` with ``w, b ~ N(0, 1)`` sampled once at init and
frozen (``self.proj.requires_grad_(False)`` in the reference, ``layers.py``
:class:`FourierEmbedding` constructor) -- cosine-only, includes a random
phase/bias, and is never updated by training. These are mathematically
distinct constructions serving the same *role* (time embedding) with
different statistics; reusing ``SinusoidalEmbedding`` here would silently
change ProteinEBM's time-conditioning distribution, so :class:`FourierEmbedding`
is ported as its own module. Its ``weight``/``bias`` fields are ordinary
(non-static) pytree leaves for now; **E8's training loop must exclude them
from the trainable/optimizer leaves** (``eqx.filter`` on a custom filter
spec) to reproduce the reference's frozen-parameter behavior -- flagged here,
not solved here (out of scope for E1).

**Reused, not reimplemented (per task brief):** ``model/diffusion_mpnn.py``'s
``SwiGLU`` class is reused directly for :class:`Transition` (an exact
structural match: ``LayerNorm`` -> ``silu(fc1(x)) * fc2(x)`` -> ``fc3`` is
precisely ``SwiGLU``'s ``w_out(silu(w_gate(x)) * w_val(x))``). **Caveat:**
the aminx ``SwiGLU`` class's three ``eqx.nn.Linear`` sublayers all carry
biases, while the reference ``Transition``'s ``fc1``/``fc2``/``fc3`` are bias-
free (``bias=False``, ``layers.py:480-482``). This is a structural deviation
for the sake of reuse (a `Linear` bias-toggle isn't exposed by the existing
class, and it lives outside this backlog node's touch-scope) -- it does not
change E1's shape/behavior contracts, but it **is** relevant to the E3.5
weight-port parity gate: the remap script will need to either zero those
extra bias leaves or accept a non-bias-free `Transition`. Flagged for E3.5,
not resolved here.

``ConditionedTransitionBlock``'s gating is genuinely *not* a two-way SwiGLU:
``b = swish_gate(a) * a_to_b(a)`` is a three-way product (chunked
gate/value from one projection, times a *second, independent* projection),
distinct from the aminx ``SwiGLU``'s two-way ``silu(gate) * val``. Reusing
the aminx class here would silently drop the third (``a_to_b``) term -- a
correctness bug, not a mechanical port. So :class:`ConditionedTransitionBlock`
implements its own (bias-free) linears + inline ``silu``-chunk gate, matching
``layers.py:132-175`` exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange

from aminx.model.diffusion_mpnn import SwiGLU

if TYPE_CHECKING:
  from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

  from aminx.ebm.contracts import Coords, DiffusionTime, ResidueMask

# ProteinEBM-x defaults (design spec §2): token_s=256, token_z=128,
# token_transformer_depth=16, token_transformer_heads=8. These are the
# project-specific instantiation values, distinct from the generic Boltz-1
# defaults still used verbatim on ``SingleConditioning``/``PairwiseConditioning``
# (``input_dim=384``, ``token_s=384``, `layers.py`) to keep those two classes a
# byte-faithful mechanical port of their reference constructors.
DEFAULT_TOKEN_S = 256
DEFAULT_TOKEN_Z = 128
DEFAULT_TRANSFORMER_DEPTH = 16
DEFAULT_TRANSFORMER_HEADS = 8


def _zero_init_linear(layer: eqx.nn.Linear, *, bias_value: float | None = None) -> eqx.nn.Linear:
  """Zero a freshly constructed ``Linear``'s weight (Equinox analog of ``final_init_``).

  Mirrors the reference's in-place ``nn.init.zeros_(weight)`` +
  ``nn.init.constant_(bias, v)`` pattern (e.g. ``layers.py:161-162``,
  ``layers.py:235`` ``final_init_``, ``layers.py:339-340``) as init-time
  construction: Equinox modules are immutable, so this runs once, right
  after ``eqx.nn.Linear`` construction, inside a module's ``__init__``.
  """
  layer = eqx.tree_at(lambda linear: linear.weight, layer, jnp.zeros_like(layer.weight))
  if bias_value is not None and layer.bias is not None:
    layer = eqx.tree_at(lambda linear: linear.bias, layer, jnp.full_like(layer.bias, bias_value))
  return layer


class FourierEmbedding(eqx.Module):
  """Fixed random-Fourier-feature time embedding.

  Ported from ``layers.py:527-552``. ``weight``/``bias`` are drawn once at
  init (``N(0, 1)``) and are **frozen** in the reference
  (``self.proj.requires_grad_(False)``); see the module docstring's note on
  excluding them from E8's optimizer leaves.
  """

  # Shape `(dim,)`; plain `jax.Array` (not an inline jaxtyping single-axis
  # literal) -- deliberate, see the module docstring's note on this
  # environment's auto-fixer stripping quotes from bare-identifier jaxtyping
  # shape strings (e.g. `Float[Array, "dim"]` -> `Float[Array, dim]`, an
  # F821/NameError landmine). Multi-axis strings (`"N N dim"` etc.) and
  # imported contract aliases (`ResidueMask`, `Coords`, ...) are unaffected
  # and used everywhere they apply.
  weight: jax.Array
  bias: jax.Array
  dim: int = eqx.field(static=True)

  def __init__(self, dim: int, *, key: PRNGKeyArray) -> None:
    """Initialize the fixed random Fourier projection.

    Args:
        dim: Output embedding dimension.
        key: PRNG key; consumed once (split internally) for the frozen
          weight/bias draw.

    """
    k_w, k_b = jax.random.split(key)
    self.weight = jax.random.normal(k_w, (dim,))
    self.bias = jax.random.normal(k_b, (dim,))
    self.dim = dim

  def __call__(self, t: DiffusionTime) -> jax.Array:  # shape (dim,)
    """Compute ``cos(2*pi*(weight*t + bias))`` for a scalar diffusion time ``t``."""
    return jnp.cos(2.0 * jnp.pi * (self.weight * t + self.bias))


class RelativePositionEncoder(eqx.Module):
  """Relative-position + relative-chain one-hot pairwise encoder.

  Ported from ``layers.py:555-613``.
  """

  linear: eqx.nn.Linear
  r_max: int = eqx.field(static=True)
  s_max: int = eqx.field(static=True)

  def __init__(self, token_z: int, r_max: int = 32, s_max: int = 1, *, key: PRNGKeyArray) -> None:
    """Initialize the relative position encoder.

    Args:
        token_z: Output pairwise representation dimension.
        r_max: Maximum clipped residue-index distance (default 32).
        s_max: Maximum clipped chain-id distance (default 1).
        key: PRNG key for the (bias-free) output linear.

    """
    self.r_max = r_max
    self.s_max = s_max
    in_dim = 2 * (r_max + 1) + (s_max + 1)
    self.linear = eqx.nn.Linear(in_dim, token_z, use_bias=False, key=key)

  def __call__(
    self,
    residue_index: jax.Array,  # int, shape (N,)
    chain_id: jax.Array,  # int, shape (N,)
  ) -> Float[Array, "N N dim"]:
    """Compute the pairwise relative-position/chain feature, projected to ``token_z``."""
    b_same_chain = chain_id[:, None] == chain_id[None, :]
    rel_pos = residue_index[:, None] - residue_index[None, :]

    d_residue = jnp.clip(rel_pos + self.r_max, 0, 2 * self.r_max)
    d_residue = jnp.where(b_same_chain, d_residue, jnp.full_like(d_residue, 2 * self.r_max + 1))
    a_rel_pos = jax.nn.one_hot(d_residue, 2 * self.r_max + 2)

    d_chain = jnp.clip(jnp.abs(chain_id[:, None] - chain_id[None, :]), 0, self.s_max)
    a_rel_chain = jax.nn.one_hot(d_chain, self.s_max + 1)

    feats = jnp.concatenate([a_rel_pos, a_rel_chain], axis=-1)
    return jax.vmap(jax.vmap(self.linear))(feats)


class AdaLN(eqx.Module):
  """Adaptive layer normalization, conditioned on a single representation.

  Ported from ``layers.py:105-129``. Operates on full ``(N, D)`` arrays
  (internally ``vmap``s its per-token sublayers over the residue axis).
  """

  a_norm: eqx.nn.LayerNorm
  s_norm: eqx.nn.LayerNorm
  s_scale: eqx.nn.Linear
  s_bias: eqx.nn.Linear

  def __init__(self, dim: int, dim_single_cond: int, *, key: PRNGKeyArray) -> None:
    """Initialize AdaLN.

    Args:
        dim: Input/output dimension of the conditioned representation ``a``.
        dim_single_cond: Dimension of the conditioning representation ``s``.
        key: PRNG key for ``s_scale``/``s_bias``.

    """
    k_scale, k_bias = jax.random.split(key)
    # a_norm: elementwise_affine=False, bias=False in the reference -> no
    # learnable weight *or* bias at all (pure standardization).
    self.a_norm = eqx.nn.LayerNorm(dim, use_weight=False, use_bias=False)
    # s_norm: bias=False only -> learnable weight, no learnable bias.
    self.s_norm = eqx.nn.LayerNorm(dim_single_cond, use_weight=True, use_bias=False)
    self.s_scale = eqx.nn.Linear(dim_single_cond, dim, key=k_scale)
    self.s_bias = eqx.nn.Linear(dim_single_cond, dim, use_bias=False, key=k_bias)

  def __call__(
    self,
    a: Float[Array, "N dim"],
    s: Float[Array, "N dim_cond"],
  ) -> Float[Array, "N dim"]:
    """Apply ``sigmoid(s_scale(s_norm(s))) * a_norm(a) + s_bias(s_norm(s))``."""
    a = jax.vmap(self.a_norm)(a)
    s = jax.vmap(self.s_norm)(s)
    return jax.nn.sigmoid(jax.vmap(self.s_scale)(s)) * a + jax.vmap(self.s_bias)(s)


class Transition(eqx.Module):
  """Two-layer SwiGLU-gated MLP; reuses ``model.diffusion_mpnn.SwiGLU`` verbatim.

  Ported from ``layers.py:454-524`` (chunked ``chunk_size`` inference path
  dropped -- an inference-memory optimization, not a math change: the
  ``chunk_size is None or self.training`` branch computes the identical
  result as the chunked branch, just without tiling). Operates on a
  **single** feature vector (``(D,)``); callers ``vmap`` (once for single
  reps, twice for pairwise reps) -- see :class:`SingleConditioning` /
  :class:`PairwiseConditioning`.
  """

  norm: eqx.nn.LayerNorm
  swiglu: SwiGLU

  def __init__(self, dim: int = 128, hidden: int = 512, out_dim: int | None = None, *, key: PRNGKeyArray) -> None:
    """Initialize the transition block.

    Args:
        dim: Input feature dimension.
        hidden: Hidden (gate/value) dimension.
        out_dim: Output dimension; defaults to ``dim``.
        key: PRNG key for the ``SwiGLU`` sublayer.

    """
    out_dim = dim if out_dim is None else out_dim
    self.norm = eqx.nn.LayerNorm(dim)
    self.swiglu = SwiGLU(dim, hidden, out_dim, key=key)

  def __call__(self, x: jax.Array) -> jax.Array:  # shape (dim,) -> (out_dim,)
    """Apply ``swiglu(norm(x))`` to a single feature vector."""
    return self.swiglu(self.norm(x))


class ConditionedTransitionBlock(eqx.Module):
  """AdaLN-conditioned, sigmoid-gated transition block (3-way SwiGLU-style gate).

  Ported from ``layers.py:132-175``. **Not** a reuse of ``model.diffusion_mpnn
  .SwiGLU`` -- see the module docstring: ``b = swish_gate(a) * a_to_b(a)`` is a
  product of *three* terms (``silu(gate(a))``, ``val(a))`` from one chunked
  projection, times an independent ``a_to_b(a)``), not the two-way
  ``silu(gate) * val`` the aminx ``SwiGLU`` class implements.
  """

  adaln: AdaLN
  swish_gate_proj: eqx.nn.Linear
  a_to_b: eqx.nn.Linear
  b_to_a: eqx.nn.Linear
  output_projection: eqx.nn.Linear
  dim_inner: int = eqx.field(static=True)

  def __init__(
    self,
    dim_single: int,
    dim_single_cond: int,
    expansion_factor: int = 2,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the conditioned transition block.

    Args:
        dim_single: Input/output dimension of ``a``.
        dim_single_cond: Dimension of the conditioning representation ``s``.
        expansion_factor: Hidden expansion factor (default 2).
        key: PRNG key, split across the AdaLN + 4 linear sublayers.

    """
    k_adaln, k_gate, k_a2b, k_b2a, k_out = jax.random.split(key, 5)
    dim_inner = int(dim_single * expansion_factor)
    self.dim_inner = dim_inner

    self.adaln = AdaLN(dim_single, dim_single_cond, key=k_adaln)
    self.swish_gate_proj = eqx.nn.Linear(dim_single, dim_inner * 2, use_bias=False, key=k_gate)
    self.a_to_b = eqx.nn.Linear(dim_single, dim_inner, use_bias=False, key=k_a2b)
    self.b_to_a = eqx.nn.Linear(dim_inner, dim_single, use_bias=False, key=k_b2a)
    self.output_projection = _zero_init_linear(
      eqx.nn.Linear(dim_single_cond, dim_single, key=k_out),
      bias_value=-2.0,
    )

  def __call__(
    self,
    a: Float[Array, "N dim_single"],
    s: Float[Array, "N dim_cond"],
  ) -> Float[Array, "N dim_single"]:
    """Apply the AdaLN-conditioned, sigmoid-gated SwiGLU-style transition."""
    a = self.adaln(a, s)
    gate_and_val = jax.vmap(self.swish_gate_proj)(a)  # (N, 2*dim_inner)
    val, gate = jnp.split(gate_and_val, 2, axis=-1)  # matches boltz_utils.SwiGLU chunk order
    gated = jax.nn.silu(gate) * val
    b = gated * jax.vmap(self.a_to_b)(a)
    out_gate = jax.nn.sigmoid(jax.vmap(self.output_projection)(s))
    return out_gate * jax.vmap(self.b_to_a)(b)


class AttentionPairBias(eqx.Module):
  """Pair-bias multi-head attention.

  Ported from ``layers.py:183-301``. The QK/softmax/AV block is
  numerics-sensitive in the reference (wrapped in
  ``torch.autocast("cuda", enabled=False)`` + explicit ``.float()`` calls);
  ported here as an explicit ``float32`` upcast around that block, cast back
  to the input dtype afterward (``layers.py:285-298``).
  """

  norm_s: eqx.nn.LayerNorm | None
  proj_q: eqx.nn.Linear
  proj_k: eqx.nn.Linear
  proj_v: eqx.nn.Linear
  proj_g: eqx.nn.Linear
  pair_norm: eqx.nn.LayerNorm
  pair_proj: eqx.nn.Linear
  proj_o: eqx.nn.Linear
  num_heads: int = eqx.field(static=True)
  head_dim: int = eqx.field(static=True)
  inf: float = eqx.field(static=True)

  def __init__(
    self,
    c_s: int,
    c_z: int,
    num_heads: int,
    *,
    inf: float = 1e6,
    initial_norm: bool = True,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the pair-bias attention layer.

    Args:
        c_s: Single (sequence) representation dimension.
        c_z: Pairwise representation dimension.
        num_heads: Number of attention heads (``c_s`` must be divisible).
        inf: Additive mask penalty (default 1e6).
        initial_norm: Whether to LayerNorm ``s`` on entry (default True).
        key: PRNG key, split across the 5 (learned) projections.

    """
    if c_s % num_heads != 0:
      msg = f"c_s ({c_s}) must be divisible by num_heads ({num_heads})"
      raise ValueError(msg)

    self.num_heads = num_heads
    self.head_dim = c_s // num_heads
    self.inf = inf

    k_q, k_k, k_v, k_g, k_pair, k_o = jax.random.split(key, 6)
    # LayerNorm has no learnable-init key (weight=ones/bias=zeros, deterministic);
    # `norm_s`'s presence (not a separate stored bool) is the single source of
    # truth for whether `__call__` normalizes `s` on entry.
    self.norm_s = eqx.nn.LayerNorm(c_s) if initial_norm else None
    self.proj_q = eqx.nn.Linear(c_s, c_s, key=k_q)
    self.proj_k = eqx.nn.Linear(c_s, c_s, use_bias=False, key=k_k)
    self.proj_v = eqx.nn.Linear(c_s, c_s, use_bias=False, key=k_v)
    self.proj_g = eqx.nn.Linear(c_s, c_s, use_bias=False, key=k_g)
    self.pair_norm = eqx.nn.LayerNorm(c_z)
    self.pair_proj = eqx.nn.Linear(c_z, num_heads, use_bias=False, key=k_pair)
    self.proj_o = _zero_init_linear(eqx.nn.Linear(c_s, c_s, use_bias=False, key=k_o))

  def __call__(
    self,
    s: Float[Array, "N c_s"],
    z: Float[Array, "N N c_z"],
    mask: ResidueMask,
    mask2d: Bool[Array, "N N"] | None = None,
  ) -> Float[Array, "N c_s"]:
    """Compute pair-bias attention over a single structure's ``s``/``z`` reps."""
    if self.norm_s is not None:
      s = jax.vmap(self.norm_s)(s)

    q = rearrange(jax.vmap(self.proj_q)(s), "n (h d) -> n h d", h=self.num_heads)
    k = rearrange(jax.vmap(self.proj_k)(s), "n (h d) -> n h d", h=self.num_heads)
    v = rearrange(jax.vmap(self.proj_v)(s), "n (h d) -> n h d", h=self.num_heads)
    z_bias = jax.vmap(jax.vmap(self.pair_proj))(jax.vmap(jax.vmap(self.pair_norm))(z))
    z_bias = rearrange(z_bias, "i j h -> h i j")
    g = jax.nn.sigmoid(jax.vmap(self.proj_g)(s))

    # Explicit float32 upcast (reference: `with torch.autocast("cuda", enabled=False)`
    # + `.float()`), cast back to the input dtype afterward.
    out_dtype = v.dtype
    q32, k32, v32 = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    attn = jnp.einsum("ihd,jhd->hij", q32, k32) / jnp.sqrt(jnp.asarray(self.head_dim, jnp.float32))
    attn = attn + z_bias.astype(jnp.float32)
    attn = attn + (1.0 - mask.astype(jnp.float32))[None, None, :] * (-self.inf)
    if mask2d is not None:
      attn = attn + (1.0 - mask2d.astype(jnp.float32))[None, :, :] * (-self.inf)
    attn = jax.nn.softmax(attn, axis=-1)
    o = jnp.einsum("hij,jhd->ihd", attn, v32).astype(out_dtype)

    o = rearrange(o, "i h d -> i (h d)")
    return jax.vmap(self.proj_o)(g * o)


class DiffusionTransformerLayer(eqx.Module):
  """One AdaLN + pair-bias-attention + conditioned-transition block.

  Ported from ``layers.py:304-371``, including the reference's explicit
  ``# NOTE: Added residual connection!`` comment on ``a = a + b`` (kept
  verbatim -- a deliberate departure from the upstream Boltz-1 design that
  this port preserves for parity with the shipped ProteinEBM weights).
  """

  adaln: AdaLN
  pair_bias_attn: AttentionPairBias
  output_projection: eqx.nn.Linear
  transition: ConditionedTransitionBlock

  def __init__(
    self,
    heads: int,
    dim: int = 384,
    dim_single_cond: int | None = None,
    dim_pairwise: int = 128,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize one diffusion transformer layer.

    Args:
        heads: Number of attention heads.
        dim: Single representation dimension (default 384, Boltz-1 default).
        dim_single_cond: Conditioning dimension; defaults to ``dim``.
        dim_pairwise: Pairwise representation dimension (default 128).
        key: PRNG key, split across AdaLN, attention, output proj, transition.

    """
    dim_single_cond = dim if dim_single_cond is None else dim_single_cond
    k_adaln, k_attn, k_out, k_trans = jax.random.split(key, 4)

    self.adaln = AdaLN(dim, dim_single_cond, key=k_adaln)
    self.pair_bias_attn = AttentionPairBias(
      c_s=dim,
      c_z=dim_pairwise,
      num_heads=heads,
      initial_norm=False,
      key=k_attn,
    )
    self.output_projection = _zero_init_linear(
      eqx.nn.Linear(dim_single_cond, dim, key=k_out),
      bias_value=-2.0,
    )
    self.transition = ConditionedTransitionBlock(dim_single=dim, dim_single_cond=dim_single_cond, key=k_trans)

  def __call__(
    self,
    a: Float[Array, "N dim"],
    s: Float[Array, "N dim_cond"],
    z: Float[Array, "N N dim_pairwise"],
    mask: ResidueMask,
    mask2d: Bool[Array, "N N"] | None = None,
  ) -> Float[Array, "N dim"]:
    """Apply one AdaLN + pair-bias-attention + conditioned-transition layer."""
    b = self.adaln(a, s)
    b = self.pair_bias_attn(b, z, mask, mask2d=mask2d)
    b = jax.nn.sigmoid(jax.vmap(self.output_projection)(s)) * b
    a = a + b  # NOTE: Added residual connection! (matches layers.py:369)
    return a + self.transition(a, s)


class DiffusionTransformer(eqx.Module):
  """Stack of ``depth`` :class:`DiffusionTransformerLayer`, AdaLN-conditioned throughout.

  Ported from ``layers.py:373-446``. Module defaults are ProteinEBM-x's
  actual instantiation config (design spec §2: ``token_s=256, token_z=128,
  token_transformer_depth=16, token_transformer_heads=8``), **not** the
  reference class's generic Boltz-1 defaults (``dim=384``). The reference's
  ``activation_checkpointing`` flag is dropped -- it is a no-op in the
  reference (see module docstring).
  """

  layers: tuple[DiffusionTransformerLayer, ...]

  def __init__(
    self,
    depth: int = DEFAULT_TRANSFORMER_DEPTH,
    heads: int = DEFAULT_TRANSFORMER_HEADS,
    dim: int = DEFAULT_TOKEN_S,
    dim_single_cond: int | None = None,
    dim_pairwise: int = DEFAULT_TOKEN_Z,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the diffusion transformer stack.

    Args:
        depth: Number of layers (ProteinEBM-x default 16).
        heads: Number of attention heads per layer (ProteinEBM-x default 8).
        dim: Single representation dimension (ProteinEBM-x default 256).
        dim_single_cond: Conditioning dimension; defaults to ``dim``.
        dim_pairwise: Pairwise representation dimension (ProteinEBM-x default 128).
        key: PRNG key, split ``depth``-ways across layers.

    """
    dim_single_cond = dim if dim_single_cond is None else dim_single_cond
    keys = jax.random.split(key, depth)
    self.layers = tuple(
      DiffusionTransformerLayer(heads, dim, dim_single_cond, dim_pairwise, key=k) for k in keys
    )

  def __call__(
    self,
    a: Float[Array, "N dim"],
    s: Float[Array, "N dim_cond"],
    z: Float[Array, "N N dim_pairwise"],
    mask: ResidueMask,
    mask2d: Bool[Array, "N N"] | None = None,
  ) -> Float[Array, "N dim"]:
    """Run all layers in sequence, threading the conditioned single rep ``a``."""
    for layer in self.layers:
      a = layer(a, s, z, mask, mask2d=mask2d)
    return a


class SingleConditioning(eqx.Module):
  """Fuse an input single embedding + Fourier time embedding into a conditioned single rep.

  Ported from ``layers.py:616-705``. Constructor defaults kept verbatim from
  the reference (generic Boltz-1 values: ``input_dim=384, token_s=384``) for
  byte-faithful mechanical fidelity; pass ProteinEBM-x's actual values
  explicitly at call sites (see :data:`DEFAULT_TOKEN_S`).
  """

  norm_single: eqx.nn.LayerNorm
  single_embed: eqx.nn.Linear
  fourier_embed: FourierEmbedding
  norm_fourier: eqx.nn.LayerNorm
  fourier_to_single: eqx.nn.Linear
  transitions: tuple[Transition, ...]
  eps: float = eqx.field(static=True)

  def __init__(
    self,
    input_dim: int = 384,
    token_s: int = 384,
    dim_fourier: int = 256,
    num_transitions: int = 2,
    transition_expansion_factor: int = 2,
    eps: float = 1e-20,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the single-conditioning fuser.

    Args:
        input_dim: Input single-embedding dimension (default 384).
        token_s: Half-width of the output single representation; output is
          ``2*token_s`` (default 384).
        dim_fourier: Fourier time-embedding dimension (default 256).
        num_transitions: Number of ``Transition`` blocks (default 2).
        transition_expansion_factor: Transition hidden expansion (default 2).
        eps: Unused epsilon, kept for structural fidelity (dead in the
          reference forward too -- ``layers.py:643``, never referenced).
        key: PRNG key, split across embed/Fourier/transition sublayers.

    """
    self.eps = eps
    keys = jax.random.split(key, 3 + num_transitions)
    self.norm_single = eqx.nn.LayerNorm(input_dim)
    self.single_embed = eqx.nn.Linear(input_dim, 2 * token_s, key=keys[0])
    self.fourier_embed = FourierEmbedding(dim_fourier, key=keys[1])
    self.norm_fourier = eqx.nn.LayerNorm(dim_fourier)
    self.fourier_to_single = eqx.nn.Linear(dim_fourier, 2 * token_s, use_bias=False, key=keys[2])
    self.transitions = tuple(
      Transition(
        dim=2 * token_s,
        hidden=transition_expansion_factor * 2 * token_s,
        key=k,
      )
      for k in keys[3:]
    )

  def __call__(
    self,
    s: Float[Array, "N input_dim"],
    times: DiffusionTime | None = None,
    direct_embedding: jax.Array | None = None,  # shape (dim_fourier,)
  ) -> tuple[Float[Array, "N two_token_s"], jax.Array]:  # 2nd: shape (dim_fourier,)
    """Fuse ``s`` with either a Fourier(``times``) or a precomputed ``direct_embedding``.

    Exactly one of ``times``/``direct_embedding`` must be provided (matches
    the reference's mutual-exclusivity check, ``layers.py:685-688``).

    Returns:
        ``(conditioned_single, normed_fourier_embedding)``.

    """
    if times is not None and direct_embedding is not None:
      msg = "Cannot provide both times and direct_embedding"
      raise ValueError(msg)

    s = jax.vmap(self.single_embed)(jax.vmap(self.norm_single)(s))

    if direct_embedding is not None:
      fourier_to_single = direct_embedding
      normed_fourier = direct_embedding
    elif times is not None:
      fourier_embed = self.fourier_embed(times)
      normed_fourier = self.norm_fourier(fourier_embed)
      fourier_to_single = self.fourier_to_single(normed_fourier)
    else:
      msg = "Either times or direct_embedding must be provided"
      raise ValueError(msg)

    s = fourier_to_single[None, :] + s

    for transition in self.transitions:
      s = jax.vmap(transition)(s) + s

    return s, normed_fourier


class PairwiseConditioning(eqx.Module):
  """Fuse a trunk pairwise rep with relative-position features into a conditioned pair rep.

  Ported from ``layers.py:708-761``.
  """

  init_norm: eqx.nn.LayerNorm
  init_proj: eqx.nn.Linear
  transitions: tuple[Transition, ...]

  def __init__(
    self,
    input_dim: int,
    token_z: int,
    dim_token_rel_pos_feats: int,
    num_transitions: int = 2,
    transition_expansion_factor: int = 2,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the pairwise-conditioning fuser.

    Args:
        input_dim: Input pairwise trunk dimension.
        token_z: Output pairwise representation dimension.
        dim_token_rel_pos_feats: Dimension of the relative-position features
          to concatenate (see :class:`RelativePositionEncoder`).
        num_transitions: Number of ``Transition`` blocks (default 2).
        transition_expansion_factor: Transition hidden expansion (default 2).
        key: PRNG key, split across the init projection + transitions.

    """
    keys = jax.random.split(key, 1 + num_transitions)
    concat_dim = input_dim + dim_token_rel_pos_feats
    self.init_norm = eqx.nn.LayerNorm(concat_dim)
    self.init_proj = eqx.nn.Linear(concat_dim, token_z, use_bias=False, key=keys[0])
    self.transitions = tuple(
      Transition(dim=token_z, hidden=transition_expansion_factor * token_z, key=k) for k in keys[1:]
    )

  def __call__(
    self,
    z_trunk: Float[Array, "N N input_dim"],
    token_rel_pos_feats: Float[Array, "N N feat_dim"],
  ) -> Float[Array, "N N token_z"]:
    """Concatenate, project, then run ``Transition`` residual blocks over the pair rep."""
    z = jnp.concatenate([z_trunk, token_rel_pos_feats], axis=-1)
    z = jax.vmap(jax.vmap(self.init_proj))(jax.vmap(jax.vmap(self.init_norm))(z))
    for transition in self.transitions:
      z = jax.vmap(jax.vmap(transition))(z) + z
    return z


# ---------------------------------------------------------------------------
# SO(3) random-rotation augmentation (boltz_utils.py:226-319, 61-116).
# ---------------------------------------------------------------------------


def quaternion_to_matrix(quaternions: Float[Array, "*batch 4"]) -> Float[Array, "*batch 3 3"]:
  """Convert real-part-first quaternions to 3x3 rotation matrices.

  Ported from ``boltz_utils.py:247-276`` (itself vendored from PyTorch3D,
  BSD license).
  """
  r, i, j, k = (quaternions[..., idx] for idx in range(4))
  two_s = 2.0 / jnp.sum(quaternions * quaternions, axis=-1)

  o = jnp.stack(
    [
      1 - two_s * (j * j + k * k),
      two_s * (i * j - k * r),
      two_s * (i * k + j * r),
      two_s * (i * j + k * r),
      1 - two_s * (i * i + k * k),
      two_s * (j * k - i * r),
      two_s * (i * k - j * r),
      two_s * (j * k + i * r),
      1 - two_s * (i * i + j * j),
    ],
    axis=-1,
  )
  return o.reshape((*quaternions.shape[:-1], 3, 3))


def random_quaternions(key: PRNGKeyArray, n: int, dtype: jnp.dtype = jnp.float32) -> Float[Array, "n 4"]:
  """Sample ``n`` random unit quaternions (nonnegative real part).

  Ported from ``boltz_utils.py:279-300``. ``torch.copysign(sqrt(s), o[:,0])``
  becomes an explicit sign multiply (JAX has no ``copysign`` in this form).
  """
  o = jax.random.normal(key, (n, 4), dtype=dtype)
  s = jnp.sum(o * o, axis=1)
  sign = jnp.where(o[:, 0] >= 0, 1.0, -1.0).astype(dtype)
  return o / (sign * jnp.sqrt(s))[:, None]


def random_rotations(key: PRNGKeyArray, n: int, dtype: jnp.dtype = jnp.float32) -> Float[Array, "n 3 3"]:
  """Sample ``n`` random SO(3) rotation matrices. Ported from ``boltz_utils.py:303-319``."""
  return quaternion_to_matrix(random_quaternions(key, n, dtype=dtype))


def center_random_augmentation(
  coords: Coords,
  mask: ResidueMask,
  key: PRNGKeyArray,
  *,
  s_trans: float = 1.0,
  augmentation: bool = True,
  centering: bool = True,
  rotate: bool = True,
) -> Coords:
  """Center and randomly rotate/translate a single structure's coordinates.

  Ported from ``boltz_utils.py:61-116``. ``rotate`` is an **explicit**,
  always-visible keyword (design spec §10 MINOR finding /
  EPIC risk register): for deterministic inference (decoy ranking, ΔE
  biasing) call with ``rotate=False`` to disable the SO(3) augmentation
  while still centering (and, if ``augmentation=True``, translating).

  ``rotate``/``augmentation``/``centering`` are plain Python ``bool``s
  (static control flow, not traced values) -- callers that ``jax.jit`` this
  function must mark them ``static_argnames``.

  PRNG discipline: ``key`` is split into a rotation sub-key and a
  translation sub-key exactly once; the rotation sub-key is simply unused
  when ``rotate=False`` (no key reuse -- it is drawn either way so the
  translation draw is identical regardless of ``rotate``, keeping the two
  code paths reproducible against the same input ``key``).

  Args:
      coords: Per-residue coordinates, ``(N, 3)``.
      mask: Residue validity mask, ``(N,)``.
      key: PRNG key for the rotation/translation draws.
      s_trans: Translation noise scale (default 1.0).
      augmentation: Whether to apply rotation + translation (default True).
      centering: Whether to mask-weighted-mean-center first (default True).
      rotate: Whether the augmentation includes a random SO(3) rotation
        (default True; set False for deterministic inference).

  Returns:
      Augmented coordinates, ``(N, 3)``.

  """
  if centering:
    mask_f = mask.astype(coords.dtype)
    denom = jnp.sum(mask_f)
    mean = jnp.sum(coords * mask_f[:, None], axis=0, keepdims=True) / denom
    coords = coords - mean

  if augmentation:
    k_rot, k_trans = jax.random.split(key)
    if rotate:
      rotation = random_rotations(k_rot, 1, dtype=coords.dtype)[0]
      coords = jnp.einsum("md,ds->ms", coords, rotation)
    translation = jax.random.normal(k_trans, (1, 3), dtype=coords.dtype) * s_trans
    coords = coords + translation

  return coords


def batched_center_random_augmentation(
  coords: Float[Array, "B N 3"],
  mask: Bool[Array, "B N"],
  key: PRNGKeyArray,
  *,
  s_trans: float = 1.0,
  augmentation: bool = True,
  centering: bool = True,
  rotate: bool = True,
) -> Float[Array, "B N 3"]:
  """Vmapped :func:`center_random_augmentation` over a batch of structures.

  Per-structure keys are derived via ``jax.random.fold_in(key, idx)`` --
  the same per-element PRNG pattern already used for decoy/mutant keys at
  ``host/kernel_dispatch.py:244`` (design spec Fork 10 / EPIC decision #10),
  so a batch of ``B`` structures gets ``B`` independent augmentations from a
  single base ``key`` rather than requiring a pre-split key array.
  """
  n = coords.shape[0]

  def _augment_one(idx: Int[Array, ""], one_coords: Coords, one_mask: ResidueMask) -> Coords:
    sub_key = jax.random.fold_in(key, idx)
    return center_random_augmentation(
      one_coords,
      one_mask,
      sub_key,
      s_trans=s_trans,
      augmentation=augmentation,
      centering=centering,
      rotate=rotate,
    )

  return jax.vmap(_augment_one)(jnp.arange(n), coords, mask)


__all__ = [
  "DEFAULT_TOKEN_S",
  "DEFAULT_TOKEN_Z",
  "DEFAULT_TRANSFORMER_DEPTH",
  "DEFAULT_TRANSFORMER_HEADS",
  "AdaLN",
  "AttentionPairBias",
  "ConditionedTransitionBlock",
  "DiffusionTransformer",
  "DiffusionTransformerLayer",
  "FourierEmbedding",
  "PairwiseConditioning",
  "RelativePositionEncoder",
  "SingleConditioning",
  "Transition",
  "batched_center_random_augmentation",
  "center_random_augmentation",
  "quaternion_to_matrix",
  "random_quaternions",
  "random_rotations",
]
