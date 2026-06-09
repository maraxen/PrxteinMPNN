"""Differentiable tree-reweighted message passing on a dense Potts MRF."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from aminx.potts._spectral import laplacian_from_adjacency
from aminx.potts._trw_flashmd import prod_pow_messages_axis0_tiled
from aminx.potts._trw_spec import PottsTRWRunSpec


def rho_from_l_pinv(
    W: Float[Array, "n n"],  # noqa: N803
    *,
    rho_floor: float = 1e-6,
    rho_cap: float = 1.0,
) -> Array:
    """Dense UST leverage rho_ij = W_ij * R_eff(i,j) via ``pinv(L)`` (legacy baseline)."""
    W = 0.5 * (W + W.T)  # noqa: N806
    L = laplacian_from_adjacency(W)  # noqa: N806
    l_pinv = jnp.linalg.pinv(L)
    diag_l = jnp.diag(l_pinv)
    r_eff = diag_l[:, None] + diag_l[None, :] - 2.0 * l_pinv
    raw = W * r_eff
    clipped = jnp.clip(raw, rho_floor, rho_cap)
    return jnp.where(W > 0, clipped, 0.0)


def spectral_rho_dense(
    W: Float[Array, "n n"],  # noqa: N803
    *,
    rho_floor: float = 1e-6,
    rho_cap: float = 1.0,
) -> Array:
    """Alias of :func:`rho_from_l_pinv` (kept for backwards-compatible imports)."""
    return rho_from_l_pinv(W, rho_floor=rho_floor, rho_cap=rho_cap)


def rho_uniform(
    W: Float[Array, "n n"],  # noqa: N803
    *,
    rho_floor: float,
    rho_cap: float,
    value: float,
) -> Array:
    """Constant ``rho`` on positive edges (control / fast path)."""
    W = 0.5 * (W + W.T)  # noqa: N806
    v = jnp.asarray(value, dtype=W.dtype)
    clipped = jnp.clip(v, rho_floor, rho_cap)
    return jnp.where(W > 0, clipped, 0.0)


def _cg_fixed_iters(
    matvec: Callable[[Array], Array],
    b: Float[Array, ...],
    *,
    iters: int,
) -> Array:
    """Fixed-iteration CG for SPD ``A`` with ``matvec(v) = A @ v`` (no early stop)."""
    x = jnp.zeros_like(b)
    r = b - matvec(x)
    p = r

    def one_step(_: int, state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        x, r, p = state
        Ap = matvec(p)  # noqa: N806
        rs_old = jnp.sum(r * r)
        denom = jnp.sum(p * Ap) + jnp.asarray(1e-20, dtype=b.dtype)
        alpha = rs_old / denom
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rs_new = jnp.sum(r_new * r_new)
        beta = rs_new / (rs_old + jnp.asarray(1e-20, dtype=b.dtype))
        p_new = r_new + beta * p
        return x_new, r_new, p_new

    return jax.lax.fori_loop(0, iters, one_step, (x, r, p))[0]


def rho_from_edge_cg(
    W: Float[Array, "n n"],  # noqa: N803
    *,
    rho_floor: float,
    rho_cap: float,
    cg_iters: int,
) -> Array:
    """Approximate effective resistances via grounded Laplacian + fixed-step CG (no ``pinv``).

    Ground node ``0``: solve ``(L_red + eps I) x = b_red`` with ``L_red = L[1:,1:]``,
    embed ``x`` with a zero at node 0, and use ``b^T x`` as a proxy for ``b^T L^+ b``.
    """
    n = int(W.shape[0])
    Wsym = 0.5 * (W + W.T)  # noqa: N806
    if n <= 1:
        return jnp.zeros_like(Wsym)

    L = laplacian_from_adjacency(Wsym)  # noqa: N806
    L_red = L[1:, 1:]  # noqa: N806
    n_red = n - 1
    eps = jnp.maximum(
        jnp.asarray(1e-8, dtype=L.dtype),
        jnp.trace(L_red) / jnp.asarray(max(n_red, 1), dtype=L.dtype) * jnp.asarray(1e-6, dtype=L.dtype),
    )

    def matvec(v: Array) -> Array:
        return L_red @ v + eps * v

    def resistance_at(i: Array, j: Array) -> Array:
        def nonzero() -> Array:
            b_full = jnp.zeros((n,), dtype=Wsym.dtype).at[i].set(1.0).at[j].add(-1.0)
            b_red = b_full[1:]
            x_red = _cg_fixed_iters(matvec, b_red, iters=cg_iters)
            x_full = jnp.concatenate([jnp.zeros((1,), dtype=Wsym.dtype), x_red], axis=0)
            return jnp.sum(b_full * x_full)

        return jax.lax.cond(
            (i != j) & (Wsym[i, j] > 0),
            nonzero,
            lambda: jnp.asarray(0.0, dtype=Wsym.dtype),
        )

    ii = jnp.arange(n, dtype=jnp.int32)
    R = jax.vmap(lambda i: jax.vmap(lambda j: resistance_at(i, j))(ii))(ii)  # noqa: N806
    R = 0.5 * (R + jnp.transpose(R))  # noqa: N806
    raw = Wsym * R
    clipped = jnp.clip(raw, rho_floor, rho_cap)
    return jnp.where(Wsym > 0, clipped, 0.0)


class DifferentiableTRW(eqx.Module):
    """Fixed-point TRW-style updates (vectorized); backend for Potts models."""

    q: int = eqx.field(static=True)
    trw_iters: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    rho_floor: float = eqx.field(static=True)
    spec: PottsTRWRunSpec = eqx.field(static=True)

    def __init__(
        self,
        q: int,
        trw_iters: int = 15,
        damping: float = 0.5,
        rho_floor: float = 1e-6,
        *,
        spec: PottsTRWRunSpec | None = None,
    ) -> None:
        """Initialize DifferentiableTRW.

        Args:
            q: Alphabet size (number of states).
            trw_iters: Number of belief propagation iterations.
            damping: Damping factor for message updates (0.0 = no damping, 1.0 = full damping).
            rho_floor: Minimum edge weight value.
            spec: PottsTRWRunSpec for backend routing (default: dense_pinv + fori).
        """
        self.q = q
        self.trw_iters = trw_iters
        self.damping = damping
        self.rho_floor = rho_floor
        self.spec = PottsTRWRunSpec.default_dense() if spec is None else spec

    def compute_spectral_rho(self, W: Float[Array, "n n"]) -> Array:  # noqa: N803
        """Compute edge weights via configured rho backend."""
        rb = self.spec.rho_backend
        if rb == "dense_pinv":
            return rho_from_l_pinv(W, rho_floor=self.rho_floor, rho_cap=1.0)
        if rb == "uniform":
            return rho_uniform(
                W,
                rho_floor=self.rho_floor,
                rho_cap=1.0,
                value=self.spec.uniform_rho_value,
            )
        if rb == "edge_lanczos":
            return rho_from_edge_cg(
                W,
                rho_floor=self.rho_floor,
                rho_cap=1.0,
                cg_iters=max(1, int(self.spec.lanczos_rank)),
            )
        msg = f"unknown rho_backend: {rb!r}"
        raise ValueError(msg)

    def __call__(
        self,
        W: Float[Array, "n n"],  # noqa: N803
        J: Float[Array, "n n q q"],  # noqa: N803
        h: Float[Array, "n q"],
    ) -> tuple[Array, Array]:
        """Run TRW belief propagation on Potts MRF.

        Args:
            W: Graph adjacency (binary, symmetric), shape (n, n).
            J: Pairwise potentials, shape (n, n, q, q).
            h: Node potentials, shape (n, q).

        Returns:
            Tuple of:
            - marginals: Node marginal probabilities, shape (n, q).
            - rho: Edge weights from spectral computation, shape (n, n).

        Raises:
            ValueError: If training=True and trw_loop="fori" (OOM risk).
        """
        n = W.shape[0]
        rho = self.compute_spectral_rho(W)
        m = jnp.ones((n, n, self.q), dtype=W.dtype) / float(self.q)

        safe_rho = jnp.maximum(rho, self.rho_floor)
        energy = (1.0 / safe_rho[..., None, None]) * J + h[:, None, :, None]
        exp_energy = jnp.exp(jnp.clip(energy, -30.0, 30.0))

        spec = self.spec
        if spec.message_backend == "tiled_scan":
            ts = int(spec.tile_size)
            if n % ts != 0:
                msg = f"tiled_scan requires n % tile_size == 0; got n={n}, tile_size={ts}"
                raise ValueError(msg)

        def total_in_from_m(m_current: Array) -> Array:
            if spec.message_backend == "dense":
                m_pow = jnp.power(m_current, rho[..., None])
                return jnp.prod(m_pow, axis=0)
            if spec.message_backend == "tiled_scan":
                return prod_pow_messages_axis0_tiled(
                    m_current,
                    rho,
                    tile_size=int(spec.tile_size),
                    rematerialize=bool(spec.checkpoint_trw_step),
                )
            msg = f"unknown message_backend: {spec.message_backend!r}"
            raise ValueError(msg)

        def trw_body(m_current: Array) -> Array:
            total_in = total_in_from_m(m_current)
            m_j_to_i = jnp.transpose(m_current, (1, 0, 2))
            ratio = total_in[:, None, :] / (m_j_to_i + 1e-12)
            msg_val = jnp.sum(exp_energy * ratio[..., None], axis=2)
            msg_val = jnp.where(W[..., None] > 0, msg_val, 1.0 / float(self.q))
            msg_sum = jnp.sum(msg_val, axis=-1, keepdims=True)
            new_m = jnp.where(msg_sum > 0, msg_val / msg_sum, 1.0 / float(self.q))
            return self.damping * m_current + (1.0 - self.damping) * new_m

        step = trw_body
        if spec.checkpoint_trw_step:
            step = jax.checkpoint(trw_body)

        if spec.trw_loop == "scan":
            m_final, _ = jax.lax.scan(lambda carry, _: (step(carry), None), m, None, length=self.trw_iters)
        else:
            m_final = jax.lax.fori_loop(0, self.trw_iters, lambda _i, m_cur: step(m_cur), m)

        m_pow_final = jnp.power(m_final, rho[..., None])
        if spec.message_backend == "dense":
            total_in_final = jnp.prod(m_pow_final, axis=0)
        else:
            total_in_final = prod_pow_messages_axis0_tiled(
                m_final,
                rho,
                tile_size=int(spec.tile_size),
                rematerialize=bool(spec.checkpoint_trw_step),
            )
        marginals = jnp.exp(jnp.clip(h, -30.0, 30.0)) * total_in_final
        marginals = marginals / (jnp.sum(marginals, axis=-1, keepdims=True) + 1e-12)
        return marginals, rho
