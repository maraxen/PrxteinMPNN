"""Diffusion noise schedules and utilities."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp


class NoiseSchedule(eqx.Module):
    """Diffsuion noise schedule (continuous time or discrete steps).
    
    Implements a cosine schedule by default, which is generally preferred
    for image/protein generation tasks over linear schedules.
    """
    
    beta_start: float = eqx.field(static=True)
    beta_end: float = eqx.field(static=True)
    num_steps: int = eqx.field(static=True)
    schedule_type: Literal["cosine", "linear"] = eqx.field(static=True)
    
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: Literal["cosine", "linear"] = "cosine",
    ):
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        if schedule_type == "linear":
            betas = jnp.linspace(beta_start, beta_end, num_steps)
        else:
            # Cosine schedule as proposed by Nichol & Dhariwal 2021
            # alpha_bar(t) = f(t) / f(0)
            # f(t) = cos^2((t/T + s) / (1+s) * pi/2)
            s = 0.008
            steps = jnp.arange(num_steps + 1, dtype=jnp.float32)
            f_t = jnp.cos(((steps / num_steps + s) / (1 + s)) * (jnp.pi / 2)) ** 2
            alphas_cumprod = f_t / f_t[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = jnp.clip(betas, 0.0001, 0.9999)
            
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def sample_forward(
        self,
        x_0: jax.Array,
        t: jax.Array,
        noise: jax.Array | None = None,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample q(x_t | x_0).
        
        Args:
            x_0: Input data [..., D]
            t: Timestep indices [...]
            noise: Optional noise tensor. If None, sampled using key.
            key: PRNG key (required if noise is None).
            
        Returns:
            Tuple of (x_t, noise)
        """
        if noise is None:
            if key is None:
                raise ValueError("Must provide either noise or key")
            noise = jax.random.normal(key, x_0.shape)
            
        # Extract coefficients for t
        # Handle broadcasting: t might be [B], x_0 might be [B, N, D]
        
        # Use dynamic_slice for safety with vmap
        start_indices = (t,)
        slice_sizes = (1,)
        
        sqrt_alpha = jax.lax.dynamic_slice(self.sqrt_alphas_cumprod, start_indices, slice_sizes)
        sqrt_one_minus_alpha = jax.lax.dynamic_slice(self.sqrt_one_minus_alphas_cumprod, start_indices, slice_sizes)
        
        # Sliced result is (1,), reshape to scalar
        sqrt_alpha = sqrt_alpha.reshape(())
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(())
        
        # Expand dims to match x_0 rank if not inside vmap (t is batched)
        if t.ndim > 0:
             shape_diff = x_0.ndim - t.ndim
             if shape_diff > 0:
                 new_shape = t.shape + (1,) * shape_diff
                 sqrt_alpha = sqrt_alpha.reshape(new_shape)
                 sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(new_shape)
            
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise
