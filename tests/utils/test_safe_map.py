
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.safe_map import safe_map

def test_safe_map_vmap_dispatch():
    """Test that safe_map works correctly when batch_size >= input_size (vmap path)."""
    xs = jnp.arange(10)
    
    def f(x):
        return x * 2
        
    # batch_size > input_size
    res = safe_map(f, xs, batch_size=20)
    assert jnp.allclose(res, xs * 2)
    
    # batch_size == input_size
    res = safe_map(f, xs, batch_size=10)
    assert jnp.allclose(res, xs * 2)
    
    # batch_size is None
    res = safe_map(f, xs, batch_size=None)
    assert jnp.allclose(res, xs * 2)

def test_safe_map_lax_map_dispatch():
    """Test that safe_map works correctly when batch_size < input_size (lax.map path)."""
    xs = jnp.arange(10)
    
    def f(x):
        return x * 2
        
    # batch_size < input_size
    res = safe_map(f, xs, batch_size=5)
    assert jnp.allclose(res, xs * 2)
    
    # batch_size = 1
    res = safe_map(f, xs, batch_size=1)
    assert jnp.allclose(res, xs * 2)

def test_safe_map_pytrees():
    """Test that safe_map handles PyTrees correctly."""
    xs = {
        "a": jnp.arange(10),
        "b": jnp.arange(10) + 10
    }
    
    def f(x):
        return x["a"] + x["b"]
        
    expected = xs["a"] + xs["b"]
    
    # vmap path
    res_vmap = safe_map(f, xs, batch_size=20)
    assert jnp.allclose(res_vmap, expected)
    
    # lax.map path
    res_map = safe_map(f, xs, batch_size=2)
    assert jnp.allclose(res_map, expected)

def test_safe_map_complex_output():
    """Test that safe_map handles complex output structures."""
    xs = jnp.arange(6)
    
    def f(x):
        return {"sq": x**2, "cube": x**3}
        
    # vmap path
    res_vmap = safe_map(f, xs, batch_size=10)
    assert jnp.allclose(res_vmap["sq"], xs**2)
    assert jnp.allclose(res_vmap["cube"], xs**3)
    
    # lax.map path
    res_map = safe_map(f, xs, batch_size=2)
    assert jnp.allclose(res_map["sq"], xs**2)
    assert jnp.allclose(res_map["cube"], xs**3)

def test_safe_map_jit_compatibility():
    """Test that safe_map works inside JIT."""
    
    @jax.jit
    def run_safe_map(x, b_size):
        # Note: batch_size must be static for lax.map usually, but safe_map takes int.
        # If passed as a tracer, safe_map's 'if' condition might fail or require static_argnums.
        # safe_map expects batch_size to be an int or None, not a tracer.
        # So we test with concrete batch_size inside JIT if possible, or static arg.
        return safe_map(lambda v: v + 1, x, batch_size=b_size)
    
    xs = jnp.arange(10)
    
    # We need to use static_argnames or static_argnums for batch_size if we were wrapping safe_map directly,
    # but here we are calling it.
    # However, 'batch_size' argument to safe_map is used in a python conditional:
    # `if batch_size is None or num_elements <= batch_size:`
    # This means batch_size MUST be static.
    
    # Let's verify it works when batch_size is passed as a static value to the jitted function
    # or hardcoded.
    
    @jax.jit
    def run_vmap_path(x):
        return safe_map(lambda v: v + 1, x, batch_size=20)
        
    @jax.jit
    def run_map_path(x):
        return safe_map(lambda v: v + 1, x, batch_size=2)

    assert jnp.allclose(run_vmap_path(xs), xs + 1)
    assert jnp.allclose(run_map_path(xs), xs + 1)
