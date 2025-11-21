import chex
import jax.numpy as jnp

from prxteinmpnn.utils.autoregression import (
    get_decoding_step_map,
    make_autoregressive_mask,
)


class TestAutoregressionMask(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_tied_autoregressive_mask(self):
        """Test the autoregressive mask with tied positions."""
        tie_group_map = jnp.array([0, 1, 0, 2])
        group_decoding_order = jnp.array([1, 0, 2])  # group 1, then 0, then 2

        get_decoding_step_map_fn = self.variant(get_decoding_step_map)
        decoding_step_map = get_decoding_step_map_fn(
            tie_group_map, group_decoding_order,
        )
        chex.assert_trees_all_equal(decoding_step_map, jnp.array([1, 0, 1, 2]))

        make_autoregressive_mask_fn = self.variant(make_autoregressive_mask)
        mask = make_autoregressive_mask_fn(decoding_step_map)

        expected_mask = jnp.array([
            [True, True, True, False],
            [False, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ], dtype=jnp.bool_)
        chex.assert_trees_all_equal(mask, expected_mask)
        chex.assert_shape(mask, (4, 4))
        chex.assert_tree_all_finite(mask)
