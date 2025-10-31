import jax.numpy as jnp
from prxteinmpnn.utils.autoregression import get_decoding_step_map, make_autoregressive_mask

def test_tied_autoregressive_mask():
    tie_group_map = jnp.array([0, 1, 0, 2])
    group_decoding_order = jnp.array([1, 0, 2])  # group 1, then 0, then 2
    decoding_step_map = get_decoding_step_map(tie_group_map, group_decoding_order)
    assert (decoding_step_map == jnp.array([1, 0, 1, 2])).all()
    mask = make_autoregressive_mask(decoding_step_map)
    # mask[0, 2] is False (same step 1)
    assert mask[0, 2] == False
    # mask[0, 0] is True (self)
    assert mask[0, 0] == True
    # mask[0, 1] is True (step 1 > step 0)
    assert mask[0, 1] == True
    # mask[1, 0] is False (step 0 < step 1)
    assert mask[1, 0] == False
    # mask[3, 0] is True (step 2 > step 1)
    assert mask[3, 0] == True
