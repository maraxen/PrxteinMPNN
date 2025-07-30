"""GeLU activation function.

prxteinmpnn.utils.gelu
"""

from functools import partial

import jax

GeLU = partial(jax.nn.gelu, approximate=False)
