"""Frozen combine-strategy ordering matches legacy strategy_map."""

from collections import OrderedDict

from prxteinmpnn.registry import _COMBINE_INDEX, combine_strategy_to_index


def test_combine_index_order_and_values() -> None:
  assert list(_COMBINE_INDEX.items()) == [
    ("arithmetic_mean", 0),
    ("geometric_mean", 1),
    ("product", 2),
  ]
  assert isinstance(_COMBINE_INDEX, OrderedDict)
  assert combine_strategy_to_index("arithmetic_mean") == 0
  assert combine_strategy_to_index("geometric_mean") == 1
  assert combine_strategy_to_index("product") == 2
