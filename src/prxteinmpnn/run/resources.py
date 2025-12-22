"""Resource allocation utilities for PrxteinMPNN."""

import os
from typing import Literal

import psutil


def compute_resource_allocation(
  strategy: Literal["auto", "full"],
  ram_budget_mb: int | None,
  max_workers: int | None,
  context: Literal["training", "inference"] = "inference",
) -> tuple[int, int]:
  """Compute effective RAM budget and max workers.

  Returns:
      Tuple of (ram_budget_mb, max_workers)

  """
  total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
  total_cpus = os.cpu_count() or 4

  if strategy == "full":
    # Full strategy: use maximum resources
    effective_ram = total_ram_mb
    effective_workers = total_cpus
  else:
    # Auto strategy: context-dependent defaults
    if context == "training":
      default_ram_pct = 0.90  # 90% for training
      default_worker_pct = 1.0  # All cores for training
    else:
      default_ram_pct = 0.50  # 50% for inference
      default_worker_pct = 0.5  # Half cores for inference

    effective_ram = int(total_ram_mb * default_ram_pct)
    effective_workers = max(1, int(total_cpus * default_worker_pct))

  # Manually specified values take precedence
  if ram_budget_mb is not None:
    effective_ram = ram_budget_mb
  if max_workers is not None:
    effective_workers = max_workers

  return effective_ram, effective_workers
