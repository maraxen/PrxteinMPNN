"""JAX sharding utilities for distributed training and inference."""

import logging
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

logger = logging.getLogger(__name__)


def create_mesh(num_devices: int | None = None) -> Mesh:
  """Create a JAX Mesh with a 'data' axis for data parallelism.

  Args:
      num_devices: Optional number of devices to use. If None, uses all available devices.

  Returns:
      A jax.sharding.Mesh object.

  """
  devices = jax.devices()
  total_devices = len(devices)

  if num_devices is not None:
    if num_devices > total_devices:
      logger.warning(
        "Requested %d devices but only %d are available. Using all available devices.",
        num_devices,
        total_devices,
      )
      num_devices = total_devices
    devices = devices[:num_devices]
  else:
    num_devices = total_devices

  if num_devices == 0:
    # Fallback for no devices (e.g. CPU only environment might return 1 or 0 depending on config?)
    # Usually jax.devices() returns at least 1 CPU if no accelerator.
    logger.warning("No devices found. Creating mesh with 1 dummy device if possible or failing.")
    # In standard JAX, there's always at least one device (CPU).

  # For data parallelism, we typically want a 1D mesh where the 'data' axis maps to all devices.
  return Mesh(devices, axis_names=("data",))


def get_batch_sharding(mesh: Mesh, dimensions: int = 1) -> NamedSharding:
  """Return a NamedSharding where the first axis (axis 0) is sharded along 'data'.

  Other axes are replicated (None).

  Args:
      mesh: The JAX Mesh to use.
      dimensions: The number of dimensions of the array to be sharded.

  Returns:
      A jax.sharding.NamedSharding object.

  """
  # Axis 0 is 'data', others are None
  spec = ("data",) + (None,) * (dimensions - 1)
  return NamedSharding(mesh, PartitionSpec(*spec))


def get_replicated_sharding(mesh: Mesh, dimensions: int = 1) -> NamedSharding:
  """Return a NamedSharding where all axes are replicated (None).

  Args:
      mesh: The JAX Mesh to use.
      dimensions: The number of dimensions of the array.

  Returns:
      A jax.sharding.NamedSharding object.

  """
  spec = (None,) * dimensions
  return NamedSharding(mesh, PartitionSpec(*spec))


T = TypeVar("T")


def shard_pytree(pytree: T, mesh: Mesh) -> T:
  """Shard the leaves of a pytree along the 'data' axis.

  Args:
      pytree: The pytree to shard.
      mesh: The JAX Mesh to use.

  Returns:
      The sharded pytree.

  """
  sharding = get_batch_sharding(mesh)
  return jax.tree.map(
    lambda x: (jax.device_put(x, sharding) if isinstance(x, (jax.Array, jnp.ndarray)) else x),
    pytree,
  )
