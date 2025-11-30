# Future JAX MD Simulation Architecture

## User Request (2025-11-30)

To achieve fine-grained control over simulation progress and data collection without memory issues or compilation overhead, the following architecture is proposed for future implementation:

### Nested Loop/Scan Structure

1. **Outer Loop (Python/Host Control)**:
    - A Python-side loop (or `jax.while_loop` with host callback) that manages the overall simulation timeline.
    - Responsible for "slotting" results into a host-side PyTree/Array to avoid OOM on GPU.

2. **Inner Scan (JAX Compiled)**:
    - A `jax.lax.scan` (or `fori_loop`) that executes a chunk of steps (e.g., `report_interval`).
    - This chunk is JIT-compiled.
    - Returns the state at the end of the chunk and the trajectory (snapshots) for that chunk.

3. **Data Transfer**:
    - As each inner scan completes, results are transferred/device_put to the host structure.
    - This allows running indefinitely long simulations without accumulating the entire trajectory in GPU memory.

### Goal

- Enable "infinite" running simulations.
- Allow real-time monitoring/plotting.
- Prevent recompilation for different durations (just run the chunk more times).
