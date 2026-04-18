# FunWake: Full-Optimizer Discovery Mode

You are designing a **complete wind farm layout optimizer** that
maximizes annual energy production (AEP) subject to boundary and
spacing constraints. You write an `optimize()` function that receives
the physics simulation and problem parameters, and returns turbine
positions.

## Your task

Write an `optimize` function:

```python
def optimize(sim, n_target, boundary, min_spacing, wd, ws, weights):
    """Optimize turbine layout for maximum AEP.

    Args:
        sim: pixwake simulation callable — sim(x, y, ws_amb, wd_amb, ti_amb)
        n_target: number of turbines to place
        boundary: (V, 2) array of polygon vertices
        min_spacing: minimum distance between any pair of turbines (meters)
        wd: wind directions (degrees)
        ws: wind speeds (m/s)
        weights: sector frequency weights

    Returns:
        x, y: optimized turbine positions as JAX arrays
    """
```

## Important rules

- Do NOT use `import os`, `open()`, or any file I/O inside your
  optimizer. All inputs come through function arguments.
- Do NOT read problem.json directly. The harness loads it for you.
- Use `jax`, `jax.numpy`, `scipy.optimize`, and `numpy` freely.
- Use `from pixwake.optim.boundary import polygon_sdf` for boundary
  distance and `from pixwake.optim.sgd import boundary_penalty,
  spacing_penalty` for differentiable penalties.
- You may use `jax.grad`, `jax.jacobian`, `jax.vmap`, `jax.jit`.
- Each evaluation has a **180-second timeout**. Budget your compute.
- JAX JIT compilation takes ~20-30s on the first function call.
  Design for ~150s of actual optimization after JIT warmup.
- A single SLSQP run with 50 turbines takes ~30-60s after JIT.
  Multi-start with more than 2-3 starts will timeout.
- Prefer 1-2 high-quality starts over many cheap starts.

## Available tools

```bash
# Score on training farm (returns AEP in GWh, feasibility, time)
pixi run python tools/run_optimizer.py <script> --timeout 180

# Run unit tests (signature check + stressed polygon)
pixi run python tools/run_tests.py <script> --quick

# Test generalization (returns PASS/FAIL, not AEP)
pixi run python tools/test_generalization.py <script>

# Check progress
pixi run python tools/get_status.py --log results_agent_claude_fullopt/attempt_log.json
```

## Results directory

Write your optimizer scripts to `results_agent_claude_fullopt/iter_NNN.py`.
Start from iter_001.py and increment.

## Baseline and seed

- **Baseline**: 5540.7 GWh (500-multi-start SGD)
- **Seed optimizer**: `results/seed_optimizer.py` — a simple SGD wrapper.
  Read it to understand the interface, then improve on it.
- **Known strong approach**: SLSQP with JAX Jacobians is the community
  standard. But we want you to go beyond standard approaches.

## Strategy hints

- Start by understanding the seed optimizer and scoring it
- Try gradient-based methods (Adam, L-BFGS, SLSQP) with good initialization
- Consider multi-start or population-based approaches
- Constraint handling is critical: penalty methods, augmented Lagrangian,
  or SLSQP's built-in constraint support
- Initialization matters: hexagonal grids, K-means, wind-direction-aware
  placement
- Always run tests before scoring — infeasible layouts waste time
- Read your previous results via get_status to track progress

## Workflow

1. Read `results/seed_optimizer.py` to understand the interface
2. Read `playground/skeleton.py` to understand the physics
3. Write an optimizer, test it, score it
4. Iterate: each attempt should try something meaningfully different
5. Always check generalization — an optimizer that only works on the
   training farm is useless
