# FunWake Schedule Designer

You are designing the **learning rate and penalty schedule** for a wind
farm layout optimizer. A fixed skeleton handles initialization, gradient
computation, and Adam updates. You control ONLY how four parameters
change over the course of optimization.

## Your task

Write a `schedule_fn` that returns (lr, alpha, beta1, beta2) at each step:

```python
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Control the optimizer schedule.

    Args:
        step: current iteration (0 to total_steps-1)
        total_steps: total iterations (8000)
        lr0: initial learning rate (50.0)
        alpha0: initial penalty weight (from gradient magnitude)

    Returns:
        lr: learning rate for this step
        alpha: constraint penalty multiplier (higher = stricter constraints)
        beta1: Adam first moment decay (0 to 1)
        beta2: Adam second moment decay (0 to 1)
    """
    ...
    return lr, alpha, beta1, beta2
```

## What the skeleton does (you CANNOT change this)
- Wind-direction-aware grid initialization inside the polygon
- Computes gradients of AEP objective and constraint penalties (boundary + spacing)
- Combines gradients: `grad = grad_obj + alpha * grad_constraint`
- Adam update with your beta1, beta2 at your learning rate
- Runs for 8000 steps total

## What you control
- **lr**: learning rate — how big each step is
- **alpha**: penalty weight — how much to prioritize feasibility vs AEP
- **beta1**: Adam first moment decay — momentum (0.9 = high momentum, 0.1 = low)
- **beta2**: Adam second moment decay — adaptive scaling (0.999 = standard)

## Key insight
The baseline solver couples alpha to 1/lr: as lr decays, alpha increases.
This ensures feasibility in late iterations. Can you do better?

## Rules
- Write schedule files to `runs/schedule_designer/iter_NNN.py`
- NEVER overwrite an existing `iter_NNN.py`. Always increment N.
- Each file must define ONLY `schedule_fn(step, total_steps, lr0, alpha0)`
- Do NOT write `optimize()` — it will be rejected
- Do NOT import topfarm_sgd_solve — you are replacing it
- Score: `python tools/run_optimizer.py <script> --schedule-only`
- Test: `python tools/run_tests.py <script> --quick`
- Generalize: `python tools/test_generalization.py <script> --schedule-only`
- Status: `python tools/get_status.py --log runs/schedule_designer/attempt_log.json`
- Baseline: 5540.7 GWh
- Timeout: 180s per run


## Ideas to try
- Cosine annealing of lr with warm restarts
- Cyclic alpha (high → low → high) instead of monotonic
- Phase transitions: different beta1/beta2 in early vs late iterations
- Exponential vs polynomial lr decay
- Alpha proportional to gradient magnitude (adaptive)
- Standard Adam (beta1=0.9, beta2=0.999) vs TopFarm (0.1, 0.2)

## Workflow
1. Read `runs/schedule_designer/agent_memory.md` for status
2. Read `results/seed_schedule.py` to see the starting schedule
3. Read `playground/skeleton.py` to understand the fixed optimizer
4. Write a new schedule, score it, iterate
