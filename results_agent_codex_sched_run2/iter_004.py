"""SGDR warm-restart schedule with coupled feasibility pressure.

HYPOTHESIS: Decreasing warm-restart peaks can revisit layout basins after the
initial grid relaxes, while each cool-down phase raises alpha enough to repair
boundary and spacing violations before the next restart.
AXIS: lr_sgdr_warm_restarts with alpha_coupled_inverse_lr, alpha_cyclic, and
TopFarm low-momentum Adam.
LESSON: pending
"""
import jax.numpy as jnp


def schedule_fn(step, total_steps, lr0, alpha0):
    t = step / jnp.maximum(total_steps - 1.0, 1.0)

    hold_frac = 0.26
    p = jnp.maximum((t - hold_frac) / (1.0 - hold_frac), 0.0)
    p = jnp.minimum(p, 1.0)

    # SGDR-style warm restarts: short broad search, medium search, long polish.
    n_cycles = 3.0
    c1 = 0.24
    c2 = 0.58

    in_first = p < c1
    in_second = p < c2

    t1 = p / c1
    t2 = (p - c1) / (c2 - c1)
    t3 = (p - c2) / (1.0 - c2)
    cycle_t = jnp.where(in_first, t1, jnp.where(in_second, t2, t3))
    cycle_t = jnp.minimum(1.0, jnp.maximum(0.0, cycle_t))

    peak = jnp.where(in_first, 0.95, jnp.where(in_second, 0.58, 0.34))
    floor = jnp.where(in_first, 0.070, jnp.where(in_second, 0.030, 0.004))
    restart_cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_t))
    cycle_scale = floor + (peak - floor) * restart_cosine

    end_gate_raw = (p - 0.82) / 0.18
    end_gate = jnp.minimum(1.0, jnp.maximum(0.0, end_gate_raw))
    end_gate = end_gate * end_gate * (3.0 - 2.0 * end_gate)

    lr_scale = jnp.where(t < hold_frac, 1.0, cycle_scale * (1.0 - 0.35 * end_gate))
    lr = lr0 * lr_scale

    coupled = alpha0 * lr0 / jnp.maximum(lr, 1e-10)
    cycle_cool = 1.0 - restart_cosine
    repair_ramp = 1.0 + 0.75 * p * p + 2.5 * end_gate
    alpha_cycle = 1.0 + 0.35 * cycle_cool
    alpha = jnp.where(t < hold_frac, alpha0, coupled * repair_ramp * alpha_cycle)

    beta1 = 0.1
    beta2 = 0.2

    return lr, alpha, beta1, beta2
