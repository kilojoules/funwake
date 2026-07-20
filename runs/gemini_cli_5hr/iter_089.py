import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 089: 5-cycle Cosine Annealing with Cyclic Alpha.
    - More cycles (5) to explore more local optima.
    - Alpha cycles from low to high in each cycle.
    - Global alpha ramp to ensure final feasibility.
    """
    n_cycles = 5
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Slightly higher peak for more exploration
    lr_peak = lr0 * 6.0 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global scale increases over time
    alpha_global = alpha0 * (1.0 + 100.0 * (t_global ** 2))
    
    # Cyclic part: low at start of cycle, high at end
    # Use a sigmoid-like ramp within the cycle
    alpha_cyclic = jax.nn.sigmoid(10.0 * (t_cycle - 0.4))
    
    # Combine global ramp and cyclic behavior
    alpha_base = alpha_global * (0.1 + 0.9 * alpha_cyclic)
    
    # Coupling to LR to maintain magnitude relative to gradient
    alpha = alpha_base * (lr0 * 6.0 / jnp.maximum(lr, 1e-10))
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_last_cycle = (cycle_idx == n_cycles - 1)
    is_squeeze = is_last_cycle & (t_cycle > 0.98)
    
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    # Reactive but slightly more stable than the most extreme attempts
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2

import jax
