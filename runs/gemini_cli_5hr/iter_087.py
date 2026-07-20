import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 087: Refined Cosine Annealing with Warm Restarts.
    - 4 cycles of cosine decay.
    - Stronger global alpha ramp.
    - More aggressive squeeze phase.
    - Reduced dip in later cycles.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 5.0 * (1.0 - 0.6 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Stronger global ramp
    alpha_global = alpha0 * (1.0 + 50.0 * (t_global ** 3))
    
    # Coupling
    alpha_local = alpha_global * (lr0 * 5.0 / jnp.maximum(lr, 1e-10))
    
    # Dip alpha at the start of the cycle, less so in later cycles
    dip_magnitude = 0.9 * (1.0 - 0.7 * t_global)
    dip_width = 0.1
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_last_cycle = (cycle_idx == n_cycles - 1)
    is_squeeze = is_last_cycle & (t_cycle > 0.97)
    
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    # Aggressive squeeze for feasibility
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2
