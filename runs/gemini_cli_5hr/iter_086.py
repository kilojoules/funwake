import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 086: Cosine Annealing with Warm Restarts.
    - 4 cycles of cosine decay.
    - Alpha inversely proportional to LR within each cycle, but with a global ramp.
    - Local alpha dips at restart to allow exploration.
    - Reactive beta1/beta2.
    """
    # ── Cycles ──────────────────────────────────────────────────
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    # Progress within current cycle (0 to 1)
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    # Global progress (0 to 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Decay the peak LR of each cycle slightly
    lr_peak = lr0 * 4.0 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.01
    
    # Cosine decay within cycle
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global alpha ramp to ensure final feasibility
    alpha_global = alpha0 * (1.0 + 10.0 * (t_global ** 2))
    
    # Local coupling: higher alpha when LR is lower
    # Use a 'soft' coupling to avoid extreme values
    alpha_local = alpha_global * (lr_peak / jnp.maximum(lr, 1e-10))
    
    # Soften alpha at the start of each cycle (except the first one maybe?)
    # or even the first one to allow some initial shuffling.
    # Dip alpha at the start of the cycle
    dip_width = 0.15
    dip = jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    
    # We want alpha to be LOWER during the dip
    alpha = alpha_local * (1.0 - 0.8 * dip)
    
    # Final squeeze in the very last few steps
    is_last_cycle = (cycle_idx == n_cycles - 1)
    is_squeeze = is_last_cycle & (t_cycle > 0.95)
    
    lr = jnp.where(is_squeeze, lr0 * 0.001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    # More momentum as we cool down in each cycle?
    # Or keep it reactive? Let's try reactive but slightly more stable than 082.
    beta1 = 0.2
    beta2 = 0.5
    
    return lr, alpha, beta1, beta2
