import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)
    
    # We allocate 90% of the budget to cyclic optimization, 10% to terminal feasibility
    cycle_end = 0.90
    num_cycles = 3.0
    
    # Map progress up to cycle_end into [0.0, 3.0]
    c_time = (progress / cycle_end) * num_cycles
    
    # Restrict cycle index to [0, 1, 2]
    cycle_idx = jnp.clip(jnp.floor(c_time), 0.0, num_cycles - 1.0)
    
    # Progress within the current cycle [0.0, 1.0]
    c_progress = jnp.clip(c_time - cycle_idx, 0.0, 1.0)

    # --- 1. SGDR Multi-Cycle Cosine Learning Rate ---
    # Decaying peak learning rate for each cycle with warm restarts.
    # Cycle 0: massive exploration. Cycle 1: medium. Cycle 2: fine-tuning.
    lr_peak = (1.5 * D_f) / (2.0 ** cycle_idx)
    
    # Cosine annealing within the cycle
    lr_main = gamma_min_f + 0.5 * (lr_peak - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * c_progress))
    
    is_terminal = progress >= cycle_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Alpha (Mid-run Feasibility Bursts) ---
    # Instead of a single continuous plateau, alpha stays soft early in each cycle
    # for free-space exploration, then exponentially bursts at the end of the 
    # cycle to pull turbines back toward feasibility boundaries.
    
    alpha_base = alpha0 * 0.1
    # Each cycle enforces a progressively stricter penalty peak (5x, 10x, 15x)
    alpha_cycle_peak = alpha0 * (5.0 * (cycle_idx + 1.0))
    
    # Burst shape: sharply rises as cycle progress approaches 1.0
    # exp(10*(x-1)) goes from ~4.5e-5 at x=0 to 1.0 at x=1.
    burst = jnp.exp(10.0 * (c_progress - 1.0))
    alpha_main = alpha_base + (alpha_cycle_peak - alpha_base) * burst
    
    # Terminal feasibility spike ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Synchronized Cyclic Adam Moments ---
    # High momentum (higher beta1 in context of this problem) / low beta2 
    # during exploration to allow layout shifts.
    # Drop momentum (lower beta1) and raise beta2 during the constraint bursts 
    # to heavily damp oscillations against the penalty walls.
    
    b1_start, b2_start = 0.12, 0.15
    b1_end, b2_end = 0.04, 0.85
    
    beta1_main = b1_start + (b1_end - b1_start) * burst
    beta2_main = b2_start + (b2_end - b2_start) * burst
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2