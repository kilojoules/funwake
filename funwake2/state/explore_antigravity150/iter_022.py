import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-cycle Cosine Annealing (SGDR) ---
    # 3 cycles of exploration, followed by a terminal cooldown.
    # Each cycle triggers a "warm restart" (LR jumps up) to escape local minima.
    terminal_start = 0.90
    n_cycles = 3.0
    
    p_main = jnp.clip(progress / terminal_start, 0.0, 1.0)
    cycle_idx = jnp.minimum(jnp.floor(p_main * n_cycles), n_cycles - 1.0)
    cycle_p = (p_main * n_cycles) - cycle_idx
    
    lr_max = 1.5 * D_f
    lr_min = 0.05 * D_f
    
    # Decreasing exploration peaks per cycle (1.0x, 0.75x, 0.5x)
    lr_max_scaled = lr_max * (1.0 - cycle_idx * 0.25) 
    
    lr_cycle = lr_min + 0.5 * (lr_max_scaled - lr_min) * (1.0 + jnp.cos(jnp.pi * cycle_p))
    
    # Terminal phase cooldown
    p_terminal = jnp.clip((progress - terminal_start) / (1.0 - terminal_start), 0.0, 1.0)
    lr_terminal_decay = lr_min - (lr_min - gamma_min_f) * p_terminal
    
    is_terminal = progress >= terminal_start
    lr = jnp.where(is_terminal, lr_terminal_decay, lr_cycle)

    # --- 2. Cyclic Alpha with Mid-run Feasibility Bursts ---
    # Alpha is anti-correlated with LR. When LR jumps up (warm restart), alpha drops
    # to allow unconstrained layout shifting. As LR decays, alpha rises to a burst peak.
    alpha_base = alpha0 * 0.1
    alpha_burst = alpha0 * 8.0
    
    # Increasing penalty peaks per cycle (1.0x, 1.5x, 2.0x)
    alpha_burst_scaled = alpha_burst * (1.0 + cycle_idx * 0.5) 
    
    # Cosine wave from alpha_base to alpha_burst_scaled
    alpha_cycle = alpha_burst_scaled - 0.5 * (alpha_burst_scaled - alpha_base) * (1.0 + jnp.cos(jnp.pi * cycle_p))
    
    # Smooth ramp to absolute max penalty during terminal phase
    final_burst = alpha_burst * (1.0 + (n_cycles - 1.0) * 0.5)
    max_penalty = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha_terminal = final_burst + (max_penalty - final_burst) * p_terminal
    
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)

    # --- 3. Synchronized Adam Moments ---
    # Moments follow the penalty phase (anti-correlated with LR)
    # High momentum (low beta2) during warm restarts, high damping at cycle ends.
    b1_start, b1_end = 0.12, 0.04
    b2_start, b2_end = 0.15, 0.85
    
    beta1_cycle = b1_end + 0.5 * (b1_start - b1_end) * (1.0 + jnp.cos(jnp.pi * cycle_p))
    beta2_cycle = b2_end - 0.5 * (b2_end - b2_start) * (1.0 + jnp.cos(jnp.pi * cycle_p))
    
    # Terminal damping
    beta1_terminal = b1_end - (b1_end - 0.01) * p_terminal
    beta2_terminal = b2_end + (0.99 - b2_end) * p_terminal
    
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_cycle)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_cycle)

    return lr, alpha, beta1, beta2