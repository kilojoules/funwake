import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Multi-Cycle SGDR with Synchronized Feasibility Bursts ---
    # Divides the main run into distinct phases (cycles). 
    # Each cycle starts with an explosive layout search (high LR, low penalty)
    # and ends with a local feasibility restoration (low LR, high penalty).
    # This prevents the solver from getting trapped by early constraint walls.
    terminal_start = 0.90
    n_cycles = 3.0
    
    # Map progress [0, 0.90] into cycle phases
    phase = (progress / terminal_start) * n_cycles
    cycle_idx = jnp.clip(jnp.floor(phase), 0.0, n_cycles - 1.0)
    cycle_frac = jnp.clip(phase - cycle_idx, 0.0, 1.0)

    # Multi-cycle Learning Rate (Cosine Annealing with Warm Restarts)
    # Peak LR decays each cycle: 1.6*D -> 0.96*D -> 0.576*D
    lr_max_base = 1.6 * D_f
    lr_max_i = lr_max_base * (0.6 ** cycle_idx)
    
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_frac))
    lr_cycle = gamma_min_f + (lr_max_i - gamma_min_f) * cosine_mult

    # Cyclic Alpha Penalty (Mid-Run Feasibility Restorations)
    # Ramps sharply at the end of each cycle as the learning rate decays.
    alpha_base = alpha0 * 0.1
    # Peaks get progressively stiffer across cycles: 5x, 10x, 15x alpha0
    alpha_peak_i = alpha0 * (5.0 + 5.0 * cycle_idx)  
    
    # Power of 4 delays the feasibility ramp to the final 20% of each cycle
    alpha_cycle = alpha_base + (alpha_peak_i - alpha_base) * (cycle_frac ** 4.0)

    # Phase-Synchronized Adam Moments
    # High momentum & low beta2 early in cycle for fast topological shifting.
    # Low momentum & high beta2 late in cycle to damp oscillations around stiff boundaries.
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.20, 0.90
    
    beta1_cycle = b1_start + (b1_end - b1_start) * (cycle_frac ** 2.0)
    beta2_cycle = b2_start + (b2_end - b2_start) * (cycle_frac ** 2.0)

    # --- 2. Terminal Phase (Absolute Feasibility & Cooldown) ---
    is_terminal = progress >= terminal_start
    
    # Linear cooldown of LR from exactly gamma_min to 0.1 * gamma_min
    terminal_frac = jnp.clip((progress - terminal_start) / (1.0 - terminal_start), 0.0, 1.0)
    lr_terminal = gamma_min_f * (1.0 - 0.9 * terminal_frac)
    
    # Terminal filter method penalty spike guarantees hard constraint compliance
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, lr_terminal, lr_cycle)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2