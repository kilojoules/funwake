import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Structurally Different: Multi-Cycle SGDR (Warm Restarts) ---
    # Moving away from a single monotonic phase. We use a two-cycle approach:
    # Cycle 1 (0% - 60%): Massive exploration, ending in a moderate "mid-run feasibility burst".
    # Cycle 2 (60% - 95%): Secondary exploration with a lower peak, ending in a strict penalty.
    # Terminal (95% - 100%): Absolute feasibility freeze.
    
    T1 = 0.60
    terminal_start = 0.95
    
    is_cycle1 = progress < T1
    is_terminal = progress >= terminal_start

    # Normalize progress within the currently active cycle [0, 1]
    cycle_prog = jnp.where(
        is_cycle1,
        progress / T1,
        (progress - T1) / (terminal_start - T1)
    )
    cycle_prog = jnp.clip(cycle_prog, 0.0, 1.0) # Safety clip

    # Cosine annealing for Learning Rate
    lr_peak = jnp.where(is_cycle1, 1.5 * D_f, 0.4 * D_f)
    lr_cycle = gamma_min_f + (lr_peak - gamma_min_f) * 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_prog))
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_cycle)

    # --- 2. Cyclic & Pulsed Penalty (Alpha) ---
    # Alpha "breathes" inversely with the LR. It drops at the start of each cycle 
    # to permit layout fluidity, then ramps up sharply to force feasibility at cycle boundaries.
    
    alpha_base = alpha0 * 0.1
    # Cycle 1 has a moderate plateau (allows layout settling), Cycle 2 has a very stiff plateau
    alpha_peak = jnp.where(is_cycle1, alpha0 * 3.0, alpha0 * 20.0)
    
    # Quartic ramp (x^4): Keeps penalty low for most of the cycle to allow movement, 
    # then spikes sharply at the end of the cycle to corral the turbines into valid spacing.
    alpha_cycle = alpha_base + (alpha_peak - alpha_base) * (cycle_prog ** 4.0)
    
    # Terminal feasibility spike (filter method) - massive penalty restoration
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cycle)

    # --- 3. Cyclic Adam Moments ---
    # Momentum (beta1) and variance decay (beta2) are also cyclic.
    # Start of cycles: low beta2 (accept variance) and high beta1 (agile momentum).
    # End of cycles: high beta2 (smooth gradients out) and low beta1 (damping).
    
    b1_start = jnp.where(is_cycle1, 0.15, 0.10)
    b1_end = jnp.where(is_cycle1, 0.05, 0.02)
    b2_start = jnp.where(is_cycle1, 0.10, 0.40)
    b2_end = jnp.where(is_cycle1, 0.85, 0.96)

    beta1_cycle = b1_start + (b1_end - b1_start) * cycle_prog
    beta2_cycle = b2_start + (b2_end - b2_start) * cycle_prog

    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_cycle)
    beta2 = jnp.where(is_terminal, 0.99, beta2_cycle)

    return lr, alpha, beta1, beta2