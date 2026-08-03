import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR: Multi-Cycle Cosine Annealing with Warm Restarts ---
    # 3 cycles to allow multiple exploration-exploitation phases.
    n_cycles = 3.0
    
    # Scale progress so the 3 cycles finish exactly at 0.95 (leaving 5% for terminal)
    eff_progress = jnp.minimum(progress / 0.95, 1.0)
    
    cyc_f = jnp.floor(eff_progress * n_cycles)
    # Clamp to prevent a 4th cycle when eff_progress == 1.0
    cyc_f = jnp.minimum(cyc_f, n_cycles - 1.0)
    
    local_progress = (eff_progress * n_cycles) - cyc_f
    
    # Peak LR decays across cycles (e.g., 1.5*D -> 0.6*D -> 0.24*D)
    lr_peak = (1.5 * D_f) * (0.4 ** cyc_f)
    
    cosine_mult = 0.5 * (1.0 + jnp.cos(jnp.pi * local_progress))
    lr_main = gamma_min_f + (lr_peak - gamma_min_f) * cosine_mult
    
    is_terminal = progress >= 0.95
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cycle-Stepped, Intra-Cycle Ramping Alpha ---
    # Alpha scales up each cycle (mimicking Augmented Lagrangian multiplier updates).
    # Within each cycle, alpha ramps up as the learning rate drops, creating 
    # mid-run feasibility restorations at the end of cycles 1 and 2.
    
    alpha_base_cyc = alpha0 * (4.0 ** cyc_f) 
    alpha_peak_cyc = alpha_base_cyc * 2.5
    
    alpha_main = alpha_base_cyc + (alpha_peak_cyc - alpha_base_cyc) * (1.0 - cosine_mult)
    
    # Terminal feasibility spike ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Cyclic Adam Moments ---
    # Synchronize with the restarts: high momentum/low beta2 at start of cycle
    # to punch through landscape, high damping/high beta2 at cycle end for stability.
    
    b1_start, b2_start = 0.15, 0.20
    b1_end, b2_end = 0.05, 0.85
    
    beta1_main = b1_end + (b1_start - b1_end) * cosine_mult
    beta2_main = b2_end + (b2_start - b2_end) * cosine_mult
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2