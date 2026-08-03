import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- 1. Cyclic Cosine Annealing (SGDR) with Warm Restarts ---
    # Structurally different from WSD: we use 3 distinct cycles.
    # Each cycle starts with a high LR (exploration) and drops to gamma_min.
    # Cycle lengths increase progressively (20%, 30%, 40% of steps).
    c1_end = 0.20
    c2_end = 0.50
    c3_end = 0.90
    
    cp1 = progress / c1_end
    cp2 = (progress - c1_end) / (c2_end - c1_end)
    cp3 = (progress - c2_end) / (c3_end - c2_end)
    
    # Progress within the current cycle [0.0, 1.0]
    cycle_progress = jnp.where(progress < c1_end, cp1,
                     jnp.where(progress < c2_end, cp2,
                     jnp.where(progress < c3_end, cp3, 1.0)))
    
    # cos_val goes from 1.0 at the start of a cycle to 0.0 at the end
    cos_val = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    
    lr_min = gamma_min_f
    # Max LR decays with each warm restart
    lr_max_c1 = 1.60 * D_f
    lr_max_c2 = 1.00 * D_f
    lr_max_c3 = 0.50 * D_f
    
    current_lr_max = jnp.where(progress < c1_end, lr_max_c1,
                     jnp.where(progress < c2_end, lr_max_c2, lr_max_c3))
    
    lr_main = lr_min + (current_lr_max - lr_min) * cos_val
    
    # --- 2. Synchronized Cyclic Alpha (Mid-run Feasibility Bursts) ---
    # Decoupled from the inverse of LR, but synchronized with the cycles.
    # At the start of a cycle (high LR), alpha is low to allow barrier crossing.
    # As the cycle ends, alpha surges to restore feasibility (bursts).
    # The peak penalty grows stronger with each successive cycle.
    alpha_base = alpha0_f * 0.1
    
    alpha_peak_c1 = alpha0_f * 5.0
    alpha_peak_c2 = alpha0_f * 15.0
    alpha_peak_c3 = alpha0_f * 30.0
    
    current_alpha_peak = jnp.where(progress < c1_end, alpha_peak_c1,
                         jnp.where(progress < c2_end, alpha_peak_c2, alpha_peak_c3))
    
    # Anti-correlated with LR: when cos_val is 0 (low LR), alpha hits its peak
    alpha_main = alpha_base + (current_alpha_peak - alpha_base) * (1.0 - cos_val)
    
    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize moments with the SGDR cycles.
    # Exploratory (cos_val=1): low momentum/variance-damping to jump out of local minima.
    # Feasibility (cos_val=0): standard high momentum to follow constraint curvature.
    b1_expl, b2_expl = 0.12, 0.15
    b1_feas, b2_feas = 0.04, 0.85
    
    beta1_main = b1_feas + (b1_expl - b1_feas) * cos_val
    beta2_main = b2_feas + (b2_expl - b2_feas) * cos_val
    
    # --- 4. Terminal Feasibility Spike ---
    # Absolute compliance during the final 10% of the optimization.
    is_terminal = progress >= c3_end
    
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2