import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. SGDR (Cosine Annealing with Warm Restarts) Learning Rate ---
    # We use 3 cycles. Cycle lengths: 40%, 35%, 20% of the total steps.
    c1_end = 0.40
    c2_end = 0.75
    c3_end = 0.95
    
    p_c1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p_c2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    p_c3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)
    
    lr_max_1 = 1.6 * D_f
    lr_max_2 = 0.8 * D_f
    lr_max_3 = 0.4 * D_f
    
    lr_c1 = gamma_min_f + 0.5 * (lr_max_1 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c1))
    lr_c2 = gamma_min_f + 0.5 * (lr_max_2 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c2))
    lr_c3 = gamma_min_f + 0.5 * (lr_max_3 - gamma_min_f) * (1.0 + jnp.cos(jnp.pi * p_c3))
    
    lr_main = jnp.where(progress < c1_end, lr_c1,
                jnp.where(progress < c2_end, lr_c2, lr_c3))

    # --- 2. Cyclic Decoupled Penalty (Alpha) ---
    # Anti-correlated with LR: alpha starts low (soft landscape) when LR is high
    # to allow massive exploration, then peaks at the end of each cycle to force
    # mid-run feasibility-restoration bursts before the next warm restart scrambles it.
    alpha_base_1, alpha_peak_1 = alpha0 * 0.05, alpha0 * 5.0
    alpha_base_2, alpha_peak_2 = alpha0 * 0.50, alpha0 * 20.0
    alpha_base_3, alpha_peak_3 = alpha0 * 2.00, alpha0 * 50.0
    
    alpha_c1 = alpha_base_1 + 0.5 * (alpha_peak_1 - alpha_base_1) * (1.0 - jnp.cos(jnp.pi * p_c1))
    alpha_c2 = alpha_base_2 + 0.5 * (alpha_peak_2 - alpha_base_2) * (1.0 - jnp.cos(jnp.pi * p_c2))
    alpha_c3 = alpha_base_3 + 0.5 * (alpha_peak_3 - alpha_base_3) * (1.0 - jnp.cos(jnp.pi * p_c3))
    
    alpha_main = jnp.where(progress < c1_end, alpha_c1,
                   jnp.where(progress < c2_end, alpha_c2, alpha_c3))

    # --- 3. Cyclic Adam Moments ---
    # High momentum (low beta2) during high-LR exploration; damping during feasibility restoration.
    b1_high, b1_low = 0.12, 0.02
    b2_low, b2_high = 0.15, 0.90
    
    beta1_c1 = b1_low + 0.5 * (b1_high - b1_low) * (1.0 + jnp.cos(jnp.pi * p_c1))
    beta1_c2 = b1_low + 0.5 * (b1_high - b1_low) * (1.0 + jnp.cos(jnp.pi * p_c2))
    beta1_c3 = b1_low + 0.5 * (b1_high - b1_low) * (1.0 + jnp.cos(jnp.pi * p_c3))

    beta2_c1 = b2_high + 0.5 * (b2_low - b2_high) * (1.0 + jnp.cos(jnp.pi * p_c1))
    beta2_c2 = b2_high + 0.5 * (b2_low - b2_high) * (1.0 + jnp.cos(jnp.pi * p_c2))
    beta2_c3 = b2_high + 0.5 * (b2_low - b2_high) * (1.0 + jnp.cos(jnp.pi * p_c3))
    
    beta1_main = jnp.where(progress < c1_end, beta1_c1,
                   jnp.where(progress < c2_end, beta1_c2, beta1_c3))
    beta2_main = jnp.where(progress < c1_end, beta2_c1,
                   jnp.where(progress < c2_end, beta2_c2, beta2_c3))

    # --- 4. Terminal Feasibility Spike ---
    is_terminal = progress >= c3_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2