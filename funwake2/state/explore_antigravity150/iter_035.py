import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- 1. Multi-Cycle SGDR (Warm Restarts) ---
    # Two main cycles of cosine decay followed by a terminal fine-tuning phase.
    # Cycle 1: Global exploration (0.0 to 0.45)
    # Cycle 2: Local refinement with warm restart (0.45 to 0.90)
    # Terminal: Feasibility enforcement (0.90 to 1.0)
    cycle1_end = 0.45
    cycle2_end = 0.90
    
    pi = jnp.pi
    
    # Progress within each cycle (0.0 to 1.0)
    c1_prog = jnp.clip(progress / cycle1_end, 0.0, 1.0)
    c2_prog = jnp.clip((progress - cycle1_end) / (cycle2_end - cycle1_end), 0.0, 1.0)
    
    # Cosine factors (1.0 at start of cycle, 0.0 at end of cycle)
    c1_cos = 0.5 * (1.0 + jnp.cos(pi * c1_prog))
    c2_cos = 0.5 * (1.0 + jnp.cos(pi * c2_prog))
    
    # Peak LR is higher than WSD because it immediately begins to decay
    lr_max_1 = 1.5 * D_f
    lr_max_2 = 0.75 * D_f  # Second cycle has half the peak LR
    
    lr_c1 = gamma_min_f + (lr_max_1 - gamma_min_f) * c1_cos
    lr_c2 = gamma_min_f + (lr_max_2 - gamma_min_f) * c2_cos
    
    lr_main = jnp.where(progress < cycle1_end, lr_c1, lr_c2)
    
    is_terminal = progress >= cycle2_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Cyclic Alpha (Synchronized with LR) ---
    # As LR decays, alpha rises to enforce constraints (mid-run feasibility bursts).
    # When LR restarts, alpha drops to allow free movement again, but to a 
    # higher floor to prevent entirely undoing the progress from Cycle 1.
    
    a_base_1 = alpha0_f * 0.1
    a_peak_1 = alpha0_f * 5.0
    
    a_base_2 = alpha0_f * 1.0
    a_peak_2 = alpha0_f * 20.0
    
    # Inverse cosine: starts at base, ends at peak
    alpha_c1 = a_peak_1 - (a_peak_1 - a_base_1) * c1_cos
    alpha_c2 = a_peak_2 - (a_peak_2 - a_base_2) * c2_cos
    
    alpha_main = jnp.where(progress < cycle1_end, alpha_c1, alpha_c2)
    
    # Terminal spike for absolute compliance
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Cyclic Adam Moments ---
    # High momentum (high b1, low b2) at cycle starts for exploration.
    # Damped moments (low b1, high b2) at cycle ends to absorb penalty curvature.
    
    b1_start_1, b1_end_1 = 0.15, 0.05
    b2_start_1, b2_end_1 = 0.10, 0.85
    
    b1_start_2, b1_end_2 = 0.10, 0.02
    b2_start_2, b2_end_2 = 0.20, 0.95
    
    beta1_c1 = b1_end_1 + (b1_start_1 - b1_end_1) * c1_cos
    beta1_c2 = b1_end_2 + (b1_start_2 - b1_end_2) * c2_cos
    
    beta2_c1 = b2_end_1 - (b2_end_1 - b2_start_1) * c1_cos
    beta2_c2 = b2_end_2 - (b2_end_2 - b2_start_2) * c2_cos
    
    beta1_main = jnp.where(progress < cycle1_end, beta1_c1, beta1_c2)
    beta2_main = jnp.where(progress < cycle1_end, beta2_c1, beta2_c2)
    
    # Extreme damping in terminal phase
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2