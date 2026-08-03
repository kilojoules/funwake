import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- Cyclic Breathing Constraints (SGDR for LR + Alpha) ---
    # We divide the main optimization into 3 cycles of increasing length.
    # Each cycle starts with a high LR (exploration) and low penalty (alpha),
    # then cosine-decays the LR to gamma_min while cosine-ramping alpha to a peak.
    # This acts as an aggressive "warm restart" that jumps out of local minima
    # while progressively tightening the feasibility requirements each cycle.
    
    t1 = 0.20
    t2 = 0.50
    t3 = 0.90
    
    cp1 = progress / t1
    cp2 = (progress - t1) / (t2 - t1)
    cp3 = (progress - t2) / (t3 - t2)
    
    cycle_progress = jnp.where(progress < t1, cp1,
                     jnp.where(progress < t2, cp2, cp3))
    cycle_progress = jnp.clip(cycle_progress, 0.0, 1.0)
    
    # Cosine decay factor: 1.0 at start of cycle, 0.0 at end of cycle
    cosine_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    # Cosine rise factor: 0.0 at start of cycle, 1.0 at end of cycle
    cosine_rise = 1.0 - cosine_decay
    
    # Cycle Peak Definitions
    # LR peaks decay across cycles; Alpha peaks escalate.
    peak_lr1 = 1.60 * D_f
    peak_lr2 = 0.80 * D_f
    peak_lr3 = 0.40 * D_f
    
    current_lr_peak = jnp.where(progress < t1, peak_lr1,
                      jnp.where(progress < t2, peak_lr2, peak_lr3))
    
    base_a1, peak_a1 = alpha0_f * 0.1, alpha0_f * 3.0
    base_a2, peak_a2 = alpha0_f * 0.5, alpha0_f * 10.0
    base_a3, peak_a3 = alpha0_f * 1.5, alpha0_f * 30.0
    
    current_alpha_base = jnp.where(progress < t1, base_a1,
                         jnp.where(progress < t2, base_a2, base_a3))
    current_alpha_peak = jnp.where(progress < t1, peak_a1,
                         jnp.where(progress < t2, peak_a2, peak_a3))
                         
    # Calculate main cyclic values
    lr_main = gamma_min_f + (current_lr_peak - gamma_min_f) * cosine_decay
    alpha_main = current_alpha_base + (current_alpha_peak - current_alpha_base) * cosine_rise
    
    # Cyclic Adam Moments
    # Momentum (beta1) drops and beta2 rises as penalty tightens in each cycle
    b1_b1, b1_p1 = 0.15, 0.05
    b1_b2, b1_p2 = 0.10, 0.02
    b1_b3, b1_p3 = 0.05, 0.01
    
    current_b1_base = jnp.where(progress < t1, b1_b1,
                      jnp.where(progress < t2, b1_b2, b1_b3))
    current_b1_peak = jnp.where(progress < t1, b1_p1,
                      jnp.where(progress < t2, b1_p2, b1_p3))
                      
    b2_b1, b2_p1 = 0.20, 0.80
    b2_b2, b2_p2 = 0.50, 0.90
    b2_b3, b2_p3 = 0.70, 0.95
    
    current_b2_base = jnp.where(progress < t1, b2_b1,
                      jnp.where(progress < t2, b2_b2, b2_b3))
    current_b2_peak = jnp.where(progress < t1, b2_p1,
                      jnp.where(progress < t2, b2_p2, b2_p3))
                      
    beta1_main = current_b1_base + (current_b1_peak - current_b1_base) * cosine_rise
    beta2_main = current_b2_base + (current_b2_peak - current_b2_base) * cosine_rise
    
    # --- Terminal Feasibility Restoration ---
    # Extreme penalty and minimal LR in the last 10% to guarantee constraints are met.
    is_terminal = progress >= t3
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    beta1_terminal = 0.01
    beta2_terminal = 0.99
    
    lr = jnp.where(is_terminal, lr_terminal, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_main)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_main)
    
    return lr, alpha, beta1, beta2