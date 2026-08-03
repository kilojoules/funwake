import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f
    
    D_f = float(D)
    gamma_min_f = float(gamma_min)
    
    # 1. 3-Cycle Cosine Annealing (SGDR)
    # Provides three distinct exploration phases.
    lr_max = 1.35 * D_f 
    
    is_warmup = progress < 0.05
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / 0.05)
    
    def cosine_anneal(p, p_start, p_end, lr_start, lr_end):
        p_norm = jnp.clip((p - p_start) / (p_end - p_start), 0.0, 1.0)
        return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + jnp.cos(jnp.pi * p_norm))
        
    lr_c1 = cosine_anneal(progress, 0.05, 0.45, lr_max, gamma_min_f * 2.0)
    lr_c2 = cosine_anneal(progress, 0.45, 0.75, lr_max * 0.60, gamma_min_f * 1.5)
    lr_c3 = cosine_anneal(progress, 0.75, 0.92, lr_max * 0.25, gamma_min_f)
    
    lr_main = jnp.where(progress < 0.45, lr_c1,
              jnp.where(progress < 0.75, lr_c2, lr_c3))
    lr = jnp.where(is_warmup, lr_warmup, 
         jnp.where(progress >= 0.92, gamma_min_f, lr_main))

    # 2. ADMM-Style Staircase Penalty (Decoupled, delayed, smoothed)
    # Penalty is flat during each cycle's descent, stepping up smoothly 
    # exactly at the cycle boundaries. This avoids the constant loss-landscape 
    # warping of a continuously rising penalty.
    
    def smooth_step(p, p_edge, width=0.015):
        return 0.5 * (1.0 + jnp.tanh((p - p_edge) / width))
        
    alpha_c1 = alpha0 * 0.1   # Relaxed penalty for maximal layout movement
    alpha_c2 = alpha0 * 3.0   # Moderate enforcement
    alpha_c3 = alpha0 * 20.0  # Strict enforcement
    
    step1 = smooth_step(progress, 0.45)
    step2 = smooth_step(progress, 0.75)
    
    alpha_main = alpha_c1 + (alpha_c2 - alpha_c1) * step1 + (alpha_c3 - alpha_c2) * step2
                 
    # Terminal feasibility spike ensuring absolute compliance
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(progress >= 0.92, alpha_terminal, alpha_main)
    
    # 3. Synchronized Moments
    # Smoothly transition Adam moments across cycles to match penalty stiffness.
    # Lower momentum and higher beta2 as constraints tighten.
    b1_c1, b2_c1 = 0.15, 0.20
    b1_c2, b2_c2 = 0.08, 0.70
    b1_c3, b2_c3 = 0.04, 0.90
    
    beta1_main = b1_c1 + (b1_c2 - b1_c1) * step1 + (b1_c3 - b1_c2) * step2
    beta2_main = b2_c1 + (b2_c2 - b2_c1) * step1 + (b2_c3 - b2_c2) * step2
                 
    beta1 = jnp.where(progress >= 0.92, 0.01, beta1_main)
    beta2 = jnp.where(progress >= 0.92, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2