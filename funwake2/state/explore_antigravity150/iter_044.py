import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Two-Phase WSD (Warmup-Stable-Decay) ---
    # Structurally different from single-WSD or cosine SGDR:
    # Phase 1: High exploration (macro-layout discovery) with very low penalty.
    # Phase 2: Moderate exploration (micro-layout refinement) with a high penalty plateau.
    # Terminal: Strict feasibility restoration (Filter method).
    
    p1_end = 0.45
    p2_end = 0.90
    
    # --- Phase 1: Macro-Exploration ---
    lr_max1 = 1.60 * D_f
    p1_warmup = 0.05
    p1_stable = 0.25
    
    lr_p1_warm = gamma_min_f + (lr_max1 - gamma_min_f) * (progress / p1_warmup)
    lr_p1_decay_prog = jnp.clip((progress - p1_stable) / (p1_end - p1_stable), 0.0, 1.0)
    lr_p1_decay = lr_max1 - (lr_max1 - gamma_min_f) * lr_p1_decay_prog
    
    lr_p1 = jnp.where(progress < p1_warmup, lr_p1_warm,
              jnp.where(progress < p1_stable, lr_max1, lr_p1_decay))

    # --- Phase 2: Micro-Refinement ---
    lr_max2 = 0.80 * D_f
    p2_warmup = 0.50
    p2_stable = 0.70
    
    lr_p2_warm_prog = jnp.clip((progress - p1_end) / (p2_warmup - p1_end), 0.0, 1.0)
    lr_p2_warm = gamma_min_f + (lr_max2 - gamma_min_f) * lr_p2_warm_prog
    
    lr_p2_decay_prog = jnp.clip((progress - p2_stable) / (p2_end - p2_stable), 0.0, 1.0)
    lr_p2_decay = lr_max2 - (lr_max2 - gamma_min_f) * lr_p2_decay_prog
    
    lr_p2 = jnp.where(progress < p2_warmup, lr_p2_warm,
              jnp.where(progress < p2_stable, lr_max2, lr_p2_decay))

    # Combine LR phases
    in_p1 = progress < p1_end
    is_terminal = progress >= p2_end
    
    lr_main = jnp.where(in_p1, lr_p1, lr_p2)
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Decoupled Alpha with Logistic Transition ---
    # Instead of cyclic alpha or 1/lr coupling, we use a logistic ramp synchronized
    # with the boundary between Phase 1 and Phase 2.
    # This allows turbines to cross constraints freely in Phase 1, then pushes
    # them into a valid configuration during Phase 2.
    
    alpha_p1 = alpha0 * 0.05
    alpha_p2 = alpha0 * 20.0
    
    # Steep logistic ramp centered at 0.48 (during Phase 2 warmup)
    k = 40.0
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k * (progress - 0.48)))
    
    alpha_main = alpha_p1 + (alpha_p2 - alpha_p1) * logistic_ramp
    
    # Terminal feasibility spike ensures absolute constraint compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Synchronized Adam Moments ---
    # High momentum (low beta1, low beta2) in Phase 1 to jump out of local minima.
    # High damping (high beta2) in Phase 2 to absorb penalty curvature and stabilize.
    
    b1_start, b2_start = 0.25, 0.15
    b1_plateau, b2_plateau = 0.02, 0.90
    
    beta1_main = b1_start + (b1_plateau - b1_start) * logistic_ramp
    beta2_main = b2_start + (b2_plateau - b2_start) * logistic_ramp
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2