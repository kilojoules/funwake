import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- SGDR with Cyclic Alpha & Warm Restarts ---
    # Implements structurally novel multi-cycle cosine annealing:
    # Cycle 1 (0% to 50%): Deep Exploration -> Mid-run Feasibility
    # Cycle 2 (50% to 90%): Refinement Restart -> Strict Feasibility
    # Terminal (90% to 100%): Absolute compliance phase
    
    c1_end = 0.50
    c2_end = 0.90
    
    p1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    
    # 1. Learning Rate (Cosine Annealing with Warm Restart)
    # A mid-run LR spike helps escape local minima found during the first feasibility phase.
    lr1_max, lr1_min = 1.50 * D_f, 0.10 * D_f
    lr2_max, lr2_min = 0.80 * D_f, gamma_min_f
    
    lr1 = lr1_min + 0.5 * (lr1_max - lr1_min) * (1.0 + jnp.cos(jnp.pi * p1))
    lr2 = lr2_min + 0.5 * (lr2_max - lr2_min) * (1.0 + jnp.cos(jnp.pi * p2))
    lr_term = gamma_min_f
    
    lr = jnp.where(progress < c1_end, lr1, 
           jnp.where(progress < c2_end, lr2, lr_term))
    
    # 2. Cyclic Decoupled Penalty (Anti-correlated with LR)
    # Alpha relaxes when LR spikes (restarts), permitting layouts to freely shift,
    # then ramps up smoothly with an inverted cosine to enforce constraints.
    a1_min, a1_max = alpha0 * 0.5, alpha0 * 12.0
    a2_min, a2_max = alpha0 * 2.0, alpha0 * 35.0
    
    alpha1 = a1_min + 0.5 * (a1_max - a1_min) * (1.0 - jnp.cos(jnp.pi * p1))
    alpha2 = a2_min + 0.5 * (a2_max - a2_min) * (1.0 - jnp.cos(jnp.pi * p2))
    
    # Terminal filter method spike for strict boundary/spacing compliance
    alpha_term = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    alpha = jnp.where(progress < c1_end, alpha1, 
              jnp.where(progress < c2_end, alpha2, alpha_term))
    
    # 3. Cyclic Adam Moments (Synchronized with alpha phase)
    # Momentum (beta1) drops and variance damping (beta2) rises exactly 
    # as the penalty constraint tightens.
    b1_c1_max, b1_c1_min = 0.15, 0.05
    b1_c2_max, b1_c2_min = 0.10, 0.02
    
    b2_c1_min, b2_c1_max = 0.10, 0.85
    b2_c2_min, b2_c2_max = 0.20, 0.95
    
    beta1_1 = b1_c1_min + 0.5 * (b1_c1_max - b1_c1_min) * (1.0 + jnp.cos(jnp.pi * p1))
    beta1_2 = b1_c2_min + 0.5 * (b1_c2_max - b1_c2_min) * (1.0 + jnp.cos(jnp.pi * p2))
    beta1_term = 0.01
    
    beta2_1 = b2_c1_max - 0.5 * (b2_c1_max - b2_c1_min) * (1.0 + jnp.cos(jnp.pi * p1))
    beta2_2 = b2_c2_max - 0.5 * (b2_c2_max - b2_c2_min) * (1.0 + jnp.cos(jnp.pi * p2))
    beta2_term = 0.99
    
    beta1 = jnp.where(progress < c1_end, beta1_1, 
              jnp.where(progress < c2_end, beta1_2, beta1_term))
    beta2 = jnp.where(progress < c1_end, beta2_1, 
              jnp.where(progress < c2_end, beta2_2, beta2_term))
    
    return lr, alpha, beta1, beta2