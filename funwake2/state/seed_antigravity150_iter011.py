import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. WSD (Warmup-Stable-Decay) Learning Rate ---
    # A structural departure from SGDR/Cosine: hold a high learning rate 
    # continuously for a long exploration phase, then cool down linearly.
    # This often outperforms cosine schedules for complex non-convex landscapes.
    warmup_end = 0.05
    stable_end = 0.50
    decay_end = 0.90
    
    lr_max = 1.25 * D_f  # Sustained high exploration rate
    
    # Warmup
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / warmup_end)
    
    # Stable
    lr_stable = lr_max
    
    # Decay (Linear cooldown)
    decay_progress = jnp.clip((progress - stable_end) / (decay_end - stable_end), 0.0, 1.0)
    lr_decay = lr_max - (lr_max - gamma_min_f) * decay_progress
    
    lr_main = jnp.where(progress < warmup_end, lr_warmup,
                jnp.where(progress < stable_end, lr_stable, lr_decay))
                
    is_terminal = progress >= decay_end
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. Decoupled, Delayed, Bounded Logistic Alpha ---
    # Exact-penalty method approach: we don't need alpha to go to infinity, 
    # just high enough to enforce constraints. We use a logistic curve to 
    # smoothly transition from 'exploration' to 'feasibility' mid-run.
    
    alpha_base = alpha0 * 0.1  # Soft penalty to allow massive layout shifts early
    alpha_plateau = alpha0 * 15.0  # Bounded penalty plateau for the main run
    
    # Logistic growth centered at progress = 0.40
    # k=25 ensures the transition mostly happens between progress 0.25 and 0.55
    k = 25.0
    p0 = 0.40
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k * (progress - p0)))
    
    alpha_main = alpha_base + (alpha_plateau - alpha_base) * logistic_ramp
    
    # Terminal feasibility spike (filter method) ensures absolute compliance at the end
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments ---
    # Synchronize moments with the logistic alpha ramp:
    # High momentum / low beta2 during exploration allows rapid layout shifting.
    # Drop momentum / raise beta2 as the penalty kicks in, to damp oscillations
    # and absorb the stiff curvature of the constraint boundaries.
    
    b1_start, b2_start = 0.12, 0.15
    b1_plateau, b2_plateau = 0.04, 0.85
    
    beta1_main = b1_start + (b1_plateau - b1_start) * logistic_ramp
    beta2_main = b2_start + (b2_plateau - b2_start) * logistic_ramp
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2