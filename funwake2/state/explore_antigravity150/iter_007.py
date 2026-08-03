import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    # Base exploratory peak proportional to D
    lr_max = 1.05 * float(D)
    gamma_min_f = float(gamma_min)

    # 1. Hold and Linear Cool-down LR (One-Cycle Variant)
    # Warmup to stabilize initial gradients, hold to explore broadly, 
    # and linearly cool down for predictable convergence.
    is_warmup = progress < 0.05
    is_hold = (progress >= 0.05) & (progress < 0.40)
    is_cooldown = (progress >= 0.40) & (progress < 0.90)
    is_terminal = progress >= 0.90
    
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / 0.05)
    lr_hold = lr_max
    lr_cooldown = lr_max - (lr_max - gamma_min_f) * ((progress - 0.40) / 0.50)
    
    lr = jnp.where(is_warmup, lr_warmup,
         jnp.where(is_hold, lr_hold,
         jnp.where(is_cooldown, lr_cooldown, gamma_min_f)))

    # 2. Delayed Bounded Logistic Alpha Ramp
    # Decouple alpha from lr: maintain a very low penalty during the hold phase
    # to allow free layout shuffling, then ramp via a smooth logistic curve
    # exactly as the layout cools down and begins to settle.
    k_steepness = 25.0
    t_mid = 0.60
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k_steepness * (progress - t_mid)))
    
    alpha_base = alpha0 * 0.5
    alpha_plateau = alpha0 * 30.0
    alpha_main = alpha_base + (alpha_plateau - alpha_base) * logistic_ramp
    
    # Terminal feasibility spike to enforce absolute constraints at the end
    alpha_terminal = alpha0 * float(D) / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # 3. Phase-transition Adam Moments via the Logistic Ramp
    # As alpha transitions from base to plateau, drop momentum (beta1) to avoid 
    # constraint overshooting, and raise curvature absorption (beta2) to 
    # dampen the stiff penalty gradients.
    beta1_main = 0.10 - 0.05 * logistic_ramp
    beta2_main = 0.20 + 0.70 * logistic_ramp
    
    beta1 = jnp.where(is_terminal, 0.05, beta1_main)
    beta2 = jnp.where(is_terminal, 0.90, beta2_main)

    return lr, alpha, beta1, beta2