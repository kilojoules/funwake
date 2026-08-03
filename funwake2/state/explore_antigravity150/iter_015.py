import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. Epsilon-Constrained Shrinking Tolerance ---
    # We define a continuous constraint tolerance `epsilon(t)` that starts near D
    # (allowing full exploration) and shrinks smoothly to `gamma_min`.
    # The shape (1 - p^4)^2 provides a very long, flat plateau for the first 60%
    # of the run (like a WSD stable phase), then shrinks rapidly, landing smoothly at 0.
    shape = (1.0 - progress**4)**2
    epsilon = gamma_min_f + (D_f - gamma_min_f) * shape
    
    # --- 2. Physically Coupled Alpha (Penalty) ---
    # The penalty is coupled inversely to the shrinking tolerance epsilon.
    # This naturally creates a "delayed alpha ramp" that stays soft (~0.5*alpha0) 
    # for most of the run, then spikes automatically as epsilon -> gamma_min.
    # We ramp a multiplier to 1.0 at the end to perfectly hit the terminal spike.
    alpha_multiplier = 0.5 + 0.5 * jnp.clip((progress - 0.80) / 0.20, 0.0, 1.0)
    alpha_main = alpha_multiplier * alpha0 * (D_f / jnp.maximum(epsilon, 1e-30))
    
    # Terminal override in the last 2% to ensure absolute convergence
    is_terminal = progress >= 0.98
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Tolerance-Coupled Learning Rate ---
    # The learning rate step size tracks the allowable tolerance `epsilon`.
    # This keeps the optimizer from taking steps larger than the shrinking constraint boundary.
    lr_max = 1.25 * D_f
    warmup_end = 0.05
    
    # Linear warmup
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / warmup_end)
    # Main run exactly tracks epsilon shape
    lr_main = gamma_min_f + (lr_max - gamma_min_f) * shape
    
    lr = jnp.where(progress < warmup_end, lr_warmup, lr_main)
    lr = jnp.where(is_terminal, gamma_min_f, lr)

    # --- 4. Kinematic Adam Moments ---
    # Momentum (beta1) drops as epsilon shrinks to prevent overshooting boundaries.
    # beta2 (curvature) rises to absorb the stiffening penalty landscape.
    beta1_main = 0.02 + 0.13 * shape  # Ramps from 0.15 down to 0.02
    beta2_main = 0.98 - 0.78 * shape  # Ramps from 0.20 up to 0.98
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2