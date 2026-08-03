import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # 1. WSD (Warmup-Stable-Decay) Learning Rate
    # A long stable high-LR phase to thoroughly explore the AEP landscape,
    # followed by a linear decay to settle into a local optimum.
    lr_max = 1.25 * D_f
    
    # Phases:
    # 0.00 - 0.10: Warmup
    # 0.10 - 0.50: Stable (High exploration)
    # 0.50 - 0.90: Decay (Linear annealing)
    # 0.90 - 1.00: Terminal (Feasibility restoration)
    
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / 0.10)
    lr_stable = lr_max
    
    decay_progress = (progress - 0.50) / 0.40
    lr_decay = lr_max - (lr_max - gamma_min_f) * decay_progress
    
    lr = jnp.where(progress < 0.10, lr_warmup,
         jnp.where(progress < 0.50, lr_stable,
         jnp.where(progress < 0.90, lr_decay, gamma_min_f)))

    # 2. Decoupled Logistic Alpha (Penalty)
    # Alpha is held very low during the stable phase to maximize unconstrained AEP exploration,
    # then transitions smoothly via a logistic curve to a strong, bounded plateau
    # during the decay phase to progressively guide turbines into feasible arrangements.
    
    alpha_low = alpha0 * 0.1
    alpha_plateau = alpha0 * 20.0
    
    # Logistic function centered at progress = 0.55, steepness = 30
    # Ramps up exactly as the LR begins its linear decay.
    logistic_val = 1.0 / (1.0 + jnp.exp(-30.0 * (progress - 0.55)))
    
    alpha_main = alpha_low + (alpha_plateau - alpha_low) * logistic_val
    
    # Terminal feasibility spike in the last 10% for absolute constraint satisfaction
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    alpha = jnp.where(progress < 0.90, alpha_main, alpha_terminal)

    # 3. Phase-Transition Adam Moments
    # Coupled to the logistic alpha transition:
    # When alpha is low (exploration), we want fast adaptation (low beta2, high-ish beta1).
    # When alpha rises (constraint enforcement), we want to damp the stiff penalty gradients (high beta2, low beta1).
    
    b1_exploratory = 0.15
    b2_exploratory = 0.20
    
    b1_constraint = 0.02
    b2_constraint = 0.95
    
    beta1_main = b1_exploratory + (b1_constraint - b1_exploratory) * logistic_val
    beta2_main = b2_exploratory + (b2_constraint - b2_exploratory) * logistic_val
    
    beta1 = jnp.where(progress < 0.90, beta1_main, 0.01)
    beta2 = jnp.where(progress < 0.90, beta2_main, 0.99)

    return lr, alpha, beta1, beta2