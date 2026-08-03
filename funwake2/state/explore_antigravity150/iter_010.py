import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions for dynamic loop variables
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    p = step_f / total_f

    # Static scalars can safely be float cast
    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # 1. WSD (Warmup, Stable, Decay) Learning Rate Schedule
    # Structurally different from cyclic: we hold LR high for a long "stable" 
    # period to deeply explore layouts unconstrained, then linearly cool down.
    lr_max = 1.25 * D_f
    
    # Phase 1: Warmup (0% to 10%)
    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (p / 0.10)
    
    # Phase 2: Stable Exploration (10% to 50%)
    # Slight downward slope (15% drop) to maintain gradient flow and stability
    lr_stable = lr_max - (lr_max * 0.15) * ((p - 0.10) / 0.40)
    
    # Phase 3: Linear Decay (50% to 90%)
    p_decay = (p - 0.50) / 0.40
    lr_decay = (lr_max * 0.85) - ((lr_max * 0.85) - gamma_min_f) * p_decay
    
    # Phase 4: Terminal Fine-tuning (90% to 100%)
    lr_terminal = gamma_min_f
    
    lr = jnp.where(p < 0.10, lr_warmup,
         jnp.where(p < 0.50, lr_stable,
         jnp.where(p < 0.90, lr_decay, lr_terminal)))

    # 2. Decoupled Logistic Alpha (Delayed Ramp to Plateau)
    # Instead of cyclic spikes or inverse-lr coupling, keep penalty very low 
    # during the WSD stable phase. Then ramp it smoothly using a logistic curve 
    # to a bounded plateau strictly during the WSD decay phase.
    alpha_base = alpha0 * 0.25
    alpha_plateau = alpha0 * 25.0
    
    # Logistic ramp centered at p=0.65, steepness k=30.0
    # Creates an S-curve that mostly transitions between p=0.5 and p=0.8
    logistic_mult = 1.0 / (1.0 + jnp.exp(-30.0 * (p - 0.65)))
    alpha_main = alpha_base + (alpha_plateau - alpha_base) * logistic_mult
    
    # Terminal feasibility spike (90% to 100%) to enforce absolute constraints
    # Exact parent formulation preserved to guarantee 5/5 feasible seeds
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(p >= 0.90, alpha_terminal, alpha_main)

    # 3. Synchronized Adam Moments
    # Smoothly transition moments along the exact same logistic curve as the penalty.
    # As penalty ramps up, drop momentum (beta1) to prevent constraint overshooting,
    # and raise beta2 to dampen stiff constraint gradients.
    beta1_main = 0.15 - 0.10 * logistic_mult  # Ramps down to 0.05
    beta2_main = 0.70 + 0.25 * logistic_mult  # Ramps up to 0.95
    
    # Snap to extreme moments during the terminal spike for rigorous snapping
    beta1 = jnp.where(p >= 0.90, 0.01, beta1_main)
    beta2 = jnp.where(p >= 0.90, 0.99, beta2_main)

    return lr, alpha, beta1, beta2