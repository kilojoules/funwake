import jax.numpy as jnp

_C = 250.0 / 240.0  # Slightly higher than parent's 200/240 for a stronger early peak

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    lr0 = _C * float(D)
    gamma_min_f = float(gamma_min)
    
    # 1. WSD Learning Rate Schedule (Warmup, Stable, Decay, Terminal)
    # 0.0 - 0.10: Linear warmup to lr0
    # 0.10 - 0.50: Stable at lr0
    # 0.50 - 0.90: Linear decay to gamma_min
    # 0.90 - 1.00: Terminal phase at gamma_min
    lr_warmup = lr0 * 0.1 + (lr0 - lr0 * 0.1) * (progress / 0.10)
    lr_decay = lr0 + (gamma_min_f - lr0) * ((progress - 0.50) / 0.40)
    
    lr = jnp.where(progress < 0.10, lr_warmup,
         jnp.where(progress < 0.50, lr0,
         jnp.where(progress < 0.90, lr_decay, gamma_min_f)))

    # 2. Delayed Alpha Ramp + Bounded Plateau + Terminal Spike
    alpha_exploration = alpha0 * 1.0
    alpha_plateau = alpha0 * 5.0
    
    # Ramps from exploration to plateau between 50% and 70% of steps
    ramp_factor = jnp.clip((progress - 0.50) / 0.20, 0.0, 1.0)
    alpha_decoupled = alpha_exploration + (alpha_plateau - alpha_exploration) * ramp_factor
    
    # Terminal feasibility spike (matches native coupling at gamma_min)
    alpha_terminal = alpha0 * float(D) / jnp.maximum(gamma_min_f, 1e-30)
    
    alpha = jnp.where(progress >= 0.90, alpha_terminal, alpha_decoupled)

    # 3. Phase-transition Adam moments
    # Late phase: beta1 drops (less momentum into walls), beta2 increases (absorb curvature)
    beta1 = jnp.where(progress >= 0.90, 0.05, 0.1)
    beta2 = jnp.where(progress >= 0.90, 0.9, 0.2)

    return lr, alpha, beta1, beta2