import jax.numpy as jnp

_C = 250.0 / 240.0  # exploration scaling factor

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions (step and total_steps may be traced)
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    lr_max = _C * float(D)
    gamma_min_f = float(gamma_min)

    # 1. Warmup-Stable-Decay (WSD) Learning Rate
    # A long plateau at lr_max enables robust broad exploration before localized refinement.
    # Warmup: 0% to 5% (starts at 10% of lr_max)
    lr_warmup = lr_max * (0.1 + 0.9 * jnp.clip(progress / 0.05, 0.0, 1.0))
    # Decay: 50% to 90% (linear cooldown to gamma_min)
    lr_decay = lr_max - (lr_max - gamma_min_f) * jnp.clip((progress - 0.50) / 0.40, 0.0, 1.0)
    
    lr_main = jnp.where(progress < 0.05, lr_warmup,
                jnp.where(progress < 0.50, lr_max, lr_decay))
                
    is_terminal = progress >= 0.90
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # 2. Delayed Logistic Alpha Plateau + Terminal Spike
    # Bounded plateau rather than diverging 1/lr. Delay the penalty ramp to mid-run (centered at 60%).
    # This fully decouples the penalty from the learning rate and allows unimpeded early exploration.
    k = 20.0
    x0 = 0.60
    logistic_ramp = 1.0 / (1.0 + jnp.exp(-k * (progress - x0)))
    
    alpha_plateau = alpha0 * 10.0
    alpha_main = alpha0 + (alpha_plateau - alpha0) * logistic_ramp
    
    # Terminal feasibility spike to forcibly resolve any remaining violation tolerance
    alpha_terminal = alpha0 * float(D) / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # 3. Phase-transition Adam moments
    # As the penalty (alpha) smoothly ramps up, we drop momentum (beta1) to prevent constraint overshoot
    # and aggressively increase curvature absorption (beta2) to handle the stiff penalty gradient.
    beta1_main = 0.1 - 0.05 * logistic_ramp
    beta2_main = 0.2 + 0.7 * logistic_ramp
    
    beta1 = jnp.where(is_terminal, 0.05, beta1_main)
    beta2 = jnp.where(is_terminal, 0.90, beta2_main)

    return lr, alpha, beta1, beta2