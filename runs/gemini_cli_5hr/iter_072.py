import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 072: Variable momentum and three-cycle LR.
    Transitions from reactive discovery (low beta2) to stable refinement (high beta2).
    Includes a three-cycle restart schedule and final surgical squeeze.
    """
    t = step / (total_steps - 1)
    
    # Momentum transition: starts low for responsiveness, ends high for stability
    beta1 = 0.0
    beta2 = 0.985 * (1.0 - t) + 0.999 * t
    
    # Three-cycle LR decay with reducing peak height
    def get_lr(t_rel, peak_factor):
        return 0.005 * lr0 + (peak_factor * lr0 - 0.005 * lr0) * 0.5 * (1 + jnp.cos(jnp.pi * t_rel))
    
    lr_raw = jnp.where(
        t < 0.40,
        get_lr(t / 0.40, 1.5),
        jnp.where(
            t < 0.75,
            get_lr((t - 0.40) / 0.35, 0.8),
            get_lr((t - 0.75) / 0.25, 0.4)
        )
    )
    
    # Alpha: smooth quadratic ramp
    alpha_raw = alpha0 * (1.0 + 11999.0 * t**2)
    
    # Final SQUEEZE (98-100%)
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr_raw)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_raw)
    
    return lr, alpha, beta1, beta2
