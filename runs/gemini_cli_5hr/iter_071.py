import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 071: High-exploration discovery with power-4 alpha ramp.
    Allows turbines more freedom to rearrange by keeping alpha low for longer,
    while using a surgical squeeze to fix the resulting constraint violations.
    """
    t = step / (total_steps - 1)
    beta1 = 0.0
    beta2 = 0.992
    
    # Discovery phase (0-30%)
    const_phase = 0.30
    lr_start = 1.4 * lr0
    lr_min = 0.005 * lr0
    
    # Main decay phase
    decay_progress = jnp.clip((t - const_phase) / (0.95 - const_phase), 0.0, 1.0)
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
    )
    
    # Shake until 85%
    shake_amp = 0.10 * jnp.sqrt(1.0 - t)
    shake = 1.0 + shake_amp * jnp.sin(120.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Alpha stays very low early on
    alpha_main = alpha0 * (1.0 + 11999.0 * t**4)
    
    # Final SQUEEZE (95-100%)
    is_squeeze = (t > 0.95)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_main)
    
    return lr, alpha, beta1, beta2
