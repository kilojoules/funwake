import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 104: Push feasibility on the 5599 GWh unfeasible best (iter_068).
    - Uses iter_068's Adam (beta1=0.0, beta2=0.992) and LR shake.
    - Much stronger alpha ramp and LR-coupling.
    """
    t = step / (total_steps - 1)
    
    # Discovery phase
    const_phase = 0.2
    lr_start = 1.3 * lr0
    lr_min = 0.001 * lr0
    
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Shake
    shake_amp = 0.06 * jnp.sqrt(1.0 - t)
    shake = 1.0 + shake_amp * jnp.sin(100.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Super strong coupled alpha
    alpha_base = alpha0 * (1.0 + 200000.0 * t**3)
    alpha = alpha_base * (lr_start / jnp.maximum(lr, 1e-10))
    
    # Squeeze
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 10000000.0 * alpha0, alpha)

    # No momentum Adam
    beta1 = 0.0
    beta2 = 0.992
    
    return lr, alpha, beta1, beta2
