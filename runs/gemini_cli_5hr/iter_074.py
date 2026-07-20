import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 074: Exponential LR decay with surgical squeeze.
    Uses exponential decay which can sometimes be more stable than cosine.
    Includes surgical squeeze for final feasibility.
    """
    t = step / (total_steps - 1)
    
    # Exponential LR decay from 1.5 to 0.005
    lr_max = 1.5 * lr0
    lr_min_target = 0.005
    lr_raw = lr_max * (lr_min_target)**t
    
    # Quadratic alpha ramp
    alpha_raw = alpha0 * (1.0 + 11999.0 * t**2)
    
    beta1 = 0.0
    beta2 = 0.995
    
    # Final surgical squeeze
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr_raw)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_raw)
    
    return lr, alpha, beta1, beta2
