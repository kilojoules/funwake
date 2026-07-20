import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Robust schedule with 10% constant phase and 1.2x lr0.
    """
    beta1 = 0.1
    beta2 = 0.999
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # 10% constant phase for LR
    const_phase = 0.1
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Alpha schedule: quadratic ramp to 2000x
    alpha = alpha0 * (1.0 + 1999.0 * t**2)
    
    return lr, alpha, beta1, beta2
