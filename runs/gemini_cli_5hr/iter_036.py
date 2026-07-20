import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_015 with delayed alpha ramp.
    """
    beta1 = 0.0
    beta2 = 0.999
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # 20% constant phase for LR
    const_phase = 0.2
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Alpha stages:
    # 0 to 0.4: alpha = alpha0 (exploration)
    # 0.4 to 1.0: quadratic ramp to 2000 * alpha0 (feasibility)
    alpha = jnp.where(
        t < 0.4,
        alpha0,
        alpha0 * (1.0 + 1999.0 * ((t - 0.4) / 0.6)**2)
    )
    
    return lr, alpha, beta1, beta2
