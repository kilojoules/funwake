import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_041 with initial zero alpha.
    """
    beta1 = 0.0
    beta2 = 0.99
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # 20% constant phase
    const_phase = 0.2
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Alpha schedule: zero for 5%, then quadratic ramp to 5000x
    alpha = jnp.where(
        t < 0.05,
        0.0,
        alpha0 * (1.0 + 4999.0 * ((t - 0.05) / 0.95)**2)
    )
    
    return lr, alpha, beta1, beta2
