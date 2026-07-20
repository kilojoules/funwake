import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_015 with decaying beta1 (0.1 -> 0.0).
    """
    t = step / (total_steps - 1)
    
    # Decaying momentum
    beta1 = 0.1 * (1.0 - t)
    beta2 = 0.999
    
    # Progress from 0 to 1
    # Learning rate schedule: 1.2x initial LR
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr = lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t))
    
    # Alpha schedule: quadratic ramp to 2000x
    alpha = alpha0 * (1.0 + 1999.0 * t**2)
    
    return lr, alpha, beta1, beta2
