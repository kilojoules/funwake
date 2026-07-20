import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_015 with cubic alpha ramp.
    """
    beta1 = 0.0
    beta2 = 0.999
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # Learning rate schedule: 1.2x initial LR
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr = lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t))
    
    # Alpha schedule: cubic ramp to 2000x
    alpha = alpha0 * (1.0 + 1999.0 * t**3)
    
    return lr, alpha, beta1, beta2
