import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Robust schedule: cosine LR from start, 2000x alpha ramp.
    """
    # Use standard Adam beta2 for better second moment estimation
    beta1 = 0.1
    beta2 = 0.999
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # Learning rate schedule: smooth cosine decay from lr0 to 0.01*lr0
    lr = 0.01 * lr0 + 0.99 * lr0 * 0.5 * (1 + jnp.cos(jnp.pi * t))
    
    # Alpha schedule: quadratic ramp to 2000x to ensure feasibility
    alpha = alpha0 * (1.0 + 1999.0 * t**2)
    
    return lr, alpha, beta1, beta2
