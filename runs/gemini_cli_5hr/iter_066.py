import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 066: 'Shaking' LR schedule.
    Adds a small high-frequency oscillation to the decaying LR to help 
    the optimizer escape small local minima or bumps in the objective.
    """
    beta1 = 0.0
    beta2 = 0.992
    
    t = step / (total_steps - 1)
    
    # 20% constant phase
    const_phase = 0.20
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Add a 5% high-frequency 'shake'
    shake = 1.0 + 0.05 * jnp.sin(50.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Alpha schedule: quadratic ramp to 6000x
    alpha = alpha0 * (1.0 + 5999.0 * t**2)
    
    return lr, alpha, beta1, beta2
