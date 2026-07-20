import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 067: Decaying shake LR schedule.
    High-frequency shake amplitude decays as optimization progresses, 
    allowing for exploration early and precision refining late.
    """
    t = step / (total_steps - 1)
    beta1 = 0.0
    beta2 = 0.992
    
    # Discovery phase
    const_phase = 0.25
    lr_start = 1.3 * lr0
    lr_min = 0.005 * lr0
    
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Decaying shake amplitude (sqrt decay for more shake in middle phase)
    shake_amp = 0.08 * jnp.sqrt(1.0 - t)
    shake = 1.0 + shake_amp * jnp.sin(80.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Quadratic alpha ramp to 7500x
    alpha = alpha0 * (1.0 + 7499.0 * t**2)
    
    return lr, alpha, beta1, beta2
