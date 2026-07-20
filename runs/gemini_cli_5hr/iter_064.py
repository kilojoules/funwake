import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 064: Variation of Iter_055 with slightly higher final alpha 
    and a small momentum (beta1=0.1) for better hill-climbing.
    """
    # Small momentum to help smooth out gradients
    beta1 = 0.1
    beta2 = 0.992
    
    t = step / (total_steps - 1)
    
    # 22% constant phase
    const_phase = 0.22
    lr_start = 1.25 * lr0
    lr_min = 0.005 * lr0
    
    lr = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Alpha: quadratic ramp to 7000x
    alpha = alpha0 * (1.0 + 6999.0 * t**2)
    
    return lr, alpha, beta1, beta2
