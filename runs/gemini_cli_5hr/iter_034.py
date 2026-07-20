import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Cosine annealing with 2 cycles.
    """
    beta1 = 0.0
    beta2 = 0.999
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # 2 cycles
    num_cycles = 2
    cycle_progress = (t * num_cycles) % 1.0
    
    # Decaying peak envelope
    peak_envelope = 1.2 - 0.6 * t
    
    lr_peak = peak_envelope * lr0
    lr_min = 0.005 * lr0
    
    lr = lr_min + (lr_peak - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * cycle_progress))
    
    # Alpha schedule: quadratic ramp to 2000x
    alpha = alpha0 * (1.0 + 1999.0 * t**2)
    
    return lr, alpha, beta1, beta2
