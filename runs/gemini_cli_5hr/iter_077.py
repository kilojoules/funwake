import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 077: 5-cycle restart with gentler peak decay.
    Maintains higher learning rates longer to avoid getting trapped,
    while using a surgical squeeze to ensure feasibility.
    """
    t = step / (total_steps - 1)
    
    # 5 Cycles
    num_cycles = 5
    cycle_idx = jnp.minimum((t * num_cycles).astype(int), num_cycles - 1)
    t_cycle = (t * num_cycles) % 1.0
    
    # Gentler peak decay for more exploration in middle cycles
    peaks = jnp.array([1.5, 1.2, 0.9, 0.6, 0.3])
    lr_peak = peaks[cycle_idx]
    
    # LR within each cycle
    lr_min = 0.005 * lr0
    lr_in_cycle = lr_min + (lr_peak * lr0 - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t_cycle))
    
    # Quadratic alpha ramp
    alpha_raw = alpha0 * (1.0 + 14999.0 * t**2)
    
    # Responsive momentum (low beta1, high-ish beta2)
    beta1 = 0.0
    beta2 = 0.99
    
    # Final surgical squeeze (99-100%)
    is_squeeze = (t > 0.99)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr_in_cycle)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_raw)
    
    return lr, alpha, beta1, beta2
