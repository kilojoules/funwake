import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_015 with exponential LR decay.
    """
    beta1 = 0.0
    beta2 = 0.999
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # 20% constant phase
    const_phase = 0.2
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    is_decay = t >= const_phase
    
    # Exponential decay parameter
    k = -jnp.log(lr_min / lr_start) / (1.0 - const_phase)
    lr_decay = lr_start * jnp.exp(-k * (t - const_phase))
    
    lr = jnp.where(is_decay, lr_decay, lr_start)
    
    # Alpha schedule: quadratic ramp to 2000x
    alpha = alpha0 * (1.0 + 1999.0 * t**2)
    
    return lr, alpha, beta1, beta2
