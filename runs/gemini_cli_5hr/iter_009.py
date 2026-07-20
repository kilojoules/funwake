import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Refined iter_007 with longer const phase and cubic alpha.
    """
    beta1 = 0.1
    beta2 = 0.999
    
    # Progress within the whole run
    t = step / (total_steps - 1)
    
    # LR schedule: Constant for 1/4, then cosine decay
    const_phase = total_steps // 4
    is_decay = step >= const_phase
    progress = jnp.clip((step - const_phase) / (total_steps - const_phase), 0.0, 1.0)
    
    lr_start = lr0
    lr_min = 0.01 * lr0
    
    lr_decay = lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(is_decay, lr_decay, lr_start)
    
    # Alpha schedule: cubic ramp up to 1000x
    alpha = alpha0 * (1.0 + 999.0 * t**3)
    
    return lr, alpha, beta1, beta2
