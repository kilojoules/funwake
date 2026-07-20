import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_007 with higher initial LR.
    """
    beta1 = 0.1
    beta2 = 0.999
    
    # Higher initial LR to explore more
    lr_mult = 1.2
    
    # iter_007 schedule
    const_phase = total_steps // 5
    is_decay = step >= const_phase
    progress = jnp.clip((step - const_phase) / (total_steps - const_phase), 0.0, 1.0)
    
    lr_start = lr_mult * lr0
    lr_min = 0.01 * lr0
    
    lr_decay = lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(is_decay, lr_decay, lr_start)
    
    # alpha ramp
    alpha = alpha0 * (1.0 + 499.0 * (step / total_steps)**2)
    
    return lr, alpha, beta1, beta2
