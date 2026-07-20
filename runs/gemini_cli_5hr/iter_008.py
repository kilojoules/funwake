import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_007 with moderate beta1.
    """
    beta1 = 0.5
    beta2 = 0.999
    
    # Same schedule as iter_007
    const_phase = total_steps // 5
    is_decay = step >= const_phase
    progress = jnp.clip((step - const_phase) / (total_steps - const_phase), 0.0, 1.0)
    lr_decay = 0.01 * lr0 + 0.99 * lr0 * 0.5 * (1 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(is_decay, lr_decay, lr0)
    
    # alpha ramp
    alpha = alpha0 * (1.0 + 499.0 * (step / total_steps)**2)
    
    return lr, alpha, beta1, beta2
