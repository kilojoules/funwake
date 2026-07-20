import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Cosine LR, higher alpha ramp, slightly more momentum.
    """
    beta1 = 0.2
    beta2 = 0.4
    
    # Learning rate: Constant for 15%, then cosine decay to 0.005
    const_phase = total_steps // 6
    is_decay = step >= const_phase
    
    progress = jnp.clip((step - const_phase) / (total_steps - const_phase), 0.0, 1.0)
    lr_decay = 0.005 * lr0 + 0.995 * lr0 * 0.5 * (1 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(is_decay, lr_decay, lr0)
    
    # Alpha schedule: quadratic ramp to 2000 * alpha0
    alpha = alpha0 * (1.0 + 1999.0 * (step / total_steps)**2)
    
    return lr, alpha, beta1, beta2
