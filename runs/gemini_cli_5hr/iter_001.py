import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Cosine annealing with warmup and coupled alpha.
    """
    # Standard Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    
    # Linear warm-up for 5% of steps
    warmup_steps = total_steps // 20
    
    # Learning rate schedule
    lr_max = lr0
    lr_min = 0.01 * lr0
    
    is_warmup = step < warmup_steps
    lr_warmup = lr_max * (step / warmup_steps)
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr_decay = lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * progress))
    
    lr = jnp.where(is_warmup, lr_warmup, lr_decay)
    
    # Alpha schedule: increase as lr decreases
    # Coupled to lr to ensure constraints are met as steps get smaller
    alpha = alpha0 * jnp.clip(lr0 / jnp.maximum(lr, 1e-10), 1.0, 200.0)
    
    return lr, alpha, beta1, beta2
