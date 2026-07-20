import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Refined iter_002 with slightly higher LR and more aggressive alpha.
    """
    # TopFarm-style beta values
    beta1 = 0.1
    beta2 = 0.2
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # Learning rate schedule
    const_phase = total_steps // 3
    is_decay = step >= const_phase
    
    progress = jnp.clip((step - const_phase) / (total_steps - const_phase), 0.0, 1.0)
    
    # Slightly higher starting point
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr_decay = lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * progress))
    lr = jnp.where(is_decay, lr_decay, lr_start)
    
    # Alpha schedule: ramp up to enforce constraints
    # Reaching 1000x alpha0 at the end
    alpha = alpha0 * (1.0 + 999.0 * t**2)
    
    return lr, alpha, beta1, beta2
