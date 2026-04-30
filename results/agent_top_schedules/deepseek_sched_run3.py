import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    progress = step / total_steps
    
    # Exponential decay: lr stays high for 60% then drops rapidly
    lr = lr0 * jnp.exp(-8.0 * jnp.maximum(progress - 0.4, 0.0) ** 2.0)
    lr = jnp.maximum(lr, lr0 * 0.0003)
    
    # Alpha starts ramping at 40% progress, quadratically
    late = jnp.maximum(progress - 0.4, 0.0) / 0.6
    alpha = alpha0 * (1.0 + 3999.0 * late ** 2.0)
    
    beta1 = 0.9
    beta2 = 0.999
    return lr, alpha, beta1, beta2