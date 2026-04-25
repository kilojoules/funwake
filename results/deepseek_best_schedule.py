import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    # Constant phase for the first half of the steps
    const_phase = total_steps // 2
    decay_steps = total_steps - const_phase
    
    # Determine if we are in the decay phase
    decaying = step >= const_phase
    decay_step = jnp.maximum(step - const_phase, 0.0)
    
    # Learning rate decays smoothly using cosine schedule
    progress = decay_step / decay_steps
    lr = jnp.where(decaying, lr0 * 0.1 * (1 + jnp.cos(jnp.pi * progress)), lr0)
    lr = jnp.maximum(lr, 0.001 * lr0)  # Ensure minimum learning rate
    
    # Penalty multiplier increases more aggressively as learning rate decays
    alpha = jnp.where(decaying,
                      alpha0 * 100 * (lr0 / lr),
                      alpha0)
    
    # Adam parameters for better convergence
    beta1 = 0.9
    beta2 = 0.999
    
    return lr, alpha, beta1, beta2