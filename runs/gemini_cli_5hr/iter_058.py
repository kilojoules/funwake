import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iter_055 with sigmoid alpha ramp.
    """
    beta1 = 0.0
    beta2 = 0.992
    
    # Progress from 0 to 1
    t = step / (total_steps - 1)
    
    # 20% constant phase
    const_phase = 0.2
    lr_start = 1.2 * lr0
    lr_min = 0.01 * lr0
    
    lr = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Sigmoid ramp for alpha
    k = 15.0
    t0 = 0.6
    sigmoid = 1.0 / (1.0 + jnp.exp(-k * (t - t0)))
    # Ensure it starts near 1.0 at t=0
    sigmoid_t0 = 1.0 / (1.0 + jnp.exp(k * t0))
    alpha = alpha0 * (1.0 + 4999.0 * (sigmoid - sigmoid_t0) / (1.0 - sigmoid_t0))
    
    return lr, alpha, beta1, beta2
