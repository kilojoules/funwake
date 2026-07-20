import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 063: Coupled alpha-LR schedule with an additional quadratic ramp.
    Tries to maintain the baseline's 'constant constraint step' logic
    while making it more aggressive over time to ensure feasibility.
    """
    beta1 = 0.0
    beta2 = 0.992
    
    t = step / (total_steps - 1)
    
    # 25% constant phase for layout discovery
    const_phase = 0.25
    lr_start = 1.4 * lr0
    lr_min = 0.002 * lr0
    
    lr = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Coupled alpha: alpha * lr = alpha0 * lr0 * (1 + ramp)
    # The baseline has alpha * lr = alpha0 * lr0
    # Here we increase the relative importance of constraints as we converge.
    ramp = 80.0 * t**2
    alpha = (alpha0 * lr0 / jnp.maximum(lr, 1e-10)) * (1.0 + ramp)
    
    return lr, alpha, beta1, beta2
