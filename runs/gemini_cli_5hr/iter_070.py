import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 070: Match Iter_067 discovery with a very late squeeze.
    Uses exactly the same schedule as Iter_067 but adds a surgical squeeze 
    in the final 2% of iterations to resolve small feasibility issues.
    """
    t = step / (total_steps - 1)
    beta1 = 0.0
    beta2 = 0.992
    
    # Phase 1: Main Optimization (exactly like Iter_067)
    const_phase = 0.25
    lr_start = 1.3 * lr0
    lr_min = 0.005 * lr0
    
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    shake_amp = 0.08 * jnp.sqrt(1.0 - t)
    shake = 1.0 + shake_amp * jnp.sin(80.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Alpha (exactly as Iter_067)
    alpha_main = alpha0 * (1.0 + 7499.0 * t**2)
    
    # Phase 2: Surgical Squeeze (98-100%)
    # Move turbines very slowly (0.5cm steps) with massive penalty
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_main)
    
    return lr, alpha, beta1, beta2
