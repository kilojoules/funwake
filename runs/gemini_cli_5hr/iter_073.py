import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 073: TopFarm-style ultra-responsive momentum.
    Combines the discovery schedule of Iter_070 with TopFarm's 
    low beta1/beta2 parameters for faster local adaptation.
    """
    t = step / (total_steps - 1)
    
    # TopFarm-style low-beta momentum
    beta1 = 0.1
    beta2 = 0.2
    
    # Discovery phase
    const_phase = 0.25
    lr_start = 1.3 * lr0
    lr_min = 0.005 * lr0
    
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Decaying shake
    shake_amp = 0.08 * jnp.sqrt(1.0 - t)
    shake = 1.0 + shake_amp * jnp.sin(80.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Alpha schedule
    alpha_main = alpha0 * (1.0 + 7499.0 * t**2)
    
    # Final surgical squeeze
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 1000000.0 * alpha0, alpha_main)
    
    return lr, alpha, beta1, beta2
