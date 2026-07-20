import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 061: Power law alpha ramp and cosine LR decay.
    Tries to balance early exploration with late-stage constraint enforcement.
    """
    t = step / (total_steps - 1)
    
    # LR: Initial boost, then cosine decay with a slight stretch
    lr_peak = 1.5 * lr0
    lr_min = 0.005 * lr0
    # Use t**1.1 to stay at higher LR slightly longer
    lr = lr_min + (lr_peak - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t**1.1))
    
    # Alpha: Power law ramp (cubic)
    # Starts at 0.5 * alpha0 to allow early movement, ends at 10000 * alpha0
    alpha = alpha0 * (0.5 + 9999.5 * t**3)
    
    # Momentum: Smooth transition
    # Lower beta1 (0.1) initially for responsiveness, higher (0.4) at end for stability
    beta1 = 0.1 * (1 - t) + 0.4 * t
    # beta2 starts at 0.9 and goes to 0.999
    beta2 = 0.9 * (1 - t) + 0.999 * t
    
    return lr, alpha, beta1, beta2
