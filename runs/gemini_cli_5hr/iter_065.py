import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 065: Cyclic alpha schedule. 
    High initial alpha to push away overlaps, then lower to explore AEP, 
    then very high to lock in feasibility.
    """
    t = step / (total_steps - 1)
    
    # LR schedule: Cosine decay from 1.3 to 0.01
    lr_start = 1.3 * lr0
    lr_min = 0.005 * lr0
    lr = lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t))
    
    # Alpha schedule:
    # Phase 1: (0% to 15%) Squeeze out initial overlaps
    # Phase 2: (15% to 40%) Relax to let the layout rearrange
    # Phase 3: (40% to 100%) Ramp up to final high penalty
    
    def alpha_schedule(t):
        alpha_initial = 1000.0 * alpha0
        alpha_low = 50.0 * alpha0
        alpha_final = 8000.0 * alpha0
        
        return jnp.where(
            t < 0.15,
            alpha_initial * (1.0 - t/0.15)**2 + alpha_low,
            jnp.where(
                t < 0.40,
                alpha_low,
                alpha_low + (alpha_final - alpha_low) * ((t - 0.40) / 0.60)**2
            )
        )
    
    alpha = alpha_schedule(t)
    
    beta1 = 0.0
    beta2 = 0.992
    
    return lr, alpha, beta1, beta2
