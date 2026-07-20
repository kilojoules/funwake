import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 062: Warm restart LR schedule with monotonic alpha ramp.
    Two cycles of cosine decay for LR. Alpha increases quadratically.
    """
    t = step / (total_steps - 1)
    
    # Two cycles of LR decay
    # Cycle 1: 0 to 65% of total steps
    # Cycle 2: 65% to 100% of total steps
    cycle1_end = 0.65
    
    def cycle_lr(t_rel, lr_max_factor):
        return 0.005 * lr0 + (lr_max_factor * lr0 - 0.005 * lr0) * 0.5 * (1 + jnp.cos(jnp.pi * t_rel))

    lr = jnp.where(
        t < cycle1_end,
        cycle_lr(t / cycle1_end, 1.4),
        cycle_lr((t - cycle1_end) / (1.0 - cycle1_end), 0.6)
    )
    
    # Alpha schedule: quadratic ramp to 10000x
    # Stays relatively low early on to allow large moves
    alpha = alpha0 * (1.0 + 9999.0 * t**2.5)
    
    # beta1=0.0 (RMSprop style) was effective in best runs
    beta1 = 0.0
    # beta2=0.99 is slightly more responsive than standard 0.999
    beta2 = 0.992
    
    return lr, alpha, beta1, beta2
