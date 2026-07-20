import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 069: Discovery with final 'squeeze' phase.
    Uses Iter_067 discovery but adds a high-penalty, low-step-size 
    squeeze in the final 5% of iterations to ensure feasibility.
    """
    t = step / (total_steps - 1)
    beta1 = 0.0
    beta2 = 0.992
    
    # Main optimization phase (0-90%)
    # Discovery phase (0-25%)
    const_phase = 0.25
    lr_start = 1.3 * lr0
    lr_min = 0.005 * lr0
    
    # Decay lr from const_phase to 90%
    decay_progress = jnp.clip((t - const_phase) / (0.90 - const_phase), 0.0, 1.0)
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * decay_progress))
    )
    
    # Shake only until 80% to allow settling before squeeze
    shake_amp = 0.06 * jnp.maximum(0.80 - t, 0.0) / 0.80
    shake = 1.0 + shake_amp * jnp.sin(100.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Alpha ramp
    alpha_main = alpha0 * (1.0 + 9999.0 * t**2)
    
    # SQUEEZE PHASE (90-100%)
    # Very high alpha, very low LR to move into feasible zone
    is_squeeze = (t > 0.90)
    lr = jnp.where(is_squeeze, 0.0002 * lr0, lr)
    alpha = jnp.where(is_squeeze, 200000.0 * alpha0, alpha_main)
    
    return lr, alpha, beta1, beta2
