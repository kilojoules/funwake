import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 068: Aggressive decaying shake.
    Based on Iter_067 which reached 5599 GWh but was infeasible.
    Dramatically increases alpha and lowers final LR to settle into a feasible state.
    """
    t = step / (total_steps - 1)
    beta1 = 0.0
    beta2 = 0.992
    
    # Discovery phase
    const_phase = 0.25
    lr_start = 1.3 * lr0
    lr_min = 0.001 * lr0 # Lowered to 0.001 for precision settlement
    
    lr_base = jnp.where(
        t < const_phase,
        lr_start,
        lr_min + (lr_start - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * (t - const_phase) / (1.0 - const_phase)))
    )
    
    # Slightly less amplitude to avoid bouncing out of feasible zones at the end
    shake_amp = 0.06 * jnp.sqrt(1.0 - t)
    shake = 1.0 + shake_amp * jnp.sin(100.0 * jnp.pi * t)
    lr = lr_base * shake
    
    # Highly aggressive alpha ramp to 20000x
    # Use power 3 to keep alpha low longer but end VERY high
    alpha = alpha0 * (1.0 + 19999.0 * t**3)
    
    return lr, alpha, beta1, beta2
