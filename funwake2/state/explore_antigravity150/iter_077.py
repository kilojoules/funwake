import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Safely cast all inputs to jnp.float32 to prevent JAX tracing errors
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)
    
    progress = step_f / total_f
    
    # --- 1. Multi-Cycle SGDR (Cosine Annealing with Warm Restarts) ---
    # Structural Shift: Break the optimization into 3 distinct cycles.
    # This prevents the optimizer from getting stuck in an early local basin by 
    # periodically spiking the learning rate to allow layout reshuffling.
    c1, c2, c3 = 0.40, 0.75, 0.92
    
    # Normalize progress within the current cycle to [0, 1]
    cp = jnp.where(progress < c1, progress / c1,
         jnp.where(progress < c2, (progress - c1) / (c2 - c1),
         jnp.where(progress < c3, (progress - c2) / (c3 - c2),
         1.0)))
         
    # Cycle boundaries for Learning Rate (peaks decay with each restart)
    lr_max_curr = jnp.where(progress < c1, 1.25 * D_f,
                  jnp.where(progress < c2, 0.70 * D_f,
                  0.30 * D_f))
                  
    lr_min_curr = jnp.where(progress < c1, 0.25 * D_f,
                  jnp.where(progress < c2, 0.10 * D_f,
                  gamma_min_f))
                  
    # Cosine decay within the cycle
    cycle_lr = lr_min_curr + 0.5 * (lr_max_curr - lr_min_curr) * (1.0 + jnp.cos(jnp.pi * cp))
    
    # Quick linear warmup at the very beginning to prevent explosive early gradients
    warmup_prog = jnp.clip(progress / 0.03, 0.0, 1.0)
    lr_main = cycle_lr * warmup_prog
    
    # --- 2. Cyclical Penalty (Synchronized with SGDR) ---
    # When LR warm-restarts (spikes), we drop the penalty so the layout can shift across boundaries.
    # As the cycle cools, we ramp the penalty up to force a feasible local minimum.
    # The penalty floor and ceiling progressively rise to guarantee convergence.
    alpha_base_curr = jnp.where(progress < c1, alpha0_f * 0.05,
                      jnp.where(progress < c2, alpha0_f * 0.40,
                      alpha0_f * 1.50))
                      
    alpha_peak_curr = jnp.where(progress < c1, alpha0_f * 3.0,
                      jnp.where(progress < c2, alpha0_f * 10.0,
                      alpha0_f * 25.0))
                      
    # Smooth ramp from base to peak using standard 1-cosine formulation
    alpha_ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * cp))
    alpha_main = alpha_base_curr + (alpha_peak_curr - alpha_base_curr) * alpha_ramp
    
    # --- 3. Phase-Transition Adam Moments ---
    # High momentum / low beta2 during exploration (start of cycle).
    # Low momentum / high beta2 during constraint enforcement (end of cycle).
    b1_start, b1_end = 0.15, 0.02
    b2_start, b2_end = 0.15, 0.90
    
    beta1_main = b1_start + (b1_end - b1_start) * alpha_ramp
    beta2_main = b2_start + (b2_end - b2_start) * alpha_ramp
    
    # --- 4. Terminal Feasibility Spike ---
    # Final phase ensures absolute compliance (filter method)
    is_terminal = progress >= c3
    
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)
    
    return lr, alpha, beta1, beta2