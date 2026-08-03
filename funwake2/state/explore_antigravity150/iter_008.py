import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    # Ensure traceable conversions
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    # Base exploratory peak proportional to D
    lr_max = 1.20 * float(D)
    gamma_min_f = float(gamma_min)

    # 1. SGDR: Cosine Annealing with Warm Restarts and Initial Warmup
    # We use two cycles to allow multiple phases of exploration and settling.
    is_warmup = progress < 0.05
    is_cycle1 = (progress >= 0.05) & (progress < 0.45)
    is_cycle2 = (progress >= 0.45) & (progress < 0.90)
    is_terminal = progress >= 0.90

    lr_warmup = gamma_min_f + (lr_max - gamma_min_f) * (progress / 0.05)

    def cosine_anneal(p, p_start, p_end, lr_start, lr_end):
        p_norm = (p - p_start) / (p_end - p_start)
        return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + jnp.cos(jnp.pi * p_norm))

    # Cycle 1: Anneal from lr_max down to a moderate learning rate
    lr_cycle1 = cosine_anneal(progress, 0.05, 0.45, lr_max, gamma_min_f * 5.0)
    
    # Cycle 2: Warm restart at a slightly lower peak, anneal fully to gamma_min
    lr_max_restart = lr_max * 0.60
    lr_cycle2 = cosine_anneal(progress, 0.45, 0.90, lr_max_restart, gamma_min_f)
    
    lr = jnp.where(is_warmup, lr_warmup,
         jnp.where(is_cycle1, lr_cycle1,
         jnp.where(is_cycle2, lr_cycle2, gamma_min_f)))

    # 2. Cyclic Alpha / Mid-run Feasibility Bursts
    # Decouple penalty from 1/lr. When LR is high, alpha is low to allow free
    # layout shuffling. As each cycle cools down, alpha bursts to guide
    # the turbines into feasible arrangements.
    
    alpha_base = alpha0 * 0.5
    
    # Burst 1: smoothly ramps up at the end of Cycle 1
    burst1_norm = jnp.clip((progress - 0.25) / 0.20, 0.0, 1.0)**2
    alpha_burst1 = alpha_base + alpha0 * 15.0 * burst1_norm
    
    # Burst 2: smoothly ramps up at the end of Cycle 2
    burst2_norm = jnp.clip((progress - 0.70) / 0.20, 0.0, 1.0)**2
    alpha_burst2 = alpha_base + alpha0 * 30.0 * burst2_norm
    
    alpha_main = jnp.where(progress < 0.45, alpha_burst1, alpha_burst2)
    
    # Terminal feasibility spike to enforce absolute constraints at the end
    alpha_terminal = alpha0 * float(D) / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # 3. Cyclic Adam Moments
    # Synchronize Adam moments with the alpha bursts: drop momentum (beta1) to 
    # prevent constraint overshooting, and raise curvature absorption (beta2) 
    # to dampen the stiff penalty gradients during the bursts.
    
    burst_norm = jnp.where(progress < 0.45, burst1_norm, burst2_norm)
    
    beta1_main = 0.10 - 0.05 * burst_norm    # Drops to 0.05 during bursts
    beta2_main = 0.70 + 0.25 * burst_norm    # Rises to 0.95 during bursts
    
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2