import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- Two-Stage WSD with Mid-Run Feasibility Burst ---
    # Tests the hypothesis that interspersing a short, violent feasibility 
    # restoration phase (a "Burst") allows for more aggressive, decoupled AEP 
    # exploration both before and after, avoiding continuous coupled alpha ramps.
    # It mimics an Augmented Lagrangian Method (ALM) with piecewise-constant penalty.

    t_exp_end = 0.35      # End of free exploration
    t_burst_end = 0.45    # End of mid-run feasibility burst
    t_refine_end = 0.90   # End of AEP refinement
    
    # 1. Learning Rate Schedule
    # Exploration: Hold high (1.5 D) for WSD warmup/stable, then linear decay to 0.4 D
    exp_progress = jnp.clip(progress / t_exp_end, 0.0, 1.0)
    lr_exp = jnp.where(exp_progress < 0.70, 1.50 * D_f, 
                       1.50 * D_f - (1.10 * D_f) * ((exp_progress - 0.70) / 0.30))
                       
    # Burst: Drop LR sharply. Focus strictly on untangling collisions locally.
    lr_burst = 0.15 * D_f
    
    # Refinement: SGDR warm restart, cosine decay down to gamma_min
    refine_progress = jnp.clip((progress - t_burst_end) / (t_refine_end - t_burst_end), 0.0, 1.0)
    lr_refine = gamma_min_f + (0.70 * D_f - gamma_min_f) * 0.5 * (1.0 + jnp.cos(jnp.pi * refine_progress))
    
    lr_main = jnp.where(progress < t_exp_end, lr_exp,
              jnp.where(progress < t_burst_end, lr_burst, lr_refine))

    # 2. Decoupled Penalty (Alpha)
    # Piecewise-constant phases mirroring ALM outer loop iterations
    alpha_exp = alpha0_f * 0.1    # Very soft penalty, allow global layout shift
    alpha_burst = alpha0_f * 50.0 # Violent spike to aggressively untangle layout
    alpha_refine = alpha0_f * 4.0 # Moderate constant ALM plateau for local AEP refinement
    
    alpha_main = jnp.where(progress < t_exp_end, alpha_exp,
                 jnp.where(progress < t_burst_end, alpha_burst, alpha_refine))

    # 3. Phase-Synchronized Adam Moments
    # Momentum drops (beta1) and variance absorption rises (beta2) during high-penalty phases
    b1_exp, b2_exp = 0.15, 0.10
    b1_burst, b2_burst = 0.01, 0.95
    b1_refine, b2_refine = 0.08, 0.80
    
    beta1_main = jnp.where(progress < t_exp_end, b1_exp,
                 jnp.where(progress < t_burst_end, b1_burst, b1_refine))
    beta2_main = jnp.where(progress < t_exp_end, b2_exp,
                 jnp.where(progress < t_burst_end, b2_burst, b2_refine))
                 
    # 4. Terminal Feasibility Restoration Spike
    # Guarantees absolute constraint satisfaction at the end of the run
    is_terminal = progress >= t_refine_end
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    beta1_terminal = 0.00
    beta2_terminal = 0.99
    
    lr = jnp.where(is_terminal, lr_terminal, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_main)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_main)
    
    return lr, alpha, beta1, beta2