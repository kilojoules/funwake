import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    a0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- 1. Two-Epoch Cascade Learning Rate ---
    # We split the optimization into two distinct phases (epochs) before the terminal phase.
    # Epoch 1 (0-50%): Unconstrained global layout exploration.
    # Epoch 2 (50-90%): Penalized feasible refinement.
    # Terminal (90-100%): Absolute constraint compliance.
    
    epoch_split = 0.50
    terminal_split = 0.90
    
    # Epoch 1: Warmup to 1.5*D, Cosine decay to 0.2*D
    ep1_warmup = jnp.clip(progress / 0.05, 0.0, 1.0)
    ep1_decay = jnp.clip((progress - 0.05) / 0.45, 0.0, 1.0)
    lr_max1 = 1.5 * D_f
    lr_min1 = 0.2 * D_f
    lr_ep1 = jnp.where(progress < 0.05,
                       gamma_min_f + ep1_warmup * (lr_max1 - gamma_min_f),
                       lr_min1 + 0.5 * (lr_max1 - lr_min1) * (1.0 + jnp.cos(jnp.pi * ep1_decay)))

    # Epoch 2: Warmup to 0.8*D, Cosine decay to gamma_min
    # The warmup aligns with the logistic penalty ramp, providing kinetic energy 
    # to escape newly formed penalty wells.
    ep2_warmup = jnp.clip((progress - epoch_split) / 0.05, 0.0, 1.0)
    ep2_decay = jnp.clip((progress - (epoch_split + 0.05)) / (terminal_split - epoch_split - 0.05), 0.0, 1.0)
    lr_max2 = 0.8 * D_f
    lr_min2 = gamma_min_f
    lr_ep2 = jnp.where(progress < (epoch_split + 0.05),
                       lr_min1 + ep2_warmup * (lr_max2 - lr_min1),
                       lr_min2 + 0.5 * (lr_max2 - lr_min2) * (1.0 + jnp.cos(jnp.pi * ep2_decay)))

    lr_main = jnp.where(progress < epoch_split, lr_ep1, lr_ep2)
    lr = jnp.where(progress >= terminal_split, gamma_min_f, lr_main)

    # --- 2. Synchronized Logistic Alpha Ramp ---
    # Transition alpha rapidly but continuously around the epoch split (progress = 0.50).
    k_alpha = 40.0  # Transition roughly covers progress 0.40 to 0.60
    switch_logistic = 1.0 / (1.0 + jnp.exp(-k_alpha * (progress - epoch_split)))
    
    alpha_ep1 = a0_f * 0.1   # Soft penalty for unimpeded exploration
    alpha_ep2 = a0_f * 20.0  # Strong penalty for feasibility
    alpha_main = alpha_ep1 + (alpha_ep2 - alpha_ep1) * switch_logistic
    
    # Terminal filter-method spike guarantees constraint satisfaction
    alpha_terminal = a0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(progress >= terminal_split, alpha_terminal, alpha_main)

    # --- 3. Phase-Transition Adam Moments ---
    # Epoch 1: Low beta2 allows rapid navigation of the chaotic unconstrained landscape.
    # Epoch 2: High beta2 absorbs the stiff curvature of the engaged penalty boundaries.
    b1_ep1, b2_ep1 = 0.15, 0.10
    b1_ep2, b2_ep2 = 0.05, 0.85
    
    beta1_main = b1_ep1 + (b1_ep2 - b1_ep1) * switch_logistic
    beta2_main = b2_ep1 + (b2_ep2 - b2_ep1) * switch_logistic
    
    # Terminal damping to freeze layout
    beta1 = jnp.where(progress >= terminal_split, 0.01, beta1_main)
    beta2 = jnp.where(progress >= terminal_split, 0.99, beta2_main)

    return lr, alpha, beta1, beta2