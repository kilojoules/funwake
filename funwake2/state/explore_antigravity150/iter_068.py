import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)
    alpha0_f = jnp.asarray(alpha0, dtype=jnp.float32)

    # --- Alternating Exploration and Projection (ADMM-Style) ---
    # A structurally new approach: rather than continuous decay, we explicitly alternate
    # between "Exploration" phases (high LR, low penalty, high momentum) to jump over 
    # barriers, and "Projection" bursts (low LR, high penalty, low momentum) to pull 
    # the layout back into feasibility. This answers the "mid-run feasibility bursts" 
    # hypothesis using clean 1D interpolation over progress keyframes.
    
    # Progress keyframes:
    xp = jnp.array([
        0.00, 0.05, 0.15,  # Warmup & Explore 1
        0.20, 0.25,        # Project 1 (Feasibility Burst)
        0.30, 0.45,        # Explore 2
        0.50, 0.55,        # Project 2 (Feasibility Burst)
        0.60, 0.75,        # Explore 3
        0.80, 0.85,        # Project 3 (Feasibility Burst)
        0.90, 1.00         # Terminal Feasibility Focus
    ])
    
    # Learning Rate Multipliers (relative to D_f)
    # Peaks decay across explore phases; dips get deeper across project phases
    lr_mult = jnp.array([
        0.00, 1.40, 1.40,  # Explore 1 peak (sustained)
        0.15, 0.15,        # Project 1 dip
        0.80, 0.80,        # Explore 2 peak
        0.10, 0.10,        # Project 2 dip
        0.40, 0.40,        # Explore 3 peak
        0.05, 0.05,        # Project 3 dip
        0.00, 0.00         # Terminal
    ])
    
    # Alpha Multipliers (relative to alpha0_f)
    # Base penalty rises across explore phases; peak penalty escalates across bursts
    alpha_mult = jnp.array([
        0.1,  0.1,  0.1,   # Explore 1 soft penalty
        5.0,  5.0,         # Project 1 burst
        0.5,  0.5,         # Explore 2 soft penalty
        15.0, 15.0,        # Project 2 burst
        2.0,  2.0,         # Explore 3 soft penalty
        30.0, 30.0,        # Project 3 burst
        100.0, 100.0       # Terminal (placeholder, overridden mathematically)
    ])
    
    # Beta1 (Momentum)
    # Drops during projection bursts to prevent constraint ping-pong
    b1_vals = jnp.array([
        0.15, 0.15, 0.15,  # E1
        0.05, 0.05,        # P1
        0.10, 0.10,        # E2
        0.03, 0.03,        # P2
        0.08, 0.08,        # E3
        0.02, 0.02,        # P3
        0.01, 0.01         # Terminal
    ])
    
    # Beta2 (RMSprop damping)
    # Rises during projection bursts to absorb stiff constraint curvature
    b2_vals = jnp.array([
        0.15, 0.15, 0.15,  # E1
        0.80, 0.80,        # P1
        0.40, 0.40,        # E2
        0.90, 0.90,        # P2
        0.60, 0.60,        # E3
        0.95, 0.95,        # P3
        0.99, 0.99         # Terminal
    ])
    
    # JAX linear interpolation smoothly handles transitions between all phases
    lr_base = D_f * jnp.interp(progress, xp, lr_mult)
    alpha_base = alpha0_f * jnp.interp(progress, xp, alpha_mult)
    beta1_base = jnp.interp(progress, xp, b1_vals)
    beta2_base = jnp.interp(progress, xp, b2_vals)
    
    # Enforce strict bounds during the dynamic run
    lr_base = jnp.maximum(lr_base, gamma_min_f)
    
    # --- Terminal Feasibility Restoration ---
    # Absolute strict compliance enforced via the exact-penalty formula in the final 10%
    is_terminal = progress >= 0.90
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0_f * D_f / jnp.maximum(gamma_min_f, 1e-30)
    beta1_terminal = 0.01
    beta2_terminal = 0.99
    
    lr = jnp.where(is_terminal, lr_terminal, lr_base)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_base)
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_base)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_base)
    
    return lr, alpha, beta1, beta2