import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)

    # --- 1. One-Cycle Super-Convergence LR ---
    # Fast warmup to a high learning rate, then a long cosine decay.
    # This acts like a single massive simulated annealing phase, 
    # breaking out of local optima early.
    warmup_end = 0.15
    phase1 = jnp.clip(progress / warmup_end, 0.0, 1.0)
    phase2 = jnp.clip((progress - warmup_end) / (0.90 - warmup_end), 0.0, 1.0)
    
    lr_max = 1.8 * D_f
    lr_min = gamma_min_f
    
    lr_warmup = lr_min + (lr_max - lr_min) * phase1
    lr_decay = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * phase2))
    
    lr_main = jnp.where(progress < warmup_end, lr_warmup, lr_decay)
    
    is_terminal = progress >= 0.90
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)

    # --- 2. LANCELOT-style Discrete Penalty Step-ups ---
    # Instead of a continuous coupling or logistic ramp, we use discrete 
    # multiplier step-ups. This mimics the classic Augmented Lagrangian 
    # (LANCELOT) approach of solving unconstrained subproblems then 
    # tightening the penalty parameter sharply.
    
    # Phase 0 (0.00 - 0.35): Almost unconstrained, layout explodes to find new topology
    # Phase 1 (0.35 - 0.70): Moderate penalty, layout organizes into feasible regions
    # Phase 2 (0.70 - 0.90): High penalty, layout tightens against constraints
    
    alpha_phase0 = alpha0 * 0.02
    alpha_phase1 = alpha0 * 1.5
    alpha_phase2 = alpha0 * 25.0
    
    alpha_main = jnp.where(progress < 0.35, alpha_phase0,
                 jnp.where(progress < 0.70, alpha_phase1, alpha_phase2))
                 
    # Terminal absolute feasibility filter ensures compliance in the final 10%
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)

    # --- 3. Momentum Restart at Phase Boundaries ---
    # When the penalty steps up, the loss landscape changes discontinuously.
    # We sharply drop momentum (beta1) and increase beta2 at these boundaries
    # to "reset" the Adam state and prevent old gradients from hurling turbines
    # out of the newly-enforced feasible regions.
    
    b1_base, b2_base = 0.12, 0.20
    
    # Apply a "reset pulse" precisely where the penalty steps up
    # Pulse width ~ 0.02 progress on either side of the step
    pulse1 = jnp.maximum(0.0, 1.0 - jnp.abs(progress - 0.35) / 0.02)
    pulse2 = jnp.maximum(0.0, 1.0 - jnp.abs(progress - 0.70) / 0.02)
    reset_pulse = jnp.clip(pulse1 + pulse2, 0.0, 1.0)
    
    # During main phases, gradually increase beta2 to absorb constraint stiffening
    b2_trend = b2_base + (0.90 - b2_base) * (progress / 0.90)
    
    # Suppress momentum (beta1 -> 0.01) and max out variance tracker (beta2 -> 0.99) during resets
    beta1_main = jnp.where(reset_pulse > 0.5, 0.01, b1_base)
    beta2_main = jnp.where(reset_pulse > 0.5, 0.99, b2_trend)
    
    # Terminal damping
    beta1 = jnp.where(is_terminal, 0.01, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2