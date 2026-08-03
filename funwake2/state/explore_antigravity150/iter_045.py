import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = jnp.asarray(D, dtype=jnp.float32)
    gamma_min_f = jnp.asarray(gamma_min, dtype=jnp.float32)

    # --- Anti-Correlated Cyclic SGDR & Breathing Penalty ---
    # STRUCTURALLY DIFFERENT from a single warmup-plateau-decay.
    # Uses 3 cycles of cosine learning rate decay with warm restarts.
    # Crucially, the penalty (alpha) "breathes": it drops at each restart
    # to allow topological exploration (turbines sliding past each other),
    # then tightens up by the end of each cycle. 
    # Adam moments also cycle: low momentum during restarts for rapid 
    # layout shifts, high damping at cycle ends for constraint adherence.

    c1_end = 0.35
    c2_end = 0.65
    c3_end = 0.90

    # Cycle Progress Variables (0 to 1 within their respective phases)
    p1 = jnp.clip(progress / c1_end, 0.0, 1.0)
    p2 = jnp.clip((progress - c1_end) / (c2_end - c1_end), 0.0, 1.0)
    p3 = jnp.clip((progress - c2_end) / (c3_end - c2_end), 0.0, 1.0)

    # Helper for Cosine Annealing
    def cos_decay(lr_max, lr_min, p):
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + jnp.cos(jnp.pi * p))

    # 1. Cyclic Learning Rates
    lr1 = cos_decay(1.50 * D_f, gamma_min_f, p1)
    lr2 = cos_decay(0.70 * D_f, gamma_min_f, p2)
    lr3 = cos_decay(0.20 * D_f, gamma_min_f, p3)

    # 2. Breathing, Progressively Stricter Penalty (Alpha)
    # Drops at restarts, but peak strictness increases each cycle.
    # Using squared progress to keep it soft early in the cycle, ramping late.
    a1 = alpha0 * (0.05 + 0.95 * p1**2)   # Cycle 1: 0.05x -> 1.0x
    a2 = alpha0 * (0.25 + 4.75 * p2**2)   # Cycle 2: 0.25x -> 5.0x
    a3 = alpha0 * (1.00 + 14.0 * p3**2)   # Cycle 3: 1.00x -> 15.0x

    # 3. Phase-Transition Cyclic Moments
    # High beta2 when alpha is high to absorb stiff constraint curvature.
    # b1 drops over the cycle, b2 ramps up.
    b1_c1 = 0.15 - 0.10 * p1   # 0.15 -> 0.05
    b2_c1 = 0.15 + 0.65 * p1   # 0.15 -> 0.80

    b1_c2 = 0.10 - 0.08 * p2   # 0.10 -> 0.02
    b2_c2 = 0.30 + 0.60 * p2   # 0.30 -> 0.90

    b1_c3 = 0.05 - 0.04 * p3   # 0.05 -> 0.01
    b2_c3 = 0.50 + 0.45 * p3   # 0.50 -> 0.95

    # Merge cycles safely with jnp.where
    lr_main = jnp.where(progress < c1_end, lr1,
              jnp.where(progress < c2_end, lr2, lr3))

    alpha_main = jnp.where(progress < c1_end, a1,
                 jnp.where(progress < c2_end, a2, a3))

    beta1_main = jnp.where(progress < c1_end, b1_c1,
                 jnp.where(progress < c2_end, b1_c2, b1_c3))

    beta2_main = jnp.where(progress < c1_end, b2_c1,
                 jnp.where(progress < c2_end, b2_c2, b2_c3))

    # --- 4. Terminal Feasibility Phase ---
    # The filter method demands strict compliance in the final 10%.
    # We crush the LR and spike the penalty massively.
    is_terminal = progress >= c3_end
    
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    
    lr = jnp.where(is_terminal, gamma_min_f, lr_main)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_main)
    beta1 = jnp.where(is_terminal, 0.00, beta1_main)
    beta2 = jnp.where(is_terminal, 0.99, beta2_main)

    return lr, alpha, beta1, beta2