import jax.numpy as jnp

def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    total_f = jnp.asarray(total_steps, dtype=jnp.float32)
    step_f = jnp.asarray(step, dtype=jnp.float32)
    progress = step_f / total_f

    D_f = float(D)
    gamma_min_f = float(gamma_min)
    
    # --- Structural Change: SGDR (Stochastic Gradient Descent with Warm Restarts) ---
    # We use 3 cycles of length 0.30 for the first 90% of optimization.
    # Each cycle explores broadly at the start, then anneals into a feasibility-focused 
    # minimum. This allows the layout to repeatedly escape local minima while still 
    # resolving constraints mid-run.
    cycle_len = 0.30 
    
    # Safe floating-point modulo for cycle progress [0.0, 1.0)
    cycle_frac = progress / cycle_len
    cycle_progress = cycle_frac - jnp.floor(cycle_frac)
    
    # Envelope from 0 to 1 over the main 0.0 -> 0.9 phase
    envelope = jnp.clip(progress / 0.90, 0.0, 1.0)
    
    # --- Learning Rate: Multi-cycle Cosine with Decaying Peaks ---
    # Max lr drops each cycle to focus the search tighter as we progress
    lr_max = D_f * (1.25 - 0.75 * envelope)  # decays from 1.25*D to 0.5*D
    lr_min = gamma_min_f * 5.0
    
    # Cosine annealing from lr_max to lr_min within each cycle
    # (1 + cos(pi * c)) / 2 goes from 1.0 (at c=0) to 0.0 (at c=1)
    cos_decay = 0.5 * (1.0 + jnp.cos(jnp.pi * cycle_progress))
    lr_cyclic = lr_min + (lr_max - lr_min) * cos_decay
    
    # --- Alpha: Cyclic Feasibility Bursts ---
    # We decouple alpha, using a low penalty during early exploration of a cycle, 
    # and a high penalty at cycle end. The peak penalty grows with each cycle.
    alpha_base = alpha0 * 0.1
    alpha_peak = alpha0 * (2.0 + 30.0 * envelope)
    
    # Peak when lr is minimum (cos_decay = 0)
    alpha_cyclic = alpha_base + (alpha_peak - alpha_base) * (1.0 - cos_decay)
    
    # --- Betas: Phase-Transition Moments ---
    # Synchronize Adam moments with the cycles:
    # High momentum (low b2) when exploring to rapidly shift layouts,
    # damped (high b2) when enforcing penalty to absorb constraint curvature.
    b1_expl, b2_expl = 0.12, 0.15
    b1_settle, b2_settle = 0.02, 0.85
    
    beta1_cyclic = b1_settle + (b1_expl - b1_settle) * cos_decay
    beta2_cyclic = b2_settle + (b2_expl - b2_settle) * cos_decay
    
    # --- Terminal Feasibility Phase ---
    # Last 10% of steps ensures absolute strict compliance using exact penalty
    is_terminal = progress >= 0.90
    
    lr_terminal = gamma_min_f
    alpha_terminal = alpha0 * D_f / jnp.maximum(gamma_min_f, 1e-30)
    beta1_terminal = 0.01
    beta2_terminal = 0.99
    
    lr = jnp.where(is_terminal, lr_terminal, lr_cyclic)
    alpha = jnp.where(is_terminal, alpha_terminal, alpha_cyclic)
    beta1 = jnp.where(is_terminal, beta1_terminal, beta1_cyclic)
    beta2 = jnp.where(is_terminal, beta2_terminal, beta2_cyclic)
    
    return lr, alpha, beta1, beta2