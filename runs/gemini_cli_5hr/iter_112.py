import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 112: 4-cycle Cosine Annealing with late Sigmoid Alpha ramp.
    - 4 cycles of 2000 steps.
    - Low Beta (0.1, 0.2) as in iter_088.
    - Alpha global ramp using sigmoid starting at t_global = 0.6.
    - Peak LR decay: 10*lr0 -> 5*lr0.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Decay the peak of each cycle
    lr_peak = lr0 * (10.0 - 5.0 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global alpha: low for 60% of time, then sharp sigmoid ramp.
    alpha_global_scale = 1.0 + 400.0 * (jax.nn.sigmoid(20.0 * (t_global - 0.65)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Local coupling to LR
    alpha_local = alpha_global * (lr0 * 10.0 / jnp.maximum(lr, 1e-10))
    
    # Sharp dip at the start of each cycle
    dip_magnitude = 0.95 * (1.0 - 0.4 * t_global)
    dip_width = 0.12
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1
    beta2 = 0.2
    
    return lr, alpha, beta1, beta2
