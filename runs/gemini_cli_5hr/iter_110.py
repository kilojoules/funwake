import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 110: 3-cycle Cosine Annealing with cyclic Beta and sharp Alpha dip.
    - 3 cycles of ~2666 steps each.
    - LR peak decays: 12*lr0 -> 8*lr0 -> 6*lr0.
    - Beta1/Beta2 ramp within EACH cycle: (0.1, 0.2) -> (0.5, 0.8).
    - Alpha base increases cubically.
    - Alpha coupling: alpha = alpha_base * (lr0 * 6.0 / lr).
    - Sharp alpha dip at start of cycle (t_cycle < 0.1).
    """
    n_cycles = 3
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Peak LR decays from 12*lr0 to 6*lr0
    lr_peak = lr0 * (12.0 - 6.0 * t_global)
    lr_min = lr0 * 0.001
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global alpha scale increases from 1 to 801
    alpha_global_scale = 1.0 + 800.0 * (t_global ** 3)
    alpha_global = alpha0 * alpha_global_scale
    
    # Local coupling: higher LR -> lower Alpha
    alpha_coupled = alpha_global * (lr0 * 6.0 / jnp.maximum(lr, 1e-10))
    
    # Sharp dip at the start of each cycle
    dip_magnitude = 0.98 * (1.0 - 0.5 * t_global)
    dip_width = 0.10
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 20000000.0, alpha)

    # ── Beta (Cyclic) ───────────────────────────────────────────
    # Resetting momentum at each cycle start to allow new exploration
    beta1 = 0.1 + 0.4 * t_cycle
    beta2 = 0.2 + 0.6 * t_cycle
    
    return lr, alpha, beta1, beta2
