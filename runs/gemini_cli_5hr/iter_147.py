import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 147: 12-cycle "Persistent Shaking".
    - 12 cycles of equal length.
    - LR peak remains higher for longer (1.0 - t^2 decay).
    - Alpha: coupled to 1/LR, global cubic ramp (200x).
    - Alpha Dips: 0.98 magnitude, becoming slightly shallower (0.98 -> 0.68).
    - Beta: Global ramp from (0.1, 0.2) to (0.4, 0.9).
    - Final Squeeze: last 1.5%.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Cycle Definitions (12 cycles) ──────────────────────────
    n_cycles = 12
    cycle_len = total_steps // n_cycles
    t_c = (step % cycle_len) / (cycle_len - 1.0)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Quadratic global decay to stay in exploration mode longer
    lr_peak_global = lr0 * (10.0 * (1.0 - t**2) + 2.0 * t**2)
    lr_min = lr0 * 0.001
    
    # Cosine annealing
    lr_cyclic = lr_min + 0.5 * (lr_peak_global - lr_min) * (1.0 + jnp.cos(jnp.pi * t_c))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global Alpha Ramp
    alpha_global_scale = 1.0 + 200.0 * (t ** 3.0)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR (reference peak 10.0)
    alpha_coupled = alpha_global * (lr0 * 10.0 / jnp.maximum(lr_cyclic, 1e-10))
    
    # Dip at start of cycle
    dip_mag = 0.98 * (1.0 - 0.3 * t)
    dip_width = 0.15
    dip = dip_mag * jnp.exp(- (t_c**2) / (2 * dip_width**2))
    
    alpha_dip = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.1 + 0.3 * t
    beta2 = 0.2 + 0.7 * t
    
    # ── Final Selection ─────────────────────────────────────────
    is_sq = (t > 0.985)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr_cyclic)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha_dip)
    
    return lr, alpha, beta1, beta2
