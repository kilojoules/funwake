import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 143: Meta-Phase Hybrid.
    - Phase 1 (75%): 10 discovery cycles, low Beta (TopFarm), high exploration.
    - Phase 2 (25%): Smooth refinement, higher Beta (stabilizing), steep Alpha.
    - Peak LR multiplier: 12.0.
    - Alpha Dips in Phase 1.
    """
    
    t = step / (total_steps - 1.0)
    
    # ── Phase 1 (Discovery) ────────────────────────────────────
    # 10 cycles of 600 steps each
    is_p1 = (t < 0.75)
    cycle_len = (total_steps * 0.75) / 10.0
    t_c1 = (step % cycle_len) / (cycle_len - 1.0)
    
    lr_peak_p1 = lr0 * (12.0 * (1.0 - t/0.75) + 4.0 * (t/0.75))
    lr_min_p1 = lr0 * 0.005
    lr_p1 = lr_min_p1 + 0.5 * (lr_peak_p1 - lr_min_p1) * (1.0 + jnp.cos(jnp.pi * t_c1))
    
    alpha_p1_global = alpha0 * (1.0 + 150.0 * (t ** 3.0))
    alpha_p1_coupled = alpha_p1_global * (lr0 * 12.0 / jnp.maximum(lr_p1, 1e-10))
    dip_p1 = 0.98 * jnp.exp(- (t_c1**2) / (2 * 0.15**2))
    alpha_p1 = alpha_p1_coupled * (1.0 - dip_p1)
    
    b1_p1 = 0.1
    b2_p1 = 0.2
    
    # ── Phase 2 (Refinement) ───────────────────────────────────
    t_conv = (t - 0.75) / 0.25
    lr_p2 = lr0 * 0.1 * (1.0 - t_conv) + lr0 * 0.0001
    alpha_p2 = alpha0 * (150.0 + 5000000.0 * t_conv**4)
    b1_p2 = 0.4
    b2_p2 = 0.9
    
    # ── Final Selection ────────────────────────────────────────
    lr = jnp.where(is_p1, lr_p1, lr_p2)
    alpha = jnp.where(is_p1, alpha_p1, alpha_p2)
    beta1 = jnp.where(is_p1, b1_p1, b1_p2)
    beta2 = jnp.where(is_p1, b2_p1, b2_p2)
    
    # Final Squeeze (last 1%)
    is_sq = (t > 0.99)
    lr = jnp.where(is_sq, lr0 * 0.00001, lr)
    alpha = jnp.where(is_sq, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
