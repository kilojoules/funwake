import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 137: Hierarchical Cyclic Strategy.
    - 4 Large Cycles (2000 steps each) providing global refinement.
    - Mini-Shakes every 500 steps (4 per large cycle) to escape local minima.
    - LR: Spikes at each mini-shake, decaying within each large cycle.
    - Alpha: Deep dips during spikes, coupled to 1/LR, with quadratic global ramp.
    - Beta: Ramps from TopFarm-style (0.15, 0.3) to more stable (0.4, 0.9).
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Cycle Definitions ──────────────────────────────────────
    # 4 Large Cycles
    large_cycle_len = 2000
    t_large = (step % large_cycle_len) / (large_cycle_len - 1)
    
    # 4 Mini-shakes per large cycle (every 500 steps)
    mini_cycle_len = 500
    t_mini = (step % mini_cycle_len) / (mini_cycle_len - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Global Peak LR (decays globally)
    lr_peak_global = lr0 * (9.0 * (1.0 - 0.6 * t_global) + 3.0 * 0.6 * t_global)
    
    # Base LR: decays linearly within each large cycle
    lr_base = lr_peak_global * (1.0 - 0.85 * t_large)
    
    # Mini-shakes: LR spikes at start of each mini-cycle
    spike_mag = 0.5 * lr_peak_global * (1.0 - 0.4 * t_global)
    spike_width = 0.12
    spike = spike_mag * jnp.exp(- (t_mini**2) / (2 * spike_width**2))
    
    lr = jnp.maximum(lr_base + spike, 1e-4 * lr0)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global Alpha Ramp
    alpha_global_scale = 1.0 + 130.0 * (t_global ** 2.5)
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling to 1/LR (using global peak as reference)
    alpha_coupled = alpha_global * (lr0 * 9.0 / jnp.maximum(lr, 1e-10))
    
    # Deep Dips during spikes
    dip_mag = 0.97 * (1.0 - 0.3 * t_global)
    dip_width = 0.18
    dip = dip_mag * jnp.exp(- (t_mini**2) / (2 * dip_width**2))
    
    alpha = alpha_coupled * (1.0 - dip)
    
    # ── Beta ────────────────────────────────────────────────────
    # Smooth global ramp for Beta to increase stability
    beta1 = 0.15 + 0.3 * t_global
    beta2 = 0.3 + 0.6 * t_global
    
    # ── Final Squeeze (last 1%) ─────────────────────────────────
    is_squeeze = (t_global > 0.99)
    lr = jnp.where(is_squeeze, lr0 * 0.000001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 1e10, alpha)
    
    return lr, alpha, beta1, beta2
