import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 099: Hybrid of iter_068 (no-momentum Adam + shake) and iter_087 (4 cycles + coupling).
    - 4 cycles of cosine decay with high-frequency LR shake.
    - No-momentum Adam (beta1=0.0, beta2=0.992) to match unfeasible best.
    - Coupled alpha for feasibility.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 4.0 * (1.0 - 0.5 * t_global)
    lr_min = lr0 * 0.005
    lr_base = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # Add high-frequency shake (from iter_068)
    shake_amp = 0.05 * jnp.sqrt(1.0 - t_global)
    shake = 1.0 + shake_amp * jnp.sin(150.0 * jnp.pi * t_global)
    lr = lr_base * shake
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global ramp
    alpha_global = alpha0 * (1.0 + 100.0 * (t_global ** 3))
    
    # Coupling to maintain relative importance
    alpha_local = alpha_global * (lr0 * 4.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of cycle
    dip = 0.9 * (1.0 - 0.5 * t_global) * jnp.exp(-(t_cycle**2) / (2 * 0.1**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 5000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    # No momentum, stable second moment (from unfeasible best 068)
    beta1 = 0.0
    beta2 = 0.992
    
    return lr, alpha, beta1, beta2
