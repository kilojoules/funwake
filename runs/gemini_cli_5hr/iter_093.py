import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 093: Supercharged iter_087 with warmups.
    - 4 cycles of cosine decay.
    - Warmup (5% of cycle) at the start of each cycle.
    - Higher initial LR and steeper alpha ramp.
    """
    n_cycles = 4
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    lr_peak = lr0 * 12.0 * (1.0 - 0.6 * t_global)
    lr_min = lr0 * 0.001
    
    # Cycle warmup (first 5% of cycle_len)
    warmup_fraction = 0.05
    is_warmup = t_cycle < warmup_fraction
    
    # Cosine part
    cosine_t = (t_cycle - warmup_fraction) / (1.0 - warmup_fraction)
    cosine_lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * jnp.maximum(cosine_t, 0.0)))
    
    # Warmup part
    warmup_lr = lr_min + (lr_peak - lr_min) * (t_cycle / warmup_fraction)
    
    lr = jnp.where(is_warmup, warmup_lr, cosine_lr)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Steeper global ramp
    alpha_global = alpha0 * (1.0 + 200.0 * (t_global ** 4))
    
    # Coupling
    alpha_local = alpha_global * (lr0 * 12.0 / jnp.maximum(lr, 1e-10))
    
    # Dip at start of cycle (covering warmup)
    dip_mag = 0.95 * (1.0 - 0.6 * t_global)
    dip = dip_mag * jnp.exp(-(t_cycle**2) / (2 * 0.08**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_squeeze = (t_global > 0.98)
    lr = jnp.where(is_squeeze, lr0 * 0.00001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2
