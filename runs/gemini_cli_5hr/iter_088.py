import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 088: 3-cycle Cosine Annealing.
    - Focus on broader exploration in early cycles by keeping alpha lower.
    - Steep alpha ramp in the final cycle.
    - Use 'alpha0' as a scale for penalty magnitude.
    """
    n_cycles = 3
    cycle_len = total_steps // n_cycles
    cycle_idx = step // cycle_len
    t_cycle = (step % cycle_len) / (cycle_len - 1)
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Higher initial peaks for early exploration
    lr_peak = lr0 * 8.0 * (1.0 - 0.7 * t_global)
    lr_min = lr0 * 0.005
    lr = lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Global alpha ramp is lower in cycles 0 & 1, then very high in 2.
    # We'll use a sigmoid-like ramp for global alpha.
    alpha_global_scale = 1.0 + 300.0 * (jax.nn.sigmoid(15.0 * (t_global - 0.75)))
    alpha_global = alpha0 * alpha_global_scale
    
    # Coupling
    alpha_local = alpha_global * (lr0 * 8.0 / jnp.maximum(lr, 1e-10))
    
    # Dip alpha at start of cycle
    dip_magnitude = 0.95 * (1.0 - 0.6 * t_global)
    dip_width = 0.12
    dip = dip_magnitude * jnp.exp(- (t_cycle**2) / (2 * dip_width**2))
    alpha = alpha_local * (1.0 - dip)
    
    # ── SQUEEZE ──────────────────────────────────────────────────
    is_last_cycle = (cycle_idx == n_cycles - 1)
    is_squeeze = is_last_cycle & (t_cycle > 0.96)
    
    lr = jnp.where(is_squeeze, lr0 * 0.0001, lr)
    alpha = jnp.where(is_squeeze, alpha0 * 10000000.0, alpha)

    # ── Beta ────────────────────────────────────────────────────
    # Try even lower beta1/beta2 for extreme reactivity?
    beta1 = 0.1
    beta2 = 0.2
    
    return lr, alpha, beta1, beta2
