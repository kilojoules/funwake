import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Refined multi-stage strategy:
    - 5 Cycles of warm restarts for LR to shake local optima.
    - Alpha stays low (p=6) to allow maximum AEP exploration.
    - Beta2 annealed from 0.95 to 0.999 for stability.
    - Aggressive squeeze in final 5% for feasibility.
    """
    t = step / (total_steps - 1)
    
    # ── LR: Warm Restarts ──────────────────────────────────────────
    num_cycles = 5
    cycle_idx = jnp.minimum((t * num_cycles).astype(int), num_cycles - 1)
    t_cycle = (t * num_cycles) % 1.0
    
    # Decay peak LR over cycles
    lr_peak = 1.5 * (0.8**cycle_idx) * lr0
    lr_min = 0.001 * lr0
    
    lr_in_cycle = lr_min + (lr_peak - lr_min) * 0.5 * (1 + jnp.cos(jnp.pi * t_cycle))
    
    # ── Alpha: Aggressive Power Ramp ──────────────────────────────
    # p=6 keeps alpha very low until the end, maximizing AEP discovery.
    alpha_base = alpha0 * (1.0 + 49999.0 * t**6)
    
    # ── Beta: Annealing ───────────────────────────────────────────
    # No momentum as it seems to help in this problem
    beta1 = 0.0
    # Anneal beta2 from reactive (0.95) to stable (0.999)
    beta2 = 0.95 + (0.999 - 0.95) * t
    
    # ── SQUEEZE PHASE (Final 5%) ──────────────────────────────────
    is_squeeze = (t > 0.95)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr_in_cycle)
    # Huge penalty at the end to force feasibility
    alpha = jnp.where(is_squeeze, 5000000.0 * alpha0, alpha_base)
    
    return lr, alpha, beta1, beta2
