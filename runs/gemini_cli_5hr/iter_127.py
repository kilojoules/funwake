import jax
import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 127: 'Stay Feasible' Strategy.
    Since we start with a grid-initialized feasible layout, we try to 
    remain in the feasible space and optimize AEP locally.
    - Constant moderate LR.
    - Constant high Alpha.
    - Standard Adam parameters for stable convergence.
    """
    
    t_global = step / (total_steps - 1)
    
    # ── Learning Rate ───────────────────────────────────────────
    # Start at lr0, slowly decay to 0.1*lr0
    lr = lr0 * (1.0 - 0.9 * t_global)
    
    # ── Alpha (Penalty) ─────────────────────────────────────────
    # Keep alpha high to prevent crossing boundaries/collisions
    alpha = alpha0 * 5000.0
    
    # ── Beta ────────────────────────────────────────────────────
    # Standard Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    
    return lr, alpha, beta1, beta2
