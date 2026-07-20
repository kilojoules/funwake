import jax.numpy as jnp

def schedule_fn(step, total_steps, lr0, alpha0):
    """Iteration 082: More aggressive discovery.
    - Higher initial LR (5x).
    - Deeper alpha dips to allow freer movement.
    - 4 LR bumps instead of 3.
    - Even more reactive beta1/beta2 (0.2/0.4).
    """
    t = step / (total_steps - 1)
    lr_init = 5.0 * lr0
    lr_min = lr_init / 2000.0
    
    # ── LR: Warmup + Cosine ──────────────────────────────────────
    warmup_end = 0.04
    warmup_lr = lr_init * t / warmup_end
    
    cosine_t = jnp.clip((t - warmup_end) / (0.96 - warmup_end), 0.0, 1.0)
    cosine_lr = lr_min + (lr_init - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
    
    lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
    
    # ── 4 LR Bumps ───────────────────────────────────────────────
    def gaussian(t, center, width, height):
        return height * jnp.exp(-0.5 * ((t - center) / width) ** 2)
    
    # More frequent, sharper bumps
    bump1 = gaussian(t, 0.35, 0.03, 0.3 * lr_init)
    bump2 = gaussian(t, 0.55, 0.03, 0.4 * lr_init)
    bump3 = gaussian(t, 0.75, 0.04, 0.5 * lr_init)
    bump4 = gaussian(t, 0.88, 0.04, 0.6 * lr_init)
    
    lr = lr_base + bump1 + bump2 + bump3 + bump4
    
    # ── Alpha: Coupled + Ramp + Dips ────────────────────────────
    # Even stronger coupling
    alpha_coupled = 8.0 * alpha0 * lr_init / jnp.maximum(lr, 1e-10)
    
    # Cubic ramp for very late enforcement
    alpha_ramp = 10.0 * alpha0 * (jnp.maximum(t - 0.3, 0.0) / 0.7)**3
    
    # Very deep dips to allow near-unconstrained moves during bumps
    dips = (0.90 * gaussian(t, 0.36, 0.04, 1.0) +
            0.92 * gaussian(t, 0.56, 0.04, 1.0) +
            0.95 * gaussian(t, 0.76, 0.05, 1.0) +
            0.98 * gaussian(t, 0.89, 0.05, 1.0))
    
    alpha = (alpha_coupled + alpha_ramp) * (1.0 - jnp.minimum(dips, 0.99))
    
    # ── SQUEEZE ───────────────────────────────────────────────────
    is_squeeze = (t > 0.98)
    lr = jnp.where(is_squeeze, 0.0001 * lr0, lr)
    alpha = jnp.where(is_squeeze, 2000000.0 * alpha0, alpha)
    
    # ── Beta: Very Reactive ───────────────────────────────────────
    beta1 = 0.2
    beta2 = 0.4
    
    return lr, alpha, beta1, beta2
