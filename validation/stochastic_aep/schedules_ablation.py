"""Bump-ablation schedule: iter_192 with the two Gaussian LR bumps removed,
everything else (warmup, cosine, α-escalation, betas) identical.

Sourced from runs/schedule_only_5hr/iter_192.py / paper_schedules
schedules.py:funwake_iter192. The ONLY change is removing the two
`bump1` and `bump2` terms from `lr_base`.
"""
import jax.numpy as jnp


def funwake_iter192_baseline_alpha(lr_init: float = 50.0):
    """iter_192 schedule with α-escalation replaced by the baseline rule
    α = α₀ · lr_init / lr (Quick 2023 Eq. 15).

    Everything else preserved: warmup, cosine, BOTH bumps at t=0.5 and
    t=0.75, betas (0.3, 0.5). Isolates whether iter_192's α-escalation
    profile drives the AEP advantage.
    """
    lr0_setting = float(lr_init)

    def apply(step, total_steps, lr0, alpha0):
        lr_ref = lr0_setting
        t = step / total_steps
        lr_peak = 4.0 * lr_ref
        lr_min = lr_peak / 10000.0
        warmup_end = 0.05
        warmup_lr = lr_peak * t / warmup_end
        cosine_t = (t - warmup_end) / (1.0 - warmup_end)
        cosine_lr = lr_min + (lr_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
        lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
        # KEEP bumps (full iter_192 lr trajectory)
        bump1 = 0.2 * lr_peak * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
        bump2 = 0.3 * lr_peak * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
        lr = lr_base + bump1 + bump2

        # SWAP α to baseline rule
        alpha = alpha0 * lr_ref / jnp.maximum(lr, 1e-10)

        return lr, alpha, jnp.float64(0.3), jnp.float64(0.5)
    return apply


def funwake_iter192_alpha_scaled(lr_init: float = 50.0, factor: float = 1.0):
    """iter_192 with α multiplied by ``factor``. lr/β unchanged. Mirrors
    tools/alpha_ablation.py FACTORS sweep, now usable under K=50 stochastic
    SGD."""
    lr0_setting = float(lr_init)
    f = float(factor)

    def apply(step, total_steps, lr0, alpha0):
        lr_ref = lr0_setting
        t = step / total_steps
        lr_peak = 4.0 * lr_ref
        lr_min = lr_peak / 10000.0
        warmup_end = 0.05
        warmup_lr = lr_peak * t / warmup_end
        cosine_t = (t - warmup_end) / (1.0 - warmup_end)
        cosine_lr = lr_min + (lr_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
        lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
        bump1 = 0.2 * lr_peak * jnp.exp(-0.5 * ((t - 0.5) / 0.04) ** 2)
        bump2 = 0.3 * lr_peak * jnp.exp(-0.5 * ((t - 0.75) / 0.05) ** 2)
        lr = lr_base + bump1 + bump2

        alpha_base = 5.0 * alpha0 * lr_peak / jnp.maximum(lr, 1e-10)
        late = jnp.maximum(t - 0.5, 0.0) / 0.5
        alpha_extra = 3.0 * alpha0 * late ** 2
        dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
        alpha = f * (alpha_base + alpha_extra) * (1.0 - dip)

        return lr, alpha, jnp.float64(0.3), jnp.float64(0.5)
    return apply


def funwake_iter192_no_bumps(lr_init: float = 50.0):
    """iter_192 verbatim minus the two LR bumps at t=0.5 and t=0.75.

    All else identical: warmup_end=0.05, cosine decay, α-escalation
    (5α₀·lr_peak/lr + 3α₀·t² late + dip at t=0.6), β₁=0.3, β₂=0.5.
    """
    lr0_setting = float(lr_init)

    def apply(step, total_steps, lr0, alpha0):
        lr_ref = lr0_setting
        t = step / total_steps
        lr_peak = 4.0 * lr_ref
        lr_min = lr_peak / 10000.0
        warmup_end = 0.05
        warmup_lr = lr_peak * t / warmup_end
        cosine_t = (t - warmup_end) / (1.0 - warmup_end)
        cosine_lr = lr_min + (lr_peak - lr_min) * 0.5 * (1.0 + jnp.cos(jnp.pi * cosine_t))
        lr_base = jnp.where(t < warmup_end, warmup_lr, cosine_lr)
        # BUMPS REMOVED — no bump1, no bump2
        lr = lr_base

        # α-escalation identical to iter_192
        alpha_base = 5.0 * alpha0 * lr_peak / jnp.maximum(lr, 1e-10)
        late = jnp.maximum(t - 0.5, 0.0) / 0.5
        alpha_extra = 3.0 * alpha0 * late ** 2
        dip = 0.5 * jnp.exp(-0.5 * ((t - 0.6) / 0.04) ** 2)
        alpha = (alpha_base + alpha_extra) * (1.0 - dip)

        beta1 = jnp.float64(0.3)
        beta2 = jnp.float64(0.5)
        return lr, alpha, beta1, beta2
    return apply
