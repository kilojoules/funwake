import jax.numpy as jnp

# Diameter-rule exploration peak: slightly above the parent's 200/240 ≈ 0.833
# rule, and held longer, to buy extra AEP exploration before the cool-down.
_C = 0.90            # lr_peak = _C * D
_WARM_FRAC = 0.03    # short one-cycle-style linear warmup
_HOLD_END = 0.40     # hold lr at peak until 40% of the run (parent: 1/3)
_DECAY_END = 0.92    # lr reaches gamma_min here; final 8% is feasibility polish
_SPIKE_MID = 0.95    # center of the bounded logistic terminal alpha spike
_SPIKE_WIDTH = 0.015
_SPIKE_GAIN = 3.0    # alpha multiplied by up to (1 + gain) in the last ~5%


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    t = step / total_steps                        # progress in [0, 1), traceable

    lr_peak = _C * D
    lr_min = jnp.maximum(gamma_min, 1e-12)

    # One-cycle-style warmup into a long hold at the exploration peak.
    warm = jnp.clip(t / _WARM_FRAC, 0.0, 1.0)
    lr_hold = lr_peak * (0.3 + 0.7 * warm)

    # Cosine-smoothed geometric cool-down from lr_peak to gamma_min over
    # [_HOLD_END, _DECAY_END], then flat at gamma_min for the terminal phase.
    s = jnp.clip((t - _HOLD_END) / (_DECAY_END - _HOLD_END), 0.0, 1.0)
    s_cos = 0.5 * (1.0 - jnp.cos(jnp.pi * s))
    lr = lr_hold * (lr_min / lr_peak) ** s_cos

    # Native coupling backbone: alpha0 = mean|grad J|/D, so alpha0*D/lr is the
    # proven mean|grad J|/lr coupling. It sits at the plateau g/lr_peak during
    # exploration, ramps as lr cools, and a bounded logistic terminal spike
    # (last ~5% of steps, lr already at gamma_min) adds extra feasibility
    # restoration to offset the higher/longer exploration peak.
    coupled = alpha0 * D / jnp.maximum(lr, 1e-30)
    spike = 1.0 + _SPIKE_GAIN / (1.0 + jnp.exp(-(t - _SPIKE_MID) / _SPIKE_WIDTH))
    alpha = coupled * spike

    # TopFarm moments — proven strictly feasible with this coupling backbone.
    beta1 = 0.1
    beta2 = 0.2
    return lr, alpha, beta1, beta2