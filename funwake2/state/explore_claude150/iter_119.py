import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): the lr waveform is
# replaced by a true ONE-CYCLE SUPERCONVERGENCE profile (prior-art menu §2,
# untried) with the momentum ANTI-CORRELATED to lr (menu §beta1, untried),
# and the discrete burst machinery is replaced by a CONTINUOUS anti-correlated
# penalty: alpha rises smoothly as lr falls, i.e. repair pressure grows
# exactly as the steps shrink (graduated ALM), instead of firing in notches.
#
# Why this is a different mechanism, not a re-tune: the best schedule
# alternates HOT/COLD regimes (hold + repair notches), so every basin hop is
# followed by a cold stop. One-cycle instead makes ONE long excursion: a slow
# half-cosine climb to a core far hotter than any tried hold (2.0*D vs the
# 1.5*D hold / 1.65*D momentary peaks), then a long, monotone half-cosine
# anneal that lets the layout settle into the best basin found at peak heat
# without ever re-heating. Momentum is lowest at peak lr (stability while
# hopping) and highest during the anneal (carries turbines down long flat
# valleys — momentum as implicit ALM multiplier). The proven feasibility
# endgame is preserved verbatim: bounded 6*alpha0 plateau, cubic-delayed
# geometric climb to the 5*alpha0*D/gamma_min terminal spike, beta2 0.2->0.9
# at the cool-down, gated terminal beta1 drop, linear lr tail landing exactly
# on gamma_min.
#
#   lr    — 3% linear warmup (proven) * one-cycle: half-cosine 0.35*D -> peak
#           2.0*D at 30% -> long half-cosine anneal to 0.5*D at 70% (floor
#           drifts 0.35 -> 0.5) -> proven straight linear tail to gamma_min.
#   alpha — 0.4*alpha0 floor + CONTINUOUS anti-correlated term
#           A_ANTI*fp*(1-lr_norm): zero at the hot peak, ~4.4*alpha0 by the
#           end of the anneal; logistic handoff to the bounded 6*alpha0
#           plateau at 72%; proven cubic terminal climb from 80% to
#           5*alpha0*D/gamma_min at the last step.
#   betas — beta1 anti-correlated with lr (0.15 in cold phases, 0.04 at peak
#           heat), gated drop to 0.02 in the terminal spike; beta2 logistic
#           0.2 -> 0.9 centered at the 70% cool-down (proven values).
_F_WARM = 0.03      # proven short linear lr warmup
_F_PEAK = 0.30      # one-cycle peak position (fraction of full run)
_F_END = 0.70       # exploration ends; linear tail to gamma_min at 100%
_LR_START = 0.35    # cycle start lr, in units of D
_LR_PEAK = 2.0      # one-cycle peak — hotter core than any tried hold
_LR_END = 0.5       # anneal endpoint; the linear tail launches from here
_A_LO = 0.4         # exploration penalty floor, in alpha0 units (proven)
_A_ANTI = 4.0       # continuous anti-correlated repair pressure amplitude
_A_PLAT = 6.0       # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.72    # logistic plateau ramp just after the anneal ends
_A_WIDTH = 0.04
_F_TERM = 0.80      # terminal geometric alpha climb start
_POW = 3.0          # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0    # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.70   # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_MAX = 0.15      # momentum when lr is cold (anneal / polish)
_B1_MIN = 0.04      # momentum at peak heat — stability while basin-hopping
_B1_LO = 0.02       # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.90
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle (half-cosine up, long half-cosine down) -> linear tail ---
    # lr_norm = 0.25*(1-cos(pi*u))*(1+cos(pi*d)) is 0 at both ends of the
    # cycle and exactly 1 at the peak: u saturates at 1 before the peak while
    # d is still 0, so the product is a smooth single hump.
    fp = jnp.clip(frac, 0.0, _F_END) / _F_END
    p_pos = _F_PEAK / _F_END
    u = jnp.clip(fp / p_pos, 0.0, 1.0)
    d = jnp.clip((fp - p_pos) / (1.0 - p_pos), 0.0, 1.0)
    lr_norm = 0.25 * (1.0 - jnp.cos(jnp.pi * u)) * (1.0 + jnp.cos(jnp.pi * d))
    base = _LR_START + (_LR_END - _LR_START) * fp             # drifting cycle floor
    lr_exp = (base + (_LR_PEAK - base) * lr_norm) * Dj
    p = jnp.clip((frac - _F_END) / (1.0 - _F_END), 0.0, 1.0)
    lr_env = gmin + (lr_exp - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # proven; lr only
    lr = lr_env * warm

    # --- alpha: floor + continuous anti-correlated repair -> plateau -> climb ---
    # The anti term is weighted by fp so alpha stays at the floor early (the
    # grid start is feasible), vanishes at peak heat, and grows through the
    # anneal as steps shrink; (1-ramp) hands it off smoothly to the plateau.
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    anti = _A_ANTI * fp * (1.0 - lr_norm) * (1.0 - ramp)
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + anti
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 anti-correlation + proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_MAX - (_B1_MAX - _B1_MIN) * lr_norm          # low momentum at peak lr
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2