import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): NO restarts, NO
# bursts. This is the prior-art menu's top untried lr bet — WSD / one-cycle
# (§6, §2): "hold near c*D, then (near-)linear cool-down beats cosine/product
# decay". Every schedule in the lineage oscillates (SGDR peaks 1.65*D->1.05*D
# with cold 0.65*D troughs, time-averaged exploration lr ~1.0*D); this one
# holds a SUSTAINED hot plateau the whole exploration phase, so every one of
# the first 60% of steps is a full-size basin-hopping step instead of half of
# them being spent in cosine troughs.
#
#   lr    — 4% linear warmup -> gently tilted stable hold 1.45*D -> 1.15*D
#           over 60% of the run (time-averaged 1.30*D, ~30% hotter than the
#           SGDR exploration average, without ever exceeding the survived
#           1.65*D ceiling) -> the proven straight linear tail landing
#           exactly on gamma_min at the last step.
#   alpha — DECOUPLED dynamic penalty (§7.9), replacing both the 1/lr
#           coupling and the burst machinery: a smooth polynomial creep
#           alpha0*(0.4 -> 1.2) across the hold, so violation debt is
#           metered continuously instead of repaid in spikes and the hold
#           never needs a cold repair window. The PROVEN 5/5-feasible
#           endgame is preserved verbatim: logistic ramp to the bounded
#           6*alpha0 ALM plateau just after the hold ends, then the
#           cubic-delayed geometric climb from 78% landing on the terminal
#           5*alpha0*D/gamma_min feasibility spike.
#   betas — one-cycle beta1 anti-correlated with lr (§2, untried): low
#           momentum 0.06 while steps are huge, rising to 0.14 during the
#           cool-down so accumulated constraint gradient acts as an implicit
#           ALM multiplier (menu bet 4) while alpha is still only moderate,
#           then the proven gated drop to 0.02 inside the terminal spike.
#           beta2 keeps the proven 0.2 -> 0.9 transition, aligned with the
#           hold end.
_F_WARM = 0.04        # linear lr warmup; slightly longer since the hold is hot
_F_HOLD = 0.60        # stable phase ends here; linear decay to gamma_min at 100%
_HI0 = 1.45           # hold start, in units of D
_HI1 = 1.15           # hold end, in units of D; the linear tail starts here
_A_LO = 0.4           # penalty creep start, in alpha0 units
_A_CREEP = 2.0        # creep = _A_LO*(1 + _A_CREEP*fc): 0.4 -> 1.2 over the hold
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.64      # logistic alpha ramp centered just after the hold ends
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the hold end
_B2_WIDTH = 0.05
_B1_HOT = 0.06        # low momentum during the hot hold (one-cycle)
_B1_ALM = 0.14        # raised momentum in the cool-down: implicit multiplier
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_UP_CENTER = 0.62  # momentum rise as the cool-down begins
_B1_UP_WIDTH = 0.05
_B1_DN_CENTER = 0.88  # proven gated drop inside the terminal spike
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted WSD hold -> linear tail to gamma_min ---
    # fc freezes at 1 past _F_HOLD, so the tail starts exactly from _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    lr_hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: decoupled polynomial creep -> plateau -> terminal climb ---
    creep = _A_LO * (1.0 + _A_CREEP * fc)                     # 0.4 -> 1.2, frozen after hold
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = creep + (_A_PLAT - creep) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 rise, proven terminal gate; proven beta2 ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_HOT + (_B1_ALM - _B1_HOT) * b1_up
    b1_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1_dn

    return lr, alpha, beta1, beta2