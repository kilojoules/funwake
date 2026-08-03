import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): NO cycles and NO
# bursts anywhere. This is the frontier row of the prior-art menu that the
# whole lineage has skipped: a WSD/one-cycle lr (warmup -> long tilted HOLD
# near c*D -> linear cool-down, prior-art §6/§2) paired with an
# EPSILON-CONSTRAINED SHRINKING-TOLERANCE alpha (§7.9): alpha is a single
# smooth law alpha = GAIN*alpha0*D/gamma(t), where gamma(t) is an enforced
# violation band that contracts log-linearly from ~2.5*D down to exactly
# gamma_min at the last step.
#
#   lr    — 3% warmup, then a SUSTAINED tilted hold 1.5*D -> 1.0*D across the
#           first 58% (mean ~1.25*D: hotter and longer total exploration than
#           the best's cosine-restart average ~1.0*D, per the parent
#           guidance), then the proven straight linear tail landing exactly
#           on gamma_min at the final step. No restarts, no troughs.
#   alpha — ADMM-style CONSTANT moderate penalty (~2*alpha0) for the whole
#           hot phase (untried menu bet: constant moderate penalty instead of
#           floors/bursts), then the band contraction takes over: a delayed
#           (from 52%), strongly back-loaded (quartic) log-space contraction
#           of gamma(t) drives alpha smoothly and monotonically up, passing
#           the old plateau scale mid-polish and landing EXACTLY on the
#           proven 5/5-seed-feasible terminal 5*alpha0*D/gamma_min. One law
#           replaces floor+bursts+logistic+geometric climb; the terminal
#           feasibility restoration is preserved and slightly strengthened.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition at cool-down start.
#           beta1 tries the untried Sutskever/momentum-as-ALM ramp: 0.1 while
#           hot, RISING to 0.4 during the contraction phase so momentum
#           integrates the persistent constraint gradient like an augmented-
#           Lagrangian multiplier (letting the still-moderate alpha enforce),
#           then the proven gate down to 0.02 for the terminal spike so
#           momentum never carries turbines back out at the end.
_F_WARM = 0.03        # linear lr warmup (proven)
_F_COOL = 0.58        # hold ends here; linear decay to gamma_min at 100%
_HI = 1.5             # hold entry lr, units of D
_LO_HOLD = 1.0        # hold exit lr, units of D (tail starts here)
_G_HI = 2.5           # initial tolerance band, units of D (=> alpha ~2*alpha0)
_GAIN = 5.0           # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_F_DELAY = 0.52       # band contraction starts here (delayed ramp)
_POW = 4.0            # quartic back-loading of the contraction
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with cool-down start (proven)
_B2_WIDTH = 0.05
_B1_HOT = 0.1         # native momentum during exploration
_B1_ALM = 0.4         # momentum-as-multiplier during the contraction phase
_B1_UP_CENTER = 0.63
_B1_UP_WIDTH = 0.05
_B1_TERM = 0.02       # near-zero momentum in the terminal spike (proven)
_B1_DN_CENTER = 0.88
_B1_DN_WIDTH = 0.025


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hold -> linear tail to gamma_min ---
    h = jnp.clip(frac / _F_COOL, 0.0, 1.0)                    # progress through the hold
    lr_hold = (_HI + (_LO_HOLD - _HI) * h) * Dj               # 1.5*D -> 1.0*D, then frozen
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: shrinking tolerance band gamma(t): ~2.5*D -> gamma_min ---
    g_hi = jnp.maximum(_G_HI * Dj, jnp.asarray(min_spacing) * 1.0)
    s = jnp.clip((frac - _F_DELAY) / (1.0 - _F_DELAY), 0.0, 1.0) ** _POW
    log_gamma = (1.0 - s) * jnp.log(g_hi) + s * jnp.log(gmin)  # log-linear contraction
    alpha = _GAIN * alpha0 * Dj / jnp.exp(log_gamma)           # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; beta1 up (ALM) then gated down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    r_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b_mid = _B1_HOT + (_B1_ALM - _B1_HOT) * r_up
    r_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b_mid + (_B1_TERM - b_mid) * r_dn

    return lr, alpha, beta1, beta2