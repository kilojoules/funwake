import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst-cycle best (+0.0533%): the two prior-art bets
# the lineage has NOT tried — an ADMM-STYLE CONSTANT MODERATE PENALTY (§7.9 /
# menu "ADMM-style constant") and a WSD/ONE-CYCLE lr (§2/§6: warmup -> long
# STABLE hot plateau -> single near-linear cool-down), replacing both the
# SGDR restarts and the floor/burst/logistic/plateau alpha machinery.
#
#   lr    — 3% linear warmup, then a SUSTAINED hot plateau at 1.30*D for
#           ~half the run (more integrated exploration heat than any restart
#           schedule dared: the best only touched 1.65*D briefly and averaged
#           ~1.0*D), then one straight linear cool-down landing exactly on
#           gamma_min at the last step (the WSD bet: hold near c*D, cool
#           linearly — no cosine, no restarts).
#   alpha — fully DECOUPLED from lr. A constant ADMM-like 1.6*alpha0 through
#           the entire hot phase (uniform moderate pressure keeps violation
#           debt bounded at all times, so no restoration bursts are needed),
#           then a SINGLE epsilon-constrained geometric contraction (§7.9):
#           alpha = 1.6*alpha0 * (5*D/(1.6*gamma_min))**s(t), i.e. the
#           enforced tolerance band eps ~ 1/alpha shrinks geometrically from
#           ~D down to gamma_min/5, reaching the proven 5/5-seed-feasible
#           terminal value 5*alpha0*D/gamma_min exactly at the last step.
#           Cubic back-loading (s = ramp**3 from 58%) keeps the climb late,
#           and by 78-90% it sits at or above the proven plateau+spike path,
#           so the terminal feasibility restoration is PRESERVED (stronger,
#           smoother, one law instead of three pieces).
#   betas — one-cycle anti-correlation (menu bet 1) + phase transition
#           (bet 4): beta1 LOW (0.05) while lr is hot so momentum never
#           compounds the huge steps, RISES to 0.20 in the cool-down where
#           momentum acts as an implicit ALM multiplier helping the moderate
#           alpha enforce constraints, then gates down to the proven 0.02
#           during the terminal contraction; beta2 does the proven
#           0.2 -> 0.9 logistic switch at the cool-down boundary.
_F_WARM = 0.03      # linear lr warmup fraction
_F_DECAY = 0.52     # stable hot plateau ends here; linear tail to gamma_min
_LR_HI = 1.30       # plateau lr in units of D — hot AND long (the AEP push)
_A_ADMM = 1.6       # constant moderate penalty during the hot phase, in alpha0
_F_ALPHA = 0.58     # geometric epsilon-contraction starts here
_POW = 3.0          # cubic back-loading of the contraction (proven shape)
_TERM_GAIN = 5.0    # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B1_HOT = 0.05      # low momentum under the hot plateau
_B1_MID = 0.20      # implicit-ALM momentum during the cool-down
_B1_LO = 0.02       # near-zero momentum in the terminal contraction (proven)
_B1_UP_CENTER = 0.55
_B1_UP_WIDTH = 0.04
_B1_DN_CENTER = 0.88
_B1_DN_WIDTH = 0.03
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.52   # beta2 switch aligned with the cool-down start
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> stable hot plateau -> single linear tail ---
    lr_hi = _LR_HI * Dj
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (lr_hi - gmin) * (1.0 - p)         # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: ADMM constant -> single geometric epsilon-contraction ---
    # ratio contains no traced quantities inside the log argument besides
    # D and gamma_min (alpha0 cancels), so the law is pure open-loop.
    s = jnp.clip((frac - _F_ALPHA) / (1.0 - _F_ALPHA), 0.0, 1.0) ** _POW
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_ADMM), 1.0))
    alpha = _A_ADMM * alpha0 * jnp.exp(s * log_ratio)  # ends at 5*alpha0*D/gmin

    # --- betas: anti-correlated one-cycle beta1 + proven beta2 switch ---
    r_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    r_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * r_up
    beta1 = b1_mid + (_B1_LO - b1_mid) * r_dn
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2