import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst SGDR best (+0.0533%): drop the
# restart machinery entirely and combine the TWO remaining untried menu bets
# into one clean law — a WSD/one-cycle TRAPEZOID lr (prior-art §6/§2: "hold
# near c*D, then near-linear cool-down beats cosine/product decay") paired
# with an ADMM-STYLE CONSTANT MODERATE PENALTY during exploration that hands
# off to an EPSILON-CONSTRAINED SHRINKING-TOLERANCE alpha (§7.9): the
# enforced violation band contracts geometrically and reaches gamma_min only
# at the very end.
#
#   lr    — 3% linear warmup (proven) -> constant hot HOLD at 1.35*D for the
#           first 55% of the run. The SGDR parent spends much of its
#           exploration budget in 0.65*D troughs; the trapezoid spends ALL of
#           it hot, which is the "higher/longer early lr" the guidance asks
#           for, delivered structurally rather than as another peak tweak.
#           Then the proven straight linear tail lands exactly on gamma_min
#           at the last step — 45% of the run for annealing.
#   alpha — DECOUPLED from lr throughout. During the hold it is a constant
#           moderate 1.0*alpha0 (ADMM logic: a fixed moderate penalty plus
#           momentum acting as an implicit multiplier keeps violation debt
#           bounded, no bursts needed). From 55% a single geometric
#           epsilon-contraction takes over: alpha rises in log-space with a
#           back-loaded exponent, passing ~parent-plateau strength (~6*alpha0)
#           mid-tail and landing on the proven 5/5-seed-feasible terminal
#           5*alpha0*D/gamma_min at the final step. One smooth law replaces
#           floor + bursts + logistic ramp + plateau + terminal spike, while
#           preserving the terminal feasibility restoration exactly.
#   betas — the proven transitions, realigned to the single phase boundary:
#           beta2 0.2 -> 0.9 at the cool-down start (adaptive scaling absorbs
#           the growing alpha*constraint curvature), beta1 0.1 -> 0.02 late
#           so momentum cannot carry turbines back across the boundary while
#           the contraction squeezes the band shut.
_F_WARM = 0.03      # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55      # hot hold ends here; linear decay to gamma_min at 100%
_LR_HI = 1.35       # sustained exploration lr, in units of D
_A_EXPL = 1.0       # ADMM-style constant moderate penalty, in alpha0 units
_POW = 2.5          # back-loads the epsilon-contraction exponent
_TERM_GAIN = 5.0    # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2        # native beta2 while exploring
_B2_HI = 0.9        # adaptive scaling for the contraction phase
_B2_CENTER = 0.55   # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1        # native momentum while exploring and polishing
_B1_LO = 0.02       # near-zero momentum while the band squeezes shut
_B1_CENTER = 0.85   # slightly earlier than parent — alpha grows earlier here
_B1_WIDTH = 0.04


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> constant hot hold -> linear tail to gamma_min ---
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_LR_HI * Dj - gmin) * (1.0 - p)   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)            # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate hold -> geometric epsilon-contraction ---
    # s runs 0 -> 1 over the tail with back-loaded shape; alpha interpolates
    # in log-space from _A_EXPL*alpha0 to the proven terminal
    # _TERM_GAIN*alpha0*D/gamma_min, i.e. the tolerated violation band
    # contracts geometrically and closes to gamma_min only at the last step.
    s = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_EXPL), 1.0))
    alpha = alpha0 * _A_EXPL * jnp.exp(s * log_term)

    # --- betas: proven transitions realigned to the single phase boundary ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2