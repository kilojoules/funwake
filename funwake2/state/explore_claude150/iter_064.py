import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lineage has
# only ever explored with an OSCILLATING lr (cosine restarts) and an
# OSCILLATING or ramping alpha. This attempt takes the two strongest untried
# menu bets and combines them:
#
#   lr    — WSD / HOT TRAPEZOID (prior-art §6/§2, flagged the top untried lr
#           idea: "hold near c*D, then (near-)linear cool-down beats
#           cosine/product decay"). Instead of paying for three cosine
#           troughs, ALL exploration budget is spent hot: 3% warmup -> a
#           gently tilted hold (>=1.1*D for 52% of the run — the longest,
#           hottest sustained peak any attempt has run, exactly the "higher/
#           longer lr peak early" the parent guidance asks for) -> a TWO-SLOPE
#           linear cool-down (fast to a 0.30*D knee at 82%, then a gentle
#           final slope landing exactly on gamma_min at the last step, so the
#           polish phase gets far more steps at fine metre scales than the
#           proven single straight tail gave it).
#           The hold height is SPACING-AWARE — built from D plus min_spacing
#           (no attempt has used min_spacing), because a basin hop must move a
#           turbine on the order of the spacing constraint — and clipped into
#           the proven-safe band so an unseen farm cannot push it divergent.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (untried direction, prior-art
#           §7.2): a flat 1.3*alpha0 through the entire hot phase. No floor
#           dips, no bursts — constant bounded pressure keeps violation debt
#           shallow while hinge-type penalty gradients leave feasible turbines
#           completely free, which is what licenses the sustained-hot hold.
#           The PROVEN endgame is preserved verbatim: logistic ramp to the
#           bounded 6*alpha0 ALM plateau at cool-down start, then the
#           cubic-delayed geometric climb from 78% landing on the 5/5-seed-
#           feasible terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at cool-down start;
#           beta1 native 0.1, gated to 0.02 inside the terminal alpha spike so
#           momentum never fights the feasibility restoration.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.55        # hot hold ends here; two-slope decay to gamma_min at 100%
_P_KNEE = 0.60        # knee at 60% of the decay phase (frac = 0.82)
_HI0_D = 0.95         # hold-start height: 0.95*D + 0.30*min_spacing ...
_HI0_S = 0.30
_HI0_MIN = 1.35       # ... clipped to the proven-safe [1.35, 1.75]*D band
_HI0_MAX = 1.75
_HI1_D = 0.72         # hold-end height: 0.72*D + 0.21*min_spacing ...
_HI1_S = 0.21
_HI1_MIN = 1.00       # ... clipped to [1.00, 1.30]*D
_HI1_MAX = 1.30
_KNEE = 0.30          # lr at the decay knee, in units of D
_A_ADMM = 1.3         # constant moderate penalty through the hot phase
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.60      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    msj = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> two-slope linear decay ---
    # Spacing-aware hold heights, clipped into the proven-safe band.
    hi0 = jnp.clip(_HI0_D * Dj + _HI0_S * msj, _HI0_MIN * Dj, _HI0_MAX * Dj)
    hi1 = jnp.clip(_HI1_D * Dj + _HI1_S * msj, _HI1_MIN * Dj, _HI1_MAX * Dj)
    h = jnp.clip(frac / _F_HOLD, 0.0, 1.0)                    # hold progress; freezes at 1
    hold = hi0 + (hi1 - hi0) * h                              # gentle tilt keeps late debt small
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    seg1 = jnp.clip(p / _P_KNEE, 0.0, 1.0)                    # fast slope to the knee
    seg2 = jnp.clip((p - _P_KNEE) / (1.0 - _P_KNEE), 0.0, 1.0)  # gentle final slope
    knee = _KNEE * Dj
    lr_env = hold + (knee - hold) * seg1 + (gmin - knee) * seg2  # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> proven plateau ramp -> proven terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_ADMM + (_A_PLAT - _A_ADMM) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions only ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2