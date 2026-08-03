import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the burst/restart
# family is saturating (last 5 attempts all at-or-below best), so this abandons
# lr cycles ENTIRELY and takes the two prior-art bets no lineage member has
# tried — WSD/one-cycle lr (§2/§6: "hold near c*D, then near-linear cool-down
# beats cosine/product decay") crossed with an ADMM-STYLE CONSTANT MODERATE
# PENALTY (§7.2/§10 bet: momentum as implicit ALM multiplier lets a moderate
# alpha enforce constraints).
#
#   lr    — TRAPEZOID, no restarts: 4% linear warmup, then a HOT CONSTANT
#           plateau at 1.15*D held for ~56% of the run. The parent's cosine
#           cycles average only ~1.0*D over exploration and waste half their
#           time near 0.65*D troughs; a flat 1.15*D plateau delivers MORE
#           integrated basin-hopping motion with zero dead time — the
#           "higher/longer peak early" the guidance asks for, realized as
#           duration rather than a taller spike. From 60% the proven straight
#           linear tail lands exactly on gamma_min at the last step.
#   alpha — ADMM-style: a CONSTANT 1.5*alpha0 through the whole hot phase.
#           No floor/burst oscillation: the parent's schedule time-averages
#           ~2*alpha0 anyway but concentrates it in troughs, leaving zero-
#           penalty windows at peak lr where turbines drift far outside; a
#           constant moderate penalty applies the same average force with no
#           such windows, so violation debt never accumulates in the first
#           place. Endgame keeps the PROVEN 5/5-seed-feasible machinery
#           untouched: logistic ramp to the bounded 6*alpha0 ALM plateau at
#           cool-down, then the cubic-delayed geometric climb from 78%
#           landing on the terminal 5*alpha0*D/gamma_min spike.
#   betas — the momentum-as-multiplier half of the ADMM bet: beta1 = 0.35
#           during the hot phase, so the constant constraint gradient
#           accumulates ~1.5x persistent push (an implicit multiplier update)
#           and the moderate alpha suffices; beta1 drops to the native 0.1
#           at cool-down for the polish, then the proven gate to 0.02 in the
#           terminal spike. beta2 keeps the proven 0.2 -> 0.9 transition at
#           cool-down start.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_COOL = 0.60        # plateau ends here; linear decay to gamma_min at 100%
_HI = 1.15            # constant plateau lr, in units of D (hotter than the
                      # parent's ~1.0*D exploration-phase AVERAGE)
_A_CONST = 1.5        # ADMM-style constant penalty through the hot phase,
                      # in alpha0 units (matches the parent's time-average
                      # without its zero-penalty peak-lr windows)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.645     # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.35        # elevated momentum = implicit ALM multiplier (hot phase)
_B1_MID = 0.1         # native momentum for the polish phase
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> constant hot plateau -> straight linear tail ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)               # damps the hot start; lr only
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr = (gmin + (_HI * Dj - gmin) * (1.0 - p)) * warm    # exact landing on gamma_min

    # --- alpha: constant moderate -> logistic plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: momentum-as-multiplier hot phase, then proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * b2r          # shed momentum at cool-down
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r

    return lr, alpha, beta1, beta2