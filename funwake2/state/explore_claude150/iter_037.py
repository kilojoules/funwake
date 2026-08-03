import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): the lineage has been
# all-cyclic (restarts + anti-phased bursts) and every recent cyclic tweak
# plateaued below the best. This attempt is the OPPOSITE shape, combining the
# two prior-art menu rows never yet realized anywhere in the lineage:
#
#   lr    — WSD / one-cycle TRAPEZOID (prior-art §2/§6): 3% linear warmup,
#           then a long HOT HOLD with a gentle tilt (1.35*D -> 1.15*D) over
#           the first 60% — no restarts, no troughs — then the proven
#           straight linear cool-down landing exactly on gamma_min at the
#           last step. Total exploration heat is far ABOVE the cyclic best
#           (its cycle-average lr was ~1.0*D; the hold never drops below
#           1.15*D), delivering the requested "hotter/longer" exploration
#           without transient extremes, and the §6 hypothesis — hold near
#           c*D then near-linear decay beats cosine machinery — is finally
#           tested directly.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (prior-art §7.2, listed as
#           untried): a flat 1.5*alpha0 through the entire hold. No floor, no
#           bursts, no coupling to lr — violations stay bounded the whole
#           time instead of oscillating, so the hot phase optimizes a FIXED
#           relaxed problem (well-conditioned, no moving target). The proven
#           5/5-feasible endgame is preserved verbatim: logistic ramp
#           (center 0.66) to the bounded 6*alpha0 ALM plateau, then the
#           cubic-delayed geometric climb from 78% landing on the terminal
#           5*alpha0*D/gamma_min feasibility spike.
#   betas — proven beta2 0.2 -> 0.9 transition at the cool-down knee, plus
#           the one-cycle momentum bet (§2/§4, untried): beta1 RISES
#           0.1 -> 0.25 as lr falls — momentum anti-correlated with lr acts
#           as an implicit ALM multiplier, letting the moderate plateau
#           enforce constraints — then the proven gate drops it to 0.02
#           before the terminal alpha spike so momentum never fights the
#           final restoration.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.60        # hot hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.35           # hold-start lr, in units of D
_HI1 = 1.15           # hold-end lr — the linear tail starts from here
_A_CONST = 1.5        # ADMM-style constant penalty during the hold, alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after the knee (proven)
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down knee
_B2_WIDTH = 0.05
_B1_EXPLORE = 0.1     # native momentum during the hot hold
_B1_DECAY = 0.25      # raised momentum during the cool-down (implicit ALM)
_B1_UP_CENTER = 0.64  # momentum rise just after the knee
_B1_UP_WIDTH = 0.05
_B1_TERM = 0.02       # near-zero momentum during the terminal alpha spike
_B1_DN_CENTER = 0.88  # proven terminal gate
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> straight linear tail ---
    # h freezes at 1 past _F_HOLD, so the tail decays from the hold-end level
    # _HI1 * D and lands exactly on gamma_min at the final step.
    h = jnp.clip(frac / _F_HOLD, 0.0, 1.0)
    lr_hold = (_HI0 + (_HI1 - _HI0) * h) * Dj
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: flat ADMM penalty -> bounded plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition + anti-correlated beta1 with gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_EXPLORE + (_B1_DECAY - _B1_EXPLORE) * b1_up  # rise as lr falls
    b1_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_mid + (_B1_TERM - b1_mid) * b1_dn              # gate for the spike

    return lr, alpha, beta1, beta2