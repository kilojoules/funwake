import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the SGDR restart
# machinery (decaying peaks, troughs, per-cycle bursts) is REMOVED entirely and
# replaced by the two top untried menu rows, executed together:
#
#   lr    — WSD / ONE-CYCLE (prior-art §2/§6, the menu's #1 untried lr bet:
#           "hold near c*D, then (near-)linear cool-down beats cosine/product
#           decay"). 3% linear warmup -> a long STABLE HOT HOLD at 1.30*D
#           until 55% -> a single straight linear tail landing exactly on
#           gamma_min at the last step. No troughs: integrated exploration
#           heat is ~25% ABOVE what the best schedule's decaying-peak cycles
#           ever delivered (its cycle-average lr is ~1.0*D), satisfying the
#           parent guidance "higher/longer lr peak early", while the earlier
#           cool-down start (55% vs 62%) buys back a longer polish to pay for
#           the extra heat.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY during the hold (an untried
#           search-state direction). The burst best applies a time-averaged
#           multiplier of ~1.6*alpha0 across its exploration phase, but
#           delivers it in violent pulses; here the SAME average is applied
#           as a flat 1.6*alpha0 for the whole hold — constraint pressure
#           never vanishes (so violation debt stays bounded without bursts)
#           and never spikes against a hot lr (so basin hops are never
#           disrupted mid-flight). The proven feasibility endgame is kept
#           VERBATIM: logistic ramp to the bounded 6*alpha0 ALM plateau just
#           after the cool-down starts, then the cubic-delayed geometric
#           climb from 78% landing on the 5/5-seed-feasible terminal
#           5*alpha0*D/gamma_min.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition, re-aligned with the
#           new cool-down boundary. beta1 realizes menu bet 4 / one-cycle
#           momentum, untried in the lineage: ANTI-CORRELATED with lr — 0.1
#           (native) while hot, ramping to 0.35 as lr decays so momentum acts
#           as an implicit ALM multiplier and averages wake-gradient noise
#           during polish — then the proven gate down to 0.02 inside the
#           terminal alpha spike so momentum never fights the restoration.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.55        # hold ends here; single linear decay to gamma_min at 100%
_HOLD = 1.30          # stable hot-hold lr, in units of D
_A_EXP = 1.6          # ADMM-style constant penalty during the hold, alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.62      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.1         # native momentum while the lr is hot
_B1_POLISH = 0.35     # one-cycle: momentum rises as lr falls
_B1_UP_CENTER = 0.60  # momentum ramp sits just inside the cool-down
_B1_UP_WIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike (proven)
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> stable hot hold at _HOLD*D -> straight linear tail ---
    # p = 0 for the whole hold (flat plateau); the tail lands exactly on
    # gamma_min at the final step.
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (_HOLD * Dj - gmin) * (1.0 - p)
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> bounded plateau -> terminal geometric climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_EXP + (_A_PLAT - _A_EXP) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition + one-cycle beta1 (up in polish, gated
    # down inside the terminal spike) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_HOT + (_B1_POLISH - _B1_HOT) * up
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r

    return lr, alpha, beta1, beta2