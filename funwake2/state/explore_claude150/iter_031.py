import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lineage is
# saturated with CYCLIC machinery (SGDR restarts, alpha bursts) — this attempt
# removes every cycle and bets on the two menu directions never tried anywhere:
# a WSD lr (warmup -> flat HOT HOLD -> straight linear cool-down, §2/§6) and an
# ADMM-STYLE CONSTANT MODERATE PENALTY through the whole exploration phase.
# Rationale: what the +0.0533% parent actually bought with its cycles is total
# hot-phase displacement; a flat hold at 1.30*D over 57% of the run delivers
# MORE integrated hot lr than the parent's decaying 1.65->1.05*D peaks over
# 62% (mean ~1.0*D), with none of the trough time wasted at 0.65*D. The
# violation debt of the long hot hold is bounded by the constant penalty and
# repaid once by the PROVEN terminal machinery, kept bit-for-bit.
#
#   lr    — 3% linear warmup (proven) -> constant 1.30*D until 60% -> pure
#           linear tail landing exactly on gamma_min at the last step (the
#           proven tail shape, started hotter and only slightly earlier).
#   alpha — ADMM surrogate: constant 1.2*alpha0 for the entire hold (balanced
#           objective/constraint pressure, no ramp, no bursts — open-loop
#           stand-in for a fixed ADMM rho), then the PROVEN endgame unchanged:
#           logistic ramp to the bounded 6*alpha0 ALM plateau centered at 66%,
#           and the cubic-delayed geometric climb from 78% landing on the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   beta1 — one-cycle policy (menu bet: anti-correlate momentum with lr, §2),
#           untried in the lineage: LOW momentum (0.05) while steps are huge
#           so overshoot never carries turbines across the boundary, rising
#           logistically to 0.40 as the lr tail shrinks — with the constant
#           moderate alpha, the accumulating first moment integrates the
#           constraint gradient like a running ALM multiplier estimate — then
#           the proven terminal gate drops it to 0.02 for the alpha spike
#           (crossing happens while alpha is still near-plateau: at 88% the
#           cubic delay has released <10% of the geometric climb).
#   beta2 — proven transition kept: 0.2 -> 0.9 logistic at the cool-down start.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD_END = 0.60    # flat hot hold ends here; linear tail to gamma_min at 100%
_HOLD = 1.30          # hold lr in units of D — hotter in integral than any parent
_A_CONST = 1.2        # ADMM-style constant moderate penalty, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum while lr is at the hot hold
_B1_CRUISE = 0.40     # one-cycle rise as lr decays (implicit ALM multiplier)
_B1_CENTER = 0.60     # momentum rise anti-correlated with the lr tail
_B1_WIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike (proven)
_B1_GATE = 0.88       # proven terminal gate position
_B1_GWIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hot hold at 1.30*D -> straight linear tail ---
    # p = 0 for the whole hold, so lr_env is exactly _HOLD*D there; p hits 1 at
    # the last step, so the tail lands exactly on gamma_min.
    p = jnp.clip((frac - _F_HOLD_END) / (1.0 - _F_HOLD_END), 0.0, 1.0)
    lr_env = gmin + (_HOLD * Dj - gmin) * (1.0 - p)
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant -> bounded plateau -> proven terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; one-cycle beta1 with terminal gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    b1_cruise = _B1_HOT + (_B1_CRUISE - _B1_HOT) * rise       # momentum up as lr goes down
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE) / _B1_GWIDTH))
    beta1 = b1_cruise + (_B1_LO - b1_cruise) * gate           # cut for the alpha spike

    return lr, alpha, beta1, beta2