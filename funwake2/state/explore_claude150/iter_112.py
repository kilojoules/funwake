import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top+repair-notch best (+0.0577%): the entire
# mid-run oscillatory machinery (lr notches + synchronized alpha bursts +
# plateau + late spike) is REPLACED by the two prior-art directions the menu
# lists as untried and the search log confirms unexplored:
#
#   (1) ADMM-STYLE CONSTANT MODERATE PENALTY (§7.2/§7.9): during the whole
#       exploration phase alpha is a flat 1.2*alpha0 — no floor, no bursts.
#       Constraint debt is paid CONTINUOUSLY at moderate pressure instead of
#       in pulsed repair windows, so the lr never has to notch down: the hold
#       runs at FULL heat, 100% duty cycle (vs the parent's 80%), and hotter
#       (1.6*D constant vs a 1.5->1.1*D decaying envelope) — the requested
#       longer/hotter exploration peak.
#   (2) EPSILON-CONSTRAINT CONTRACTING BAND (§7.9): after the hold, lr and
#       the enforced violation tolerance contract GEOMETRICALLY together —
#       lr decays log-linearly from 1.6*D to exactly gamma_min while alpha
#       climbs log-linearly (back-loaded, power 2.5) from 1.2*alpha0 to a
#       terminal 6*alpha0*D/gamma_min. The tolerance band shrinks to
#       gamma_min only at the very end; the last ~10% is automatically a
#       feasibility-restoration spike (huge alpha, metre-scale lr), i.e. the
#       proven terminal restoration emerges from the contraction law itself.
#       The geometric tail also spends far more steps at small lr than the
#       parent's linear tail — a long fine-polish runway.
#
#   betas — beta2 keeps the proven 0.2 -> 0.9 switch at the end of the hold.
#           beta1 implements the menu's untried ONE-CYCLE ANTI-CORRELATION:
#           0.1 at full heat, rising to 0.3 in mid cool-down (momentum acts
#           as an averaged ALM multiplier while alpha climbs), then the
#           proven gated drop to 0.02 inside the terminal spike so momentum
#           cannot carry turbines back across the boundary.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.60        # full-heat hold ends here; geometric contraction after
_HI = 1.6             # hold lr in units of D — hotter and 100% duty cycle
_A_EXP = 1.2          # ADMM-style constant exploration penalty, alpha0 units
_A_POW = 2.5          # back-loading of the log-linear alpha climb
_TERM_GAIN = 6.0      # terminal alpha = 6*alpha0*D/gamma_min
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60     # beta2 switch aligned with the hold end
_B2_WIDTH = 0.05
_B1_BASE = 0.1        # low momentum at high lr (one-cycle)
_B1_MID = 0.3         # raised momentum in mid cool-down (momentum-as-ALM)
_B1_LO = 0.02         # near-zero momentum during the terminal spike (proven)
_B1_UP_CENTER = 0.70
_B1_UP_WIDTH = 0.04
_B1_DN_CENTER = 0.90
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat 1.6*D hold -> geometric contraction to gamma_min ---
    # q is the contraction progress; lr is log-linear in time, landing exactly
    # on gamma_min at the final step.
    hold = _HI * Dj
    q = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = hold * jnp.exp(q * jnp.log(gmin / hold))
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> contracting-band geometric climb ---
    # Flat 1.2*alpha0 through the hold; then log-linear growth in warped time
    # w = q^2.5, continuous at the hold end, exploding only in the final ~10%
    # into the terminal restoration spike at 6*alpha0*D/gamma_min.
    w = q ** _A_POW
    log_gain = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_EXP), 1.0))
    alpha = alpha0 * _A_EXP * jnp.exp(w * log_gain)

    # --- betas: proven beta2 switch + one-cycle beta1 arc with gated drop ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    b1_mid = _B1_BASE + (_B1_MID - _B1_BASE) * up
    beta1 = b1_mid + (_B1_LO - b1_mid) * dn

    return lr, alpha, beta1, beta2