import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst SGDR best (+0.0533%): no restarts,
# no cycles, no bursts. This is the strongest UNTRIED lr bet in the prior-art
# menu (§6/§2): a WSD / trapezoid "one-cycle" — warmup -> long HOT STABLE HOLD
# -> single long linear cool-down to gamma_min. Every lineage member oscillates
# lr; here exploration heat is delivered as a sustained plateau instead. The
# time-integral of lr is HIGHER than any parent (hold at 1.30*D for half the
# run vs. peaks that only touch 1.65*D momentarily), which is the "higher/
# longer peak" the guidance asks for, delivered by shape rather than amplitude.
#
#   lr    — 4% linear warmup to 1.30*D, hold flat to 55%, then one straight
#           linear tail landing exactly on gamma_min at the last step (the
#           proven landing). Prior-art hypothesis §6: hold near c*D + linear
#           cool-down beats cosine/product/restart decay.
#   alpha — DECOUPLED from lr throughout. Exploration uses the §7.9 dynamic
#           penalty alpha0*(1+C*t)^p — a gentle graduated climb (0.4 -> ~1.2
#           alpha0 by 55%) instead of a flat floor or bursts, so violation
#           debt is metered continuously while the layout is hot. A logistic
#           ramp centered just after the cool-down start lifts it onto the
#           proven bounded 6*alpha0 ALM plateau, and the proven cubic-delayed
#           geometric climb from 78% lands it on the 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min. The terminal restoration is kept
#           bit-for-bit — feasibility insurance is untouched.
#   betas — one-cycle momentum anti-correlation (§2/§4, untried): LOW beta1
#           (0.08) while lr is hot, RISING to 0.30 during the cool-down where
#           momentum acts as an implicit ALM multiplier behind the bounded
#           plateau (menu bet 4), then the proven terminal gate drops it to
#           0.02 so momentum never re-violates during the final spike. beta2
#           keeps the proven 0.2 -> 0.9 transition at the cool-down start.
_F_WARM = 0.04        # linear lr warmup
_F_DECAY = 0.55       # hold ends; linear decay to gamma_min at 100%
_LR_HOLD = 1.30       # sustained exploration lr, in units of D
_A_LO = 0.4           # exploration penalty at t=0, in alpha0 units
_A_C = 1.35           # dynamic-penalty growth rate  (alpha ~ (1+C*t)^p, §7.9)
_A_P = 2.0            # dynamic-penalty exponent
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.62      # logistic ramp onto the plateau, just after decay start
_A_WIDTH = 0.05
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.08        # low momentum while lr is hot (one-cycle anti-phase)
_B1_MID = 0.30        # raised momentum = implicit ALM multiplier in cool-down
_B1_RISE_CENTER = 0.62
_B1_RISE_WIDTH = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_GATE_CENTER = 0.88
_B1_GATE_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hot hold -> single linear tail to gamma_min ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_hold = _LR_HOLD * Dj
    lr = (gmin + (lr_hold - gmin) * (1.0 - p)) * warm   # exact landing on gamma_min

    # --- alpha: graduated dynamic penalty -> bounded plateau -> terminal climb ---
    a_dyn = _A_LO * (1.0 + _A_C * frac) ** _A_P          # §7.9 continuous metering
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = a_dyn + (_A_PLAT - a_dyn) * ramp       # smooth blend onto plateau
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 (low hot -> raised mid -> gated terminal) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_RISE_CENTER) / _B1_RISE_WIDTH))
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * b1_rise
    b1_gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE_CENTER) / _B1_GATE_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1_gate

    return lr, alpha, beta1, beta2