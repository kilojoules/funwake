import jax.numpy as jnp

# STRUCTURALLY NEW vs both the 3-restart anti-phased best (+0.0533%) and the
# 8-cycle fast-CLR parent (+0.0467%): this schedule has NO CYCLES AT ALL.
# Every recent attempt has been a variation on kick-and-repair oscillations,
# and all of them plateaued at or below the best — so this takes the two
# untried search-state directions simultaneously:
#
#   lr    — WSD / trapezoid "hold-then-cool" (prior-art §6, one-cycle §2):
#           3% linear warmup -> a SUSTAINED HOT HOLD at 1.5*D for ~half the
#           run -> one long straight linear tail landing exactly on gamma_min
#           at the last step. No parent ever HELD a hot lr: the cyclic
#           schedules only touch their peaks momentarily, so their integrated
#           time above 1.3*D is small. Here the layout basin-hops
#           continuously for 47% of the run at a temperature the restarts
#           only ever visited. The min_spacing guard from gen 62 is kept: the
#           hold is capped at 0.9*min_spacing so a tight-spacing farm can
#           never be blown through in one step.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (the explicitly-untried
#           direction). During the entire hold, alpha is a flat 2.5*alpha0 —
#           no bursts, no coupling to lr. The ADMM mechanism: a fixed,
#           moderate quadratic weight keeps violation BOUNDED (turbines orbit
#           the boundary rather than fleeing it) while never dominating the
#           AEP gradient, so exploration is continuous instead of being
#           interrupted by repair pulses. Feasibility is then collected by
#           the PROVEN endgame, preserved verbatim from the lineage: logistic
#           ramp to the bounded 6*alpha0 ALM plateau just after cool-down
#           starts, then the cubic-delayed geometric climb landing on the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   beta1 — one-cycle momentum (menu bet 4, §2): ANTI-CORRELATED with lr.
#           Low momentum (0.05) during the hot hold so momentum never
#           compounds a boundary overshoot at high temperature, rising to the
#           native 0.1 as lr cools (fast coherent descent into the chosen
#           basin), then the proven gate down to 0.02 through the terminal
#           alpha spike.
#   beta2 — proven 0.2 -> 0.9 logistic aligned with the cool-down start.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_DEC = 0.50         # hold ends here; single linear decay to gamma_min at 100%
_HI = 1.50            # sustained hold temperature, in units of D
_MS_CAP = 0.9         # hold lr never exceeds 0.9 * min_spacing (proven guard)
_A_HOLD = 2.5         # ADMM-style flat penalty during the hold, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.56      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.50     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.05        # low momentum while lr is held hot (one-cycle)
_B1_HI = 0.1          # native momentum during the cool-down / polish
_B1_CENTER_UP = 0.50  # momentum rises as lr falls (anti-correlation)
_B1_WIDTH_UP = 0.05
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> sustained hot hold -> single linear tail to gamma_min ---
    hold_lr = jnp.minimum(_HI * Dj, _MS_CAP * ms)             # spacing-aware hold
    p = jnp.clip((frac - _F_DEC) / (1.0 - _F_DEC), 0.0, 1.0)  # 0 during hold
    lr_env = gmin + (hold_lr - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM flat hold -> bounded ALM plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_HOLD + (_A_PLAT - _A_HOLD) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle beta1 anti-correlated with lr + proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER_UP) / _B1_WIDTH_UP))
    b1_mid = _B1_HOT + (_B1_HI - _B1_HOT) * b1_up             # rises as lr cools
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r                  # gated down for the spike

    return lr, alpha, beta1, beta2