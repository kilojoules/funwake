import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top/three-notch best (+0.0577%): the periodic
# repair machinery is REMOVED entirely and replaced by a BIG-VALLEY "W"
# schedule — one SUPERHEAT epoch, one GREAT REPAIR, one cooler second
# exploration, then the proven WSD tail. Where the parent interleaves three
# small lr notches with three modest alpha bursts (repairing debt it keeps
# re-creating at full heat), this schedule batches the work: run hotter than
# any prior attempt (a sustained 1.8*D hold, not a momentary peak) while
# feasibility debt accrues cheaply, then pay it ALL down once, mid-run, in a
# single deep synchronized repair valley (lr 0.25*D, alpha 10*alpha0 for ~6%
# of the run — more consecutive repair steps than the parent's three notches
# combined), and spend the remaining exploration budget inside the repaired
# feasible region at a moderate 1.0*D reheat. All waveforms are
# piecewise-linear via jnp.interp (traceable; constant breakpoints), not
# sin-power pulses.
#
#   lr    — warmup to a 1.8*D -> 1.45*D superheat hold (0-44%) -> single deep
#           repair valley at 0.25*D (50-56%) -> reheat to 1.0*D (63%) easing
#           to 0.85*D at 72% -> proven straight linear tail landing exactly
#           on gamma_min at the last step.
#   alpha — DECOUPLED from lr throughout: graduated floor 0.3 -> 0.8*alpha0
#           during superheat (dynamic-penalty ramp, prior-art §7.9), one
#           10*alpha0 restoration burst filling the lr valley, relaxation to
#           1.5*alpha0 for the reheat, delayed ramp to the proven bounded
#           6*alpha0 ALM plateau by 70%, and the proven cubic-delayed
#           geometric climb from 80% to the terminal 5*alpha0*D/gamma_min
#           spike (the 5/5-seed-feasible endgame, kept intact).
#   betas — proven transitions: beta2 0.2 -> 0.9 at the cool-down; beta1 0.1
#           with a dip to 0.04 across the repair valley (momentum must not
#           drag turbines back over the boundary mid-repair) and the gated
#           drop to 0.02 during the terminal alpha spike.
_F_COOL = 0.72        # exploration ends; linear decay to gamma_min at 100%
_LR_X = (0.0, 0.04, 0.44, 0.50, 0.56, 0.63, _F_COOL)   # lr breakpoints (frac)
_LR_Y = (0.12, 1.80, 1.45, 0.25, 0.25, 1.00, 0.85)     # lr levels (units of D)
_A_X = (0.0, 0.44, 0.50, 0.56, 0.62, 0.70, 1.0)        # alpha breakpoints
_A_Y = (0.30, 0.80, 10.0, 10.0, 1.50, 6.00, 6.00)      # alpha (units of alpha0)
_A_PLAT = 6.0         # bounded ALM plateau the terminal climb launches from
_F_TERM = 0.80        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = _F_COOL  # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_VALLEY = 0.04     # reduced momentum across the Great Repair
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_VCENTER = 0.53    # gaussian dip centered on the repair valley
_B1_VWIDTH = 0.05
_B1_CENTER = 0.90
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: superheat hold -> Great Repair valley -> reheat -> linear tail ---
    # The exploration waveform is frozen at its 0.85*D end value past _F_COOL;
    # the proven linear blend then lands exactly on gamma_min at the last step.
    fw = jnp.clip(frac, 0.0, _F_COOL)
    lr_hold = jnp.interp(fw, jnp.array(_LR_X), jnp.array(_LR_Y)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr = gmin + (lr_hold - gmin) * (1.0 - p)

    # --- alpha: graduated floor -> single big burst -> plateau -> climb ---
    # The burst fills the lr valley exactly (breakpoints 0.50-0.56 on both
    # waveforms): huge penalty while the steps are small and surgical. The
    # delayed ramp to the bounded plateau and the terminal spike are the
    # proven feasibility endgame, unchanged.
    alpha_units = jnp.interp(frac, jnp.array(_A_X), jnp.array(_A_Y))
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)     # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + valley beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    dip = jnp.exp(-((frac - _B1_VCENTER) / _B1_VWIDTH) ** 2)
    b1_exp = _B1_HI - (_B1_HI - _B1_VALLEY) * dip            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2