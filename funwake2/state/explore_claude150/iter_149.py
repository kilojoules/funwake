import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top/notch best (+0.0577%): the three narrow
# periodic repair notches are REPLACED by a SINGLE WIDE MID-RUN FEASIBILITY-
# RESTORATION EPOCH — the filter/funnel-method structure from the prior-art
# menu (§7.5) applied as a contiguous phase, not a pulse train. This is a
# genuine phase-architecture change, not a constant tweak:
#
#   SPRINT 1 (3%..40%)  — uninterrupted exploration HOTTER than any prior
#       attempt: hold envelope 1.7*D -> 1.4*D with NO notches and a lower
#       alpha floor (0.35*alpha0). The +0.0533 -> +0.0577 jump came from
#       raising sustained heat; this pushes further and stops paying the
#       notch tax (the parent interrupts full heat 3 times).
#   RESTORATION EPOCH (40%..50%) — one 800-step raised-cosine block: lr sinks
#       to 0.30*D while alpha bursts to 8*alpha0 and beta1 dips to 0.05. A
#       wide epoch can FULLY restore feasibility (narrow notches only nibble
#       at the accumulated debt), so sprint 1 can afford to run hotter.
#   SPRINT 2 (50%..62%) — re-optimize AEP from the near-feasible layout at a
#       moderate 1.15*D -> 1.10*D hold with a FIRMER floor (0.6*alpha0) so
#       the repaired state is not squandered; ends at the proven 1.1*D.
#   ENDGAME (62%..100%) — the 5/5-seed-feasible machinery kept EXACTLY:
#       straight linear lr tail launching from 1.1*D and landing on gamma_min
#       at the last step; logistic alpha ramp centered at 66% to the bounded
#       6*alpha0 ALM plateau; cubic-delayed geometric climb from 78% to the
#       terminal 5*alpha0*D/gamma_min spike; beta2 0.2 -> 0.9 at cool-down;
#       beta1 gated down to 0.02 during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_R_CENTER = 0.45      # center of the single restoration epoch
_R_HALFW = 0.05       # raised-cosine half-width -> epoch spans 40%..50% (~800 steps)
_R_LRBOT = 0.30       # lr floor inside the epoch, in units of D
_R_BURST = 8.0        # restoration alpha burst height, in alpha0 units (proven max)
# hold envelope and alpha floor as piecewise-linear knots over frac
_KNOT_X = jnp.asarray([0.0, 0.40, 0.50, 0.62, 1.0])
_ENV_Y = jnp.asarray([1.70, 1.40, 1.15, 1.10, 1.10])   # hold lr, in units of D
_FLR_Y = jnp.asarray([0.35, 0.35, 0.60, 0.60, 0.60])   # alpha floor, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while sprinting and polishing
_B1_REP = 0.05        # reduced momentum inside the restoration epoch
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- restoration window: raised cosine, 1 at 45%, exactly 0 outside 40-50% ---
    t = jnp.clip(jnp.abs(frac - _R_CENTER) / _R_HALFW, 0.0, 1.0)
    w = 0.5 * (1.0 + jnp.cos(jnp.pi * t))

    # --- lr: warmup -> hot sprint / wide repair epoch / moderate sprint -> tail ---
    env = jnp.interp(frac, _KNOT_X, _ENV_Y)                   # hold envelope, D units
    lr_hold = (_R_LRBOT + (env - _R_LRBOT) * (1.0 - w)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: phase-dependent floor + one wide burst -> plateau -> climb ---
    floor = jnp.interp(frac, _KNOT_X, _FLR_Y)                 # low then firm floor
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - floor) * ramp + _R_BURST * w
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + momentum cut inside the repair epoch ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_REP) * w                  # cut while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2