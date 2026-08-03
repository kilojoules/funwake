import jax.numpy as jnp

# STRUCTURALLY NEW vs the notch-burst flat-top best (+0.0577%): the cyclic
# burst-repair architecture is REMOVED entirely and replaced by a pure
# ONE-CYCLE WSD with a MONOTONE DYNAMIC PENALTY — the prior-art menu's §7.9
# hypothesis ("alpha0*(1+Ct)^p dynamic penalty") fused with §2's one-cycle
# "warmup -> sustained peak -> cool-down".
#
# The bet is a DUTY-CYCLE COMPLETION. The parent still spends ~20% of every
# exploration cycle inside deep 0.35*D repair notches, i.e. ~12% of the whole
# run at cold, non-exploring lr. Here the exploration phase runs at FULL HEAT
# for 100% of its duration — no notches, no bursts — on a slightly hotter
# hold (1.6*D -> 1.15*D). Constraint debt is bounded not by periodic repair
# stops but by a smoothly GROWING penalty: alpha rises quadratically from
# 0.5*alpha0 (nearly free early, when basin-hopping matters most) to
# 2.5*alpha0 by the end of exploration (drift capped exactly when positions
# start to matter). All repair is then concentrated where lr is already
# shrinking: the proven logistic climb to the bounded 6*alpha0 ALM plateau at
# 66% — which begins while lr is still ~1*D, so violators are pulled back
# fast — followed by the 5/5-seed-feasible cubic-delayed geometric spike to
# the terminal 5*alpha0*D/gamma_min wall. More cumulative heat than any
# ancestor, one uninterrupted exploration arc, and the entire proven endgame
# preserved verbatim.
#
#   lr    — 3% linear warmup (proven) -> uninterrupted flat-top hold decaying
#           linearly 1.6*D -> 1.15*D over the exploration phase -> proven
#           straight linear tail from 62% landing exactly on gamma_min at the
#           last step.
#   alpha — quadratically back-loaded dynamic penalty 0.5 -> 2.5 alpha0 during
#           exploration (replaces floor + bursts), then the proven machinery:
#           logistic ramp to the bounded 6*alpha0 plateau centered at 66%,
#           and the cubic-delayed geometric climb from 78% to the terminal
#           5*alpha0*D/gamma_min spike.
#   betas — proven transitions only: beta2 0.2 -> 0.9 logistic at cool-down
#           start; beta1 0.1 throughout, with the gated drop to 0.02 during
#           the terminal alpha spike (no notch dips — there are no notches).
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.6            # initial hold level, in units of D — hotter, sustained
_HI1 = 1.15           # final hold level; the linear tail launches from here
_A_START = 0.5        # dynamic-penalty start, in alpha0 units (near-free early)
_A_EXP_END = 2.5      # dynamic-penalty level at end of exploration
_A_POW = 2.0          # quadratic back-loading of the exploration penalty
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
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> uninterrupted decaying flat-top hold -> linear tail ---
    # fc sweeps 0 -> 1 across the exploration phase and freezes at 1 past
    # _F_COOL, so the tail launches cleanly from the final hold level _HI1*D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj                # full-duty hot hold
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: monotone dynamic penalty -> bounded plateau -> terminal climb ---
    # Quadratic growth keeps early exploration nearly unpenalized and caps
    # constraint drift just as exploration ends; frozen fc hands 2.5*alpha0 to
    # the proven logistic plateau ramp, then the proven bounded endgame fires.
    grow = _A_START + (_A_EXP_END - _A_START) * fc ** _A_POW
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = grow + (_A_PLAT - _A_EXP_END) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions, nothing else ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2