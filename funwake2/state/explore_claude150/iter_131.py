import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): the notch/burst
# repair cadence is REMOVED entirely and replaced by a classic SUMT-style
# PENALTY CONTINUATION IN COUNTERPOINT — a monotone DESCENDING lr STAIRCASE
# against a monotone ASCENDING alpha staircase, the sequential-penalty
# homotopy from the prior-art menu (§7.9 dynamic penalty + §6 WSD holds)
# fused with the untried §2 one-cycle bet: beta1 ANTI-CORRELATED with lr.
#
# Mechanism change, not a knob tweak. The best schedule interleaves heat and
# repair on a fast duty cycle (notches + synchronized bursts), paying
# constraint debt every ~20% of a cycle. Here each phase does ONE job:
#   stage 1 — hotter and longer than any tried hold (1.8*D, ~19% of steps
#             after warmup) at the bare 0.4*alpha0 floor: pure basin hopping,
#             debt allowed to accumulate;
#   stage 2 — cool to 1.3*D, tighten to 1.2*alpha0: layouts consolidate;
#   stage 3 — cool to 0.9*D, tighten to 3*alpha0: bulk constraint repair at
#             genuinely small steps (below the old 1.1*D tail launch point);
# every lr drop is matched by a penalty tightening — "when you cool, you
# tighten" — so repair happens continuously in the cold stages instead of in
# brief notches. Momentum runs one-cycle style: low (0.06) while hot so
# momentum cannot fling turbines across basins, rising to 0.14 as lr falls to
# accelerate travel along consolidated valleys.
#
# The PROVEN 5/5-seed feasibility endgame is preserved verbatim: linear lr
# tail landing exactly on gamma_min at the last step, logistic ramp to the
# bounded 6*alpha0 ALM plateau at 66%, cubic-delayed geometric climb from 78%
# to the terminal 5*alpha0*D/gamma_min spike, beta2 0.2 -> 0.9 at cool-down,
# and the gated beta1 drop to 0.02 during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_T1 = 0.22            # first staircase transition (hot -> mid)
_T2 = 0.44            # second staircase transition (mid -> repair)
_TW = 0.02            # sigmoid width of the staircase transitions
_L0 = 1.8             # stage-1 hold lr, in units of D — hottest sustained hold tried
_L1 = 1.3             # stage-2 hold lr, in units of D
_L2 = 0.9             # stage-3 hold lr, in units of D; tail launches from here
_A0 = 0.4             # stage-1 penalty, in alpha0 units (proven exploration floor)
_A1 = 1.2             # stage-2 penalty, in alpha0 units
_A2 = 3.0             # stage-3 penalty, in alpha0 units
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
_B1_HOT = 0.06        # momentum while hot (anti-correlated with lr)
_B1_COOL = 0.14       # momentum once cooled to the repair stage
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # smooth staircase gates; frozen at their end values past each transition
    g1 = 1.0 / (1.0 + jnp.exp(-(frac - _T1) / _TW))
    g2 = 1.0 / (1.0 + jnp.exp(-(frac - _T2) / _TW))

    # --- lr: warmup -> descending three-level staircase -> linear tail ---
    lr_units = _L0 + (_L1 - _L0) * g1 + (_L2 - _L1) * g2      # 1.8 -> 1.3 -> 0.9
    lr_hold = lr_units * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ascending staircase in counterpoint -> plateau -> climb ---
    # Every lr drop is matched by a penalty tightening; no bursts, repair is
    # continuous in the cold stages. Proven bounded endgame takes over at 66%.
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = (_A0 + (_A1 - _A0) * g1 + (_A2 - _A1) * g2
                   + (_A_PLAT - _A2) * ramp)
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: beta1 anti-correlated with the lr staircase + proven endgame ---
    m = (_L0 - lr_units) / (_L0 - _L2)                        # 0 while hot -> 1 cooled
    b1_exp = _B1_HOT + (_B1_COOL - _B1_HOT) * m
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2