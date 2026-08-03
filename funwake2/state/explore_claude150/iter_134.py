import jax.numpy as jnp

# STRUCTURALLY NEW vs BOTH the flat-top-notch best (+0.0577%) and the one-cycle
# parent (+0.0520%): those delay ALL serious feasibility work to the terminal
# spike, so the endgame slam must drag every violating turbine home at once —
# and whatever AEP the hot phase found near the boundary gets shoved blind. This
# schedule moves the prior-art §7.5 FEASIBILITY-RESTORATION phase to MID-RUN
# (restore -> warm-restart -> interior polish), a three-act architecture no
# attempt has tried:
#
#   Act I  (0-46%)  HOT EXPLORATION — sustained ~1.8*D flat hold (hotter and
#                   longer-held than any prior sustained level; the best held
#                   only 1.5*D) with a 0.35*alpha0 penalty floor: maximum basin
#                   hopping while constraints are merely advisory.
#   Act II (46-50%) MID-RUN RESTORATION — lr collapses to a near-cold 0.06*D
#                   trough while an isolated Gaussian alpha spike of
#                   3*alpha0*sqrt(D/gamma_min) (the GEOMETRIC MIDPOINT between
#                   plateau scale and the terminal D/gamma_min scale) pins the
#                   enforced violation band to a few metres. ~320 surgical steps
#                   snap the layout essentially feasible at the run's midpoint.
#   Act III(50-100%) INTERIOR POLISH — SGDR-style warm restart to a reduced
#                   1.15*D level, but alpha does NOT return to the floor: it
#                   settles on an ADMM-STYLE MODERATE CONSTANT 2.5*alpha0
#                   plateau (explicitly untried per the search notes), so AEP
#                   is refined while STAYING near-feasible. The proven endgame
#                   is preserved verbatim: logistic tightening to 6*alpha0,
#                   then the cubic-delayed geometric climb to the 5/5-feasible
#                   5*alpha0*D/gamma_min terminal spike — which now only needs
#                   to correct metres, not basins.
#
#   beta1 — 0.12 while hot; dips to 0.03 through the restoration trough
#           (momentum must not carry turbines back across the boundary);
#           rises to 0.40 during interior polish, where momentum averages the
#           persistent constraint gradient and acts as an implicit ALM
#           multiplier backing the moderate 2.5*alpha0 plateau; gated to the
#           proven 0.02 during the terminal spike.
#   beta2 — proven 0.2 -> 0.9 logistic, centered ON the restoration pivot.
#
# lr is one piecewise-linear waveform (jnp.interp, fully traceable), in units
# of D, landing EXACTLY on gamma_min at the last step.
_LR_X = jnp.array([0.00, 0.05, 0.38, 0.46, 0.50, 0.58, 0.72, 1.00])
_LR_Y = jnp.array([0.00, 1.80, 1.45, 0.06, 0.06, 1.15, 0.85, 0.00])  # units of D

_A_FLOOR = 0.35       # penalty floor during hot exploration, in alpha0 units
_A_ADMM = 2.5         # moderate constant plateau for interior polish (ADMM-style)
_A_PLAT = 6.0         # pre-terminal bounded plateau (proven)
_A_MID = 3.0          # mid spike = 3*alpha0*sqrt(D/gmin): geometric-mean scale
_MID_C = 0.48         # restoration spike center (inside the lr trough)
_MID_W = 0.035        # Gaussian width — isolated, ~zero outside 40-58%
_ADMM_C = 0.55        # floor -> ADMM plateau handoff, just after the restart
_ADMM_W = 0.03
_TIGHT_C = 0.74       # ADMM -> 6*alpha0 pre-terminal tightening
_TIGHT_W = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.46     # beta2 pivots exactly at the restoration phase
_B2_WIDTH = 0.05
_B1_HOT = 0.12        # momentum while exploring
_B1_DIP = 0.03        # momentum through the restoration trough
_B1_POLISH = 0.40     # momentum-as-ALM-multiplier during interior polish
_B1_END = 0.02        # near-zero momentum during the terminal spike (proven)
_B1_DIP_W = 0.05      # wider than the alpha spike: momentum drains first
_B1_RISE_C = 0.62
_B1_RISE_W = 0.04
_B1_GATE_C = 0.86
_B1_GATE_W = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: hot hold -> mid-run cold trough -> warm restart -> tail to gmin ---
    lr_shape = jnp.interp(frac, _LR_X, _LR_Y)          # piecewise linear, >= 0
    lr = gmin + lr_shape * Dj                          # exact gamma_min landing

    # --- alpha: floor -> isolated restoration spike -> ADMM plateau -> endgame ---
    bump = jnp.exp(-(((frac - _MID_C) / _MID_W) ** 2))          # restoration window
    admm = 1.0 / (1.0 + jnp.exp(-(frac - _ADMM_C) / _ADMM_W))   # floor -> 2.5
    tight = 1.0 / (1.0 + jnp.exp(-(frac - _TIGHT_C) / _TIGHT_W))  # 2.5 -> 6
    alpha_units = (_A_FLOOR
                   + (_A_ADMM - _A_FLOOR) * admm
                   + (_A_PLAT - _A_ADMM) * tight
                   + _A_MID * jnp.sqrt(Dj / gmin) * bump)
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)   # ends at 5*alpha0*D/gmin

    # --- beta1: hot 0.12 -> trough dip 0.03 -> polish 0.40 -> terminal 0.02 ---
    dip = jnp.exp(-(((frac - _MID_C) / _B1_DIP_W) ** 2))
    rise = 1.0 / (1.0 + jnp.exp(-(frac - _B1_RISE_C) / _B1_RISE_W))
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE_C) / _B1_GATE_W))
    b1_base = _B1_HOT + (_B1_POLISH - _B1_HOT) * rise
    b1_dipped = b1_base * (1.0 - dip) + _B1_DIP * dip
    beta1 = b1_dipped * (1.0 - gate) + _B1_END * gate

    # --- beta2: proven low -> high logistic, pivoting at the restoration ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2