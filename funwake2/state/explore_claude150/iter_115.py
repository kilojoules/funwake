import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top+notch best (+0.0577%): the periodic
# repair-notch cadence is replaced by a TWO-TEMPERATURE MACRO-PHASE design
# with ONE deep MID-RUN FEASIBILITY-RESTORATION TRENCH — the explicitly
# untried "mid-run feasibility-restoration" direction (filter/funnel, §7.5)
# scaled from three brief notches up to a single dedicated 10%-of-budget
# restoration phase, fused with an ADMM-style bounded plateau (§7.2/7.3).
#
# Mechanism / duty-cycle argument: the parent interleaves repair with
# exploration, so every notch interrupts a basin hop and every hop re-incurs
# constraint debt that the next notch must repay. Here the run is split into
# two clean regimes instead:
#   Block A (0-45%):  UNINTERRUPTED hot hold at 1.7*D (hotter and longer than
#                     any sustained level tried) with alpha pinned at a low
#                     0.3*alpha0 floor — pure basin search, debt allowed.
#   Trench (45-55%):  lr collapses to ~0.3*D while alpha bursts to 10*alpha0
#                     for ~800 consecutive steps — one full restoration phase
#                     that pays the entire exploration debt at once, with a
#                     momentum cut so repaired turbines are not dragged back.
#   Block B (55-75%): moderate reheat to 0.9*D under a bounded 5*alpha0 ALM
#                     plateau — feasible-region refinement, not exploration.
#   Tail (75-100%):   proven straight linear lr decay landing exactly on
#                     gamma_min, plus the 5/5-seed-feasible cubic-delayed
#                     geometric alpha climb to the 5*alpha0*D/gamma_min spike.
# Net effect: strictly more cumulative heat early (the requested hotter/longer
# peak) AND a longer, deeper single repair window than the parent's notches,
# with the entire proven terminal feasibility machinery kept intact.
_F_WARM = 0.04        # linear lr warmup over the first 4% (damps the 1.7*D start)
_F_IN = 0.45          # trench entry: exploration block A ends here
_F_OUT = 0.55         # trench exit: refinement block B starts here
_F_TAIL = 0.75        # linear cool-down to gamma_min begins here
_W = 0.015            # logistic edge width for the (traceable) phase switches
_HOT = 1.7            # block-A hold level, in units of D — sustained, no decay
_TRENCH = 0.25        # trench lr floor, in units of D — surgical repair steps
_REFINE = 0.9         # block-B refinement level, in units of D
_A_FLOOR = 0.3        # block-A penalty floor, in alpha0 units — free exploration
_A_REST = 10.0        # restoration-trench alpha, in alpha0 units
_A_PLAT = 5.0         # bounded ALM plateau for block B and the tail (proven scale)
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.75     # beta2 transition aligned with the tail start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and refining
_B1_TRENCH = 0.03     # near-zero momentum inside the restoration trench
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- phase switches (smooth, jit-safe) ---
    down = 1.0 / (1.0 + jnp.exp(-(frac - _F_IN) / _W))    # 0 -> 1 entering trench
    up = 1.0 / (1.0 + jnp.exp(-(frac - _F_OUT) / _W))     # 0 -> 1 leaving trench
    trench = down * (1.0 - up)                            # ~1 only inside 45-55%

    # --- lr: warmup -> hot hold -> trench -> refinement hold -> linear tail ---
    lr_units = _HOT + (_TRENCH - _HOT) * down + (_REFINE - _TRENCH) * up
    lr_hold = lr_units * Dj
    p = jnp.clip((frac - _F_TAIL) / (1.0 - _F_TAIL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)          # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)               # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: low floor -> one deep restoration burst -> bounded plateau ---
    # Decoupled from lr (no 1/lr coupling): floor while hot, 10*alpha0 inside
    # the trench, then the ADMM-style 5*alpha0 plateau, then the proven
    # cubic-delayed geometric climb to the terminal 5*alpha0*D/gmin spike.
    alpha_units = _A_FLOOR + (_A_PLAT - _A_FLOOR) * up + _A_REST * trench
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + trench momentum cut ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_base = _B1_HI - (_B1_HI - _B1_TRENCH) * trench     # cut momentum while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_base + (_B1_LO - b1_base) * b1r

    return lr, alpha, beta1, beta2