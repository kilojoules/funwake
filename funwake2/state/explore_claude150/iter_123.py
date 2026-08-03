import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top+3-notch best (+0.0577%): the periodic
# repair cycling is replaced by GRADUATED NON-CONVEXITY WITH A SINGLE MID-RUN
# FEASIBILITY-RESTORATION WINDOW — the untried "mid-run restoration burst"
# direction fused with a two-block heat staircase, keeping the proven endgame
# (linear tail, logistic ALM plateau, cubic-delayed terminal spike) intact.
#
# Mechanism. The parent pays constraint debt THREE times: each notch spends
# exploration budget on repair and each repair perturbs the layout it just
# found. Here the exploration phase is split into two regimes instead:
#
#   Block 1 (0 -> 40%): RELAXED ASCENT. lr holds at 1.7*D — hotter and longer
#     than any sustained hold tried (meta-signal: push heat early) — while
#     alpha sits at a low 0.25*alpha0 floor. The optimizer solves the nearly
#     unconstrained AEP problem first (graduated/homotopy continuation),
#     hopping basins without fighting the penalty field.
#   Restoration window (~40%, one Gaussian notch, sigma 2.5% of the run):
#     lr dives to 0.30*D while alpha bursts to 7*alpha0 and beta1 dips to
#     0.05 — one concentrated filter-style feasibility restoration that pulls
#     the relaxed layout back onto the feasible manifold, paid ONCE.
#   Block 2 (40 -> 62%): CONSTRAINED REFINEMENT. lr holds at 1.15*D with the
#     alpha floor raised to 1.0*alpha0, refining the repaired layout while
#     staying near-feasible so the endgame starts from a healthy iterate.
#
# Endgame (62 -> 100%) is the proven 5/5-seed-feasible machinery, unchanged:
# straight linear lr tail landing exactly on gamma_min; logistic alpha ramp to
# the bounded 6*alpha0 plateau at 66%; cubic-delayed geometric climb from 78%
# to the terminal 5*alpha0*D/gamma_min spike; beta2 0.2 -> 0.9 at cool-down;
# beta1 gated down to 0.02 during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_F_REPAIR = 0.40      # center of the single mid-run restoration window
_W_REPAIR = 0.025     # Gaussian sigma of the restoration window (frac units)
_W_BLOCK = 0.02       # width of the block-1 -> block-2 transition
_HI_HOT = 1.7         # block-1 hold level, in units of D — relaxed ascent
_HI_REF = 1.15        # block-2 hold level, in units of D — refinement
_LO = 0.30            # lr at the restoration-window bottom
_A_HOT = 0.25         # near-free penalty floor during relaxed ascent
_A_REF = 1.0          # raised floor during constrained refinement
_A_BURST = 7.0        # restoration burst height, in alpha0 units
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
_B1_NOTCH = 0.05      # reduced momentum inside the restoration window
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # Shared phase machinery: smooth block gate + single Gaussian repair notch.
    # The notch is ~exp(-39) at frac=_F_COOL, so the linear tail launches from
    # the clean refinement hold _HI_REF*D.
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _F_REPAIR) / _W_BLOCK))
    notch = jnp.exp(-0.5 * ((frac - _F_REPAIR) / _W_REPAIR) ** 2)

    # --- lr: warmup -> hot hold -> single deep restoration dip -> refinement
    # hold -> proven straight linear tail landing exactly on gamma_min ---
    hi = _HI_HOT + (_HI_REF - _HI_HOT) * gate                 # two-level staircase
    lr_hold = (_LO + (hi - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: graduated floor (0.25 -> 1.0 alpha0) + one restoration burst
    # -> proven bounded plateau -> proven cubic-delayed terminal spike ---
    floor = _A_HOT + (_A_REF - _A_HOT) * gate
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - floor) * ramp + _A_BURST * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + beta1 dip inside the restoration window ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2