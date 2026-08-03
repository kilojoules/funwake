import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-hold-with-3-notches best (+0.0577%): the
# periodic hold/notch waveform is replaced by an APERIODIC TWO-REGIME HOMOTOPY
# — classic "relax constraints, expand, then compress once" from graduated /
# homotopy methods, fused with the proven bounded-ALM endgame.
#
# Phase A (0 -> 30%): SUPER-HOT FREE FLIGHT. A sustained 2.0*D hold — far
# hotter than any hold or peak tried (best held 1.5*D) — with the alpha floor
# dropped to 0.2*alpha0 so the layout expands almost unconstrained (the AEP
# objective pushes turbines apart/past the boundary; we deliberately let it),
# and beta1 raised to 0.3 (untried §4 momentum ramp: ballistic moves help hop
# basins while beta2=0.2 keeps steps adaptive).
#
# ONE consolidation interlude (~35%): instead of three periodic repair
# notches, a single deep Gaussian event — lr collapses to 0.3*D, alpha bursts
# to 7*alpha0, beta1 drops to 0.05 — that compresses the expanded cloud back
# through the boundary exactly once, converting the free expansion into a
# tighter feasible packing.
#
# Phase B (35% -> 62%): MODERATE REFINEMENT. Flat 1.1*D hold (WSD §6) with a
# 0.6*alpha0 floor and native betas — polish the consolidated layout while
# keeping constraint debt bounded before the endgame.
#
# Endgame (62% -> 100%): kept byte-identical in structure to the proven 5/5-
# feasible machinery — straight linear lr tail from 1.1*D landing exactly on
# gamma_min; logistic alpha ramp to the bounded 6*alpha0 ALM plateau at 66%;
# cubic-delayed geometric climb from 78% to the terminal 5*alpha0*D/gamma_min
# feasibility spike; beta2 0.2 -> 0.9 at cool-down; beta1 gated to 0.02
# during the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_GATE_C = 0.30        # hot -> moderate regime transition center
_GATE_W = 0.02        # smooth (traceable) transition width
_HOT = 2.0            # phase-A sustained hold, in units of D — hotter than any tried
_MOD = 1.1            # phase-B hold; the proven linear tail launches from here
_LO = 0.3             # lr at the bottom of the consolidation interlude
_NOTCH_C = 0.35       # single Gaussian consolidation event, just after phase A
_NOTCH_S = 0.02       # Gaussian sigma (~300 steps of surgical repair)
_F_COOL = 0.62        # exploration ends; linear decay to gamma_min at 100% (proven)
_A_HOT = 0.2          # near-free expansion floor in phase A, in alpha0 units
_A_MOD = 0.6          # bounded-debt floor in phase B, in alpha0 units
_A_BURST = 7.0        # consolidation burst height, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start (proven)
_B2_WIDTH = 0.05
_B1_HOT = 0.3         # ballistic momentum during the super-hot free flight
_B1_MID = 0.1         # native momentum for refinement and polishing
_B1_NOTCH = 0.05      # reduced momentum inside the consolidation interlude
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- regime gate and the single consolidation event ---
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _GATE_C) / _GATE_W))   # 0 in phase A, 1 in phase B
    notch = jnp.exp(-0.5 * ((frac - _NOTCH_C) / _NOTCH_S) ** 2) # one Gaussian interlude

    # --- lr: warmup -> 2.0*D free flight -> deep interlude -> 1.1*D hold -> tail ---
    hold = _HOT + (_MOD - _HOT) * gate
    lr_hold = (_LO + (hold - _LO) * (1.0 - notch)) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: low free-flight floor -> single big burst -> plateau -> climb ---
    floor = _A_HOT + (_A_MOD - _A_HOT) * gate                 # debt allowed early, bounded later
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - floor) * ramp + _A_BURST * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: ballistic -> native momentum, interlude dip, proven endgame ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HOT + (_B1_MID - _B1_HOT) * gate             # momentum boost only while hot
    b1_exp = b1_exp + (_B1_NOTCH - b1_exp) * notch            # dip while consolidating
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2