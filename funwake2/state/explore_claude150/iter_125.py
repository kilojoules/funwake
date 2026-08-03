import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-with-notches best (+0.0577%): the lr
# waveform is INVERTED IN TIME into a QUENCH-AND-REHEAT SAWTOOTH. Every prior
# lineage (cosine SGDR, anti-phased bursts, flat-top holds with mid-cycle
# notches) puts heat FIRST and cold in the middle or end of a smooth cycle.
# Here each exploration cycle runs the other way and ends in a hard
# discontinuity:
#
#   [cold repair hold ~15%] -> [logistic reheat] -> [hot hold ~60%] -> QUENCH
#
# Mechanism: the instantaneous quench (hi*D -> 0.35*D in one step) freezes the
# layout wherever the boldest basin-hopping steps just put it; the cycle then
# OPENS with a cold hold + synchronized alpha burst that repairs boundary/
# spacing debt while steps are surgical, and only then re-heats — so every
# hot phase launches from a freshly-repaired, locked-in configuration instead
# of dragging constraint debt through the whole cycle. The cycle coordinate is
# built to hit 1 exactly at the exploration end, so the final hot hold flows
# straight into the linear tail from FULL heat (1.25*D, hotter than the 1.1*D
# tail launch of the best) — a longer effective hot peak, per the guidance.
#
#   lr    — 3% linear warmup (proven) -> 3 reverse-WSD cycles: cold hold at
#           0.35*D, logistic reheat, hot hold at an envelope decaying
#           1.7*D -> 1.25*D, hard quench at each cycle boundary -> proven
#           straight linear tail from 62% landing exactly on gamma_min.
#   alpha — proven feasibility machinery kept intact: 0.4*alpha0 exploration
#           floor, growing repair bursts (3 -> 8 alpha0) now fired INSIDE the
#           post-quench cold holds, logistic ramp to the bounded 6*alpha0 ALM
#           plateau at 66%, and the 5/5-seed-feasible cubic-delayed geometric
#           climb from 78% to the terminal 5*alpha0*D/gamma_min spike.
#   betas — proven transitions: beta2 0.2 -> 0.9 at cool-down; beta1 0.1 with
#           a dip to 0.05 inside each post-quench repair window (hot-phase
#           momentum must not drag turbines back over the boundary right
#           after the quench) and the gated drop to 0.02 in the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three quench-and-reheat cycles in the exploration phase
_LO = 0.35            # cold repair-hold lr, in units of D (proven notch depth)
_HI0 = 1.7            # first hot-hold level, in units of D — hotter than tried peaks
_HI1 = 1.25           # last hot-hold level; the linear tail launches from here
_R_CENTER = 0.27      # logistic reheat center within each cycle
_R_WIDTH = 0.07       # reheat sharpness: ~15% cold, ~25% rise, ~60% hot
_C0 = 0.09            # repair-burst center within each cycle (inside cold hold)
_CW = 0.06            # repair-burst width in cycle coordinate
_A_LO = 0.4           # exploration penalty floor at full heat, in alpha0 units
_A_B0 = 3.0           # first repair burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last repair burst height, in alpha0 units (proven)
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
_B1_QUENCH = 0.05     # reduced momentum inside each post-quench repair window
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- cycle coordinate c in (0, 1] ---
    # c = c_raw - ceil(c_raw - 1) maps to (0, 1] instead of [0, 1): c reaches
    # 1 exactly at each cycle end, so the LAST exploration step sits at full
    # heat and the tail launches from the hot hold, not from a quench floor.
    # The jump 1 -> ~0 across each cycle boundary IS the quench. Past _F_COOL,
    # fc freezes at 1, so c freezes at 1 (hot) and the tail takes over.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    c_raw = fc * _N_CYC
    c = c_raw - jnp.ceil(c_raw - 1.0)

    # --- lr: warmup -> reverse-WSD cycles (cold hold -> reheat -> hot hold,
    #     hard quench between cycles) -> proven linear tail to gamma_min ---
    reheat = 1.0 / (1.0 + jnp.exp(-(c - _R_CENTER) / _R_WIDTH))
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying hot-hold envelope
    lr_cyc = (_LO + (hi - _LO) * reheat) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + post-quench repair bursts -> plateau -> terminal climb ---
    # The burst gate is a gaussian in the cycle coordinate centered inside the
    # cold hold: repair fires right after each quench, while steps are small,
    # and is ~0 during hot holds and everywhere past _F_COOL (c frozen at 1).
    quench = jnp.exp(-(((c - _C0) / _CW) ** 2))
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * quench
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + post-quench beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_QUENCH) * quench          # kill momentum while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2