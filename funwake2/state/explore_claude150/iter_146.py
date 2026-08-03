import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-with-notches best (+0.0577%): the cyclic
# machinery is replaced by a GREEDY / REPAIR / CONSOLIDATE two-block schedule —
# one long, HOTTER uninterrupted greed phase, ONE wide flat-bottomed repair
# TRENCH, then a second warm consolidation block with a RAISED alpha floor.
#
# Rationale: every winning move in this lineage increased sustained heat and
# made repair windows fewer/deeper (cosine restarts -> 80%-duty flat-top).
# This takes that trend to its limit while changing the waveform class:
#  * Block A (3%-38%): lr pinned at 1.7*D — hotter and far longer without
#    interruption than anything tried (parent never exceeded 1.5*D and broke
#    its hold three times). alpha at a low 0.35*alpha0 floor: pure basin
#    exploration, constraint debt deliberately accumulated.
#  * Trench (38%-46%): a single wide smooth-box restoration window (not a
#    momentary sin^12 notch): lr sinks to 0.25*D and STAYS there while a
#    9*alpha0 burst repays the whole debt at once; beta1 drops to 0.04 so
#    momentum cannot drag turbines back out of bounds, and beta2 PULSES up to
#    0.8 (the untried §4 bet: high beta2 absorbs the stiff ~alpha constraint
#    curvature exactly when the penalty is stiff). Sustained small steps under
#    high alpha = a real projection pass, not a graze.
#  * Block B (46%-64%): lr 1.15*D with alpha floor raised to 1.0*alpha0 —
#    consolidation near the feasible manifold, so cooldown starts from a
#    nearly-repaired layout instead of relying on late bursts.
#  * Endgame (64%-100%): the proven, 5/5-seed-feasible machinery verbatim:
#    straight linear lr tail landing exactly on gamma_min, logistic alpha ramp
#    to the bounded 6*alpha0 ALM plateau, cubic-delayed geometric climb from
#    78% to the terminal 5*alpha0*D/gamma_min spike, beta2 0.2->0.9 at
#    cooldown, gated beta1 drop to 0.02 under the terminal spike.
_F_WARM = 0.03      # linear lr warmup over the first 3% (proven)
_F_T0 = 0.38        # repair trench opens
_F_T1 = 0.46        # repair trench closes
_W_EDGE = 0.012     # trench edge sharpness (smooth box, fully traceable)
_F_MID = 0.42       # block A -> block B transition (hidden under the trench)
_W_MID = 0.02
_F_COOL = 0.64      # exploration ends; linear decay to gamma_min at 100%
_LR_A = 1.7         # block-A hold, in units of D — hotter than any prior hold
_LR_B = 1.15        # block-B consolidation hold, in units of D
_LR_TRENCH = 0.25   # trench-bottom lr — sustained surgical repair
_A_FLOOR_A = 0.35   # greed-phase penalty floor, in alpha0 units
_A_FLOOR_B = 1.0    # raised consolidation floor — stay near-feasible
_A_BURST = 9.0      # single big trench burst, in alpha0 units
_A_PLAT = 6.0       # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.68    # logistic alpha ramp centered after cooldown starts
_A_WIDTH = 0.04
_F_TERM = 0.78      # terminal geometric alpha climb starts here (proven)
_POW = 3.0          # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0    # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.64   # beta2 transition aligned with cooldown start
_B2_WIDTH = 0.05
_B2_TRENCH = 0.8    # beta2 pulse inside the trench (penalty-curvature damping)
_B1_HI = 0.1        # native momentum while exploring and polishing
_B1_TRENCH = 0.04   # near-zero momentum during the repair trench
_B1_LO = 0.02       # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    def sig(x, c, w):
        # numerically stable logistic via tanh; traceable, no branches
        return 0.5 * (1.0 + jnp.tanh((x - c) / (2.0 * w)))

    # smooth box = 1 inside [_F_T0, _F_T1], 0 outside — the single repair trench
    box = sig(frac, _F_T0, _W_EDGE) * (1.0 - sig(frac, _F_T1, _W_EDGE))
    blockB = sig(frac, _F_MID, _W_MID)       # 0 in block A, 1 in block B

    # --- lr: warmup -> hot hold -> wide trench -> warm hold -> linear tail ---
    hold = (_LR_A + (_LR_B - _LR_A) * blockB) * Dj
    lr_expl = _LR_TRENCH * Dj + (hold - _LR_TRENCH * Dj) * (1.0 - box)
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_expl - gmin) * (1.0 - p)   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)        # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: low floor -> single big trench burst -> raised floor ->
    #     bounded plateau -> proven terminal geometric spike ---
    floor = _A_FLOOR_A + (_A_FLOOR_B - _A_FLOOR_A) * blockB
    ramp = sig(frac, _A_CENTER, _A_WIDTH)
    alpha_units = floor + (_A_PLAT - floor) * ramp + _A_BURST * box
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + trench-synchronized pulse/dip ---
    b2 = _B2_LO + (_B2_HI - _B2_LO) * sig(frac, _B2_CENTER, _B2_WIDTH)
    beta2 = b2 + (_B2_TRENCH - b2) * box           # pulse up only inside trench
    b1 = _B1_HI - (_B1_HI - _B1_TRENCH) * box      # kill momentum while repairing
    beta1 = b1 + (_B1_LO - b1) * sig(frac, _B1_CENTER, _B1_WIDTH)

    return lr, alpha, beta1, beta2