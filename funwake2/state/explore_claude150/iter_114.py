import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%): SUMT / ALM OUTER
# ITERATIONS REALIZED AS A DEFERRED-REPAIR STAIRCASE. Both prior champions
# interleave repair with exploration (anti-phased cosine bursts; notches cut
# into the hold). Here the duty-cycle logic is pushed to its limit: the hot
# phase is ONE UNINTERRUPTED MAXIMAL-HEAT HOLD — no notch, no restart, no
# repair window ever steals exploration budget — and ALL mid-run repair is
# deferred to the descent, which becomes a classic sequential-penalty
# (SUMT/§7.2-7.5) outer loop: three discrete lr LANDINGS, each colder than
# the last, each paired with a stronger restoration burst. Each landing is
# one "outer iteration": drop the step size, raise the penalty, repair, then
# descend again. The long linear anneal of the best is thereby replaced by
# staged annealing-with-repair, and the hold is both hotter (1.55*D start >
# any sustained level tried) and longer-uninterrupted (~52% of the run) than
# anything in the lineage.
#
#   lr    — 3% linear warmup (proven) -> uninterrupted hold with envelope
#           decaying 1.55*D -> 1.25*D until 55.5% -> sharp sigmoid staircase
#           to landings 0.85*D / 0.50*D / 0.28*D -> proven straight linear
#           tail from 78% landing exactly on gamma_min at the last step.
#   alpha — 0.4*alpha0 exploration floor for the whole hold (debt accrues
#           freely while basins are hopped), then GROWING Gaussian repair
#           bursts (4 -> 6.5 -> 9 alpha0) fired just AFTER each lr drop
#           settles, so every repair runs at the new, colder step size;
#           logistic ramp to the proven bounded 6*alpha0 ALM plateau
#           (centered 70%), and the proven 5/5-seed-feasible cubic-delayed
#           geometric climb from 78% to the terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions: beta2 0.2 -> 0.9 at the end of the hold;
#           beta1 0.1 with dips to 0.05 inside each repair burst (momentum
#           must not carry turbines back across the boundary mid-repair)
#           and the gated drop to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.555       # uninterrupted hot hold ends here
_H0 = 1.55            # hold start, in units of D — hotter than any sustained level tried
_H1 = 1.25            # hold end (decaying envelope, proven helpful)
_C1 = 0.575           # staircase drop centers (outer-iteration boundaries)
_C2 = 0.655
_C3 = 0.735
_WS = 0.006           # sigmoid sharpness of each lr drop
_L1 = 0.85            # landing lr levels, in units of D — colder each stage
_L2 = 0.50
_L3 = 0.28
_F_TAIL = 0.78        # linear tail to gamma_min starts here (proven landing)
_A_LO = 0.4           # exploration penalty floor during the hold, in alpha0 units
_M1 = 0.595           # repair-burst centers — just after each drop settles
_M2 = 0.675
_M3 = 0.750
_WB = 0.013           # Gaussian burst width (~200 steps at 8000)
_HB1 = 4.0            # burst heights, in alpha0 units — stronger each stage
_HB2 = 6.5
_HB3 = 9.0
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.70      # logistic alpha ramp centered inside the staircase
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.56     # beta2 transition aligned with the end of the hold
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_DIP = 0.05        # reduced momentum inside each repair burst
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> uninterrupted hot hold -> 3-landing staircase -> tail ---
    # The hold envelope freezes at _H1 past _F_HOLD; each sharp sigmoid then
    # steps lr down to the next landing, so the descent is a staircase of
    # settled plateaus rather than a continuous anneal.
    hold_env = _H0 + (_H1 - _H0) * jnp.clip(frac / _F_HOLD, 0.0, 1.0)
    s1 = 1.0 / (1.0 + jnp.exp(-(frac - _C1) / _WS))
    s2 = 1.0 / (1.0 + jnp.exp(-(frac - _C2) / _WS))
    s3 = 1.0 / (1.0 + jnp.exp(-(frac - _C3) / _WS))
    lr_units = hold_env + (_L1 - hold_env) * s1 + (_L2 - _L1) * s2 + (_L3 - _L2) * s3
    p = jnp.clip((frac - _F_TAIL) / (1.0 - _F_TAIL), 0.0, 1.0)
    lr_env = gmin + (lr_units * Dj - gmin) * (1.0 - p)        # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> per-landing growing repair bursts -> plateau -> climb ---
    # Each Gaussian fires just after its lr drop has settled, so repair always
    # runs at the new colder step size; all three vanish before the terminal
    # phase, where the proven bounded endgame takes over.
    g1 = jnp.exp(-0.5 * ((frac - _M1) / _WB) ** 2)
    g2 = jnp.exp(-0.5 * ((frac - _M2) / _WB) ** 2)
    g3 = jnp.exp(-0.5 * ((frac - _M3) / _WB) ** 2)
    bursts = _HB1 * g1 + _HB2 * g2 + _HB3 * g3
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + bursts
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    bshape = jnp.clip(g1 + g2 + g3, 0.0, 1.0)
    b1_exp = _B1_HI - (_B1_HI - _B1_DIP) * bshape             # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2