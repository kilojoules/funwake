import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lr backbone is
# replaced wholesale. The best (and every restart parent before it) uses
# SYMMETRIC cosine cycles — half of every cycle is wasted climbing back up, and
# troughs stay high (0.65*D), so its "restoration windows" never really stop
# the layout moving. Here the exploration phase is a CLASSIC SGDR SAWTOOTH ON A
# WARPED CLOCK — anneal -> repair -> INSTANT reheat:
#
#   lr    — 3% warmup, then 4 sawtooth cycles: each cycle is a pure cosine
#           DECAY from a decaying peak (1.85*D -> 1.0*D, hotter than the tried
#           1.65*D ceiling) down to a DEEP trough at 0.25*D, followed by a
#           discontinuous jump back to the next peak (the SGDR perturbation the
#           symmetric shape forfeits). A time warp fc**0.6 makes cycle periods
#           GROW (SGDR T_mult-style: ~0.10/0.22/0.30/0.38 of the phase), so
#           early cycles hop basins fast and late cycles refine longly at
#           mid-lr. Exploration ends at 64% with the clock exactly at a
#           restart, so the proven straight linear tail IS the final cycle's
#           decay: 1.0*D -> gamma_min, landing exactly at the last step.
#   alpha — floor 0.4*alpha0 while hot, and an END-OF-CYCLE restoration burst
#           (cyc_pos**3, growing 3 -> 9 alpha0 across cycles): unlike the
#           parent's mid-cycle bursts at lively 0.65*D steps, each burst here
#           lands in a DEEP trough where steps are near-frozen, so repair is
#           pure — then the reheat instantly returns alpha to the floor. The
#           whole proven feasibility endgame is preserved verbatim in shape:
#           logistic ramp to the bounded 6*alpha0 ALM plateau after cool-down,
#           cubic-delayed geometric climb from 80% to the 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions kept (beta2 0.2 -> 0.9 at cool-down; beta1
#           gated to 0.02 in the terminal spike), with the burst-aligned beta1
#           dip deepened to 0.04 so no momentum survives a repair into the
#           next reheat.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.64        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 4.0          # four sawtooth anneal-repair-reheat cycles
_K_WARP = 0.6         # clock warp: cycle periods grow across the phase
_HI0 = 1.85           # first restart peak — hotter than anything tried
_HI1 = 1.0            # last peak; the linear tail is its decay (proven landing)
_LO = 0.25            # DEEP trough lr, in units of D — near-frozen repair steps
_A_LO = 0.4           # exploration penalty floor while hot, in alpha0 units
_A_B0 = 3.0           # first end-of-cycle burst height, in alpha0 units
_A_B1 = 9.0           # last burst height — stronger, since steps are tiny there
_Q = 3.0              # burst = cyc_pos**Q: alpha low most of each cycle
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.68      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.80        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.64     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_BURST = 0.04      # near-zero momentum inside each deep-trough repair
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.89
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> warped sawtooth restarts -> linear tail ---
    # fc freezes at 1 past _F_COOL; there N*fc**k is an integer, so cyc_pos
    # wraps to 0 (a fresh restart) and the tail decays from the final peak.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc_pos = jnp.mod(_N_CYC * fc ** _K_WARP, 1.0)            # 0 at reheat, ->1 at trough
    shape = 0.5 * (1.0 + jnp.cos(jnp.pi * cyc_pos))           # pure decay within a cycle
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * shape) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + end-of-cycle growing bursts -> plateau -> terminal climb ---
    # burst peaks exactly at each deep trough (just before the reheat) and is 0
    # at reheats and throughout the tail (cyc_pos wraps to 0 past _F_COOL).
    burst = cyc_pos ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + burst-aligned beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst            # kill momentum while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2