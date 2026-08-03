import jax.numpy as jnp

# STRUCTURALLY NEW vs the notched flat-top best (+0.0577%): the OSCILLATORY
# machinery is removed entirely. The lr waveform becomes a MONOTONE STEP-DECAY
# LADDER (three descending flat holds, WSD-with-multiple-holds, §6), and alpha
# becomes a MONOTONE PENALTY RATCHET (§7.9 dynamic penalty / classic ALM):
# every time the lr steps DOWN a stair, alpha steps UP one — a permanent
# regime shift instead of the parent's transient repair notches + bursts.
#
# Rationale: the parent's periodic notch/burst cycle repairs constraint debt,
# then re-heats and re-incurs it — repair work is repeatedly thrown away. The
# ladder converts each repair into a one-way transition: the first stair is
# HOTTER and LONGER than any tried hold (1.7*D for the first ~25% of the run,
# per the "higher/longer early peak" direction), and each subsequent stair
# trades a little heat for a permanently higher penalty, so basins found late
# in a stair are refined, not re-scrambled. Feasibility never depends on the
# exploration phase: the proven terminal restoration (bounded 6*alpha0 ALM
# plateau -> cubic-delayed geometric climb to 5*alpha0*D/gamma_min, lr tail
# landing exactly on gamma_min) is kept verbatim from the 5/5-seed-feasible
# endgame.
#
#   lr    — 3% linear warmup (proven) -> stairs 1.7*D -> 1.4*D -> 1.1*D via
#           sharp sigmoids inside the exploration phase (extended to 70%),
#           launching the proven straight linear tail from the proven 1.1*D
#           level down to gamma_min exactly at the last step.
#   alpha — monotone ratchet 0.4 -> 1.1 -> 2.6 alpha0 locked to the lr stair
#           drops (repair continuously at each colder/stricter regime), then
#           the proven logistic ramp onto the bounded 6*alpha0 plateau and the
#           proven cubic-backloaded terminal spike from 80%.
#   betas — beta2 0.2 -> 0.9 at cool-down (proven, recentered to 70%); beta1
#           one-cycle ANTI-CORRELATED with lr: 0.08 on the hottest stair
#           rising to 0.12 on the coldest (momentum as implicit ALM
#           multiplier, §2/§4), then the proven gated drop to 0.02 during the
#           terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.70        # exploration extended: linear decay to gamma_min after 70%
_C1 = 0.36            # first stair drop, in exploration-phase units
_C2 = 0.72            # second stair drop
_W = 0.045            # sigmoid stair sharpness (~2-3% of the run per transition)
_L0 = 1.70            # hottest hold, in units of D — higher AND longer than tried
_L1 = 1.40            # middle stair
_L2 = 1.10            # final stair; the proven tail launches from 1.1*D
_A0 = 0.4             # exploration penalty floor on the hot stair (proven value)
_A1 = 1.1             # ratchet level on the middle stair
_A2 = 2.6             # ratchet level on the final stair
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.74      # logistic ramp onto the plateau, just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.80        # terminal geometric alpha climb starts here
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.70     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_S0 = 0.08         # low momentum on the hottest stair (one-cycle anti-phase)
_B1_S1 = 0.10         # native momentum on the middle stair
_B1_S2 = 0.12         # highest momentum on the coldest stair
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike (proven)
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL   # exploration-phase clock, freezes at 1

    # Shared stair transitions: one sigmoid per drop, reused by lr, alpha, beta1
    # so the penalty ratchet and momentum shift fire exactly at each lr drop.
    sig1 = 1.0 / (1.0 + jnp.exp(-(fc - _C1) / _W))
    sig2 = 1.0 / (1.0 + jnp.exp(-(fc - _C2) / _W))

    # --- lr: warmup -> monotone three-stair ladder -> proven linear tail ---
    lr_hold = (_L0 - (_L0 - _L1) * sig1 - (_L1 - _L2) * sig2) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: monotone ratchet locked to the stairs -> plateau -> climb ---
    ladder = _A0 + (_A1 - _A0) * sig1 + (_A2 - _A1) * sig2    # one-way penalty ratchet
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = ladder + (_A_PLAT - _A2) * ramp             # continues the ratchet to 6
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; beta1 anti-correlated with the lr stairs ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_ladder = _B1_S0 + (_B1_S1 - _B1_S0) * sig1 + (_B1_S2 - _B1_S1) * sig2
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_ladder + (_B1_LO - b1_ladder) * b1r

    return lr, alpha, beta1, beta2