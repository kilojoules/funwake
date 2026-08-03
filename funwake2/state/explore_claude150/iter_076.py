import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the bursts REPAY
# violation debt at every trough but then RELEASE all pressure back to the
# 0.4*alpha0 floor, so each new hot peak re-accumulates the same debt from
# scratch. This schedule replaces release-bursts with a classic ALM MONOTONE
# MULTIPLIER RATCHET (prior-art §7.2/§7.9, the ADMM/epsilon-constraint bet):
# each restoration leaves behind a PERMANENT floor increment, so the enforced
# violation band contracts monotonically across cycles instead of breathing.
# The steadily rising floor is what licenses the two other structural changes:
# a hotter first peak than any attempt (1.75*D) and SGDR-CANONICAL LENGTHENING
# CYCLES (T_mult=2, lengths 1:2:4) — never tried; all parents used equal
# cycles. The long final cycle behaves like the untried WSD/one-cycle hold:
# an extended late exploration stretch that merges smoothly into the proven
# linear tail.
#
#   lr    — 3% warmup -> three cosine restarts with lengths 1:2:4 inside the
#           exploration phase (extended to 64%), peaks decaying 1.75*D ->
#           1.05*D, bounded troughs at 0.65*D, then the proven straight
#           linear tail landing exactly on gamma_min at the last step.
#   alpha — RATCHETED RESTORATION: exploration floor starts at 0.4*alpha0 and
#           climbs by a smooth logistic stair at each lr trough (+0.5, +0.6,
#           +0.7 -> floor 2.2*alpha0), so later basin hops explore inside a
#           progressively tighter feasible band. Growing anti-phased bursts
#           (2.5 -> 6*alpha0) still concentrate repair where lr is near zero,
#           but they are smaller than the parent's because the ratchet holds
#           pressure between troughs. After 64% the proven logistic ramp
#           lifts the 2.2 floor to the bounded 6*alpha0 ALM plateau, and the
#           proven cubic-delayed geometric climb from 78% lands exactly on
#           the 5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions (beta2 0.2 -> 0.9 at cool-down; beta1 gated
#           0.1 -> 0.02 in the terminal spike) plus the proven per-burst
#           beta1 dip to 0.05 so momentum never drags turbines back across
#           the boundary while a restoration is pulling them in.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.64        # exploration ends here; linear decay to gamma_min at 100%
_CB1 = 1.0 / 7.0      # cycle boundaries in exploration-phase units: lengths 1:2:4
_CB2 = 3.0 / 7.0
_HI0 = 1.75           # first restart peak — hottest yet; ratchet pays for it
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # initial exploration penalty floor, in alpha0 units
_S1, _S2, _S3 = 0.5, 0.6, 0.7   # permanent ratchet increments per trough
_M1 = 1.0 / 14.0      # trough centers in exploration-phase units
_M2 = 2.0 / 7.0
_M3 = 5.0 / 7.0
_SW = 0.03            # stair width (exploration-phase units)
_A_B0 = 2.5           # first restoration burst height, in alpha0 units
_A_B1 = 6.0           # last burst height (smaller than parent: ratchet helps)
_Q = 3.0              # sharpens bursts so alpha is near-floor most of a cycle
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.68      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.64     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_BURST = 0.05      # reduced momentum inside each restoration burst
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 lengthening (1:2:4) decaying-peak restarts -> tail ---
    # fc freezes at 1 past _F_COOL; each c_i equals 1 outside its own cycle
    # (cos(0)=cos(2*pi)=1), so the product is the active cycle's cosine and
    # cyc=1 at fc=1 pins the cool-down start at the final peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    t1 = jnp.clip(fc / _CB1, 0.0, 1.0)
    t2 = jnp.clip((fc - _CB1) / (_CB2 - _CB1), 0.0, 1.0)
    t3 = jnp.clip((fc - _CB2) / (1.0 - _CB2), 0.0, 1.0)
    c1 = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * t1))
    c2 = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * t2))
    c3 = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * t3))
    cyc = c1 * c2 * c3                                        # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ratcheted floor + anti-phased bursts -> plateau -> terminal ---
    # Each logistic stair is centered on an lr trough and NEVER releases: the
    # monotone-multiplier floor climbs 0.4 -> 2.2 alpha0 across the cycles.
    stair = (_S1 / (1.0 + jnp.exp(-(fc - _M1) / _SW))
             + _S2 / (1.0 + jnp.exp(-(fc - _M2) / _SW))
             + _S3 / (1.0 + jnp.exp(-(fc - _M3) / _SW)))
    floor = _A_LO + stair                                     # ratchet: monotone in fc
    burst = (1.0 - cyc) ** _Q                                 # 1 at troughs, ~0 at peaks
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    stair_full = _S1 + _S2 + _S3
    alpha_units = floor + (_A_PLAT - _A_LO - stair_full) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst            # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2