import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the cycle
# GEOMETRY itself. Every restart schedule in the lineage — including the
# best — uses UNIFORM cycle periods. This child uses ACCELERATING ("chirped")
# warm restarts: the cosine phase runs on a warped clock w = fc^2, so cycle
# periods SHORTEN geometrically (reverse-SGDR, T_mult < 1). Two things fall
# out that no uniform-cycle parent can express:
#
#   1. The first cycle stretches to ~58% of the exploration phase, holding
#      lr near its hottest envelope (~1.3-1.75*D) for the first ~20% of the
#      run — exactly the "higher AND longer early peak" the search guidance
#      asks for, and hotter (1.75*D) than any attempt has dared, because...
#   2. ...the restoration bursts (kept anti-phased at the lr troughs, the
#      mechanism that made the parent win) become an accelerating, growing
#      chirp: repairs land at ~25%, ~44%, and ~57% of the run, each sharper
#      and stronger (4 -> 6 -> 8 alpha0) than the last, with the final burst
#      immediately before cool-down — the layout enters the polish phase
#      freshly repaired. Cycle frequency anneals like a temperature: slow
#      basin-scale exploration first, rapid explore/repair flicker last.
#
#   lr    — 3% warmup -> 3 shortening cosine restarts on the warped clock
#           with peaks decaying 1.75*D -> 1.05*D and bounded 0.65*D troughs
#           (w=1 at fc=1 keeps cos(2*pi*N)=1, so cool-down still starts at
#           the proven 1.05*D peak) -> the proven straight linear tail
#           landing exactly on gamma_min at the last step.
#   alpha — 0.4*alpha0 exploration floor at lr peaks; anti-phased bursts
#           whose amplitude grows with the WARPED phase (3 + 6*w: troughs
#           get 4, 6, 8 alpha0); then the proven endgame verbatim — logistic
#           ramp to the bounded 6*alpha0 ALM plateau at the cool-down, and
#           the cubic-delayed geometric climb from 78% onto the 5/5-seed-
#           feasible terminal 5*alpha0*D/gamma_min. Terminal restoration is
#           untouched: feasibility is guaranteed by the same spike that has
#           held on every feasible ancestor.
#   betas — proven transitions (beta2 0.2 -> 0.9 at cool-down; beta1 gated
#           0.1 -> 0.02 in the terminal spike) plus the proven per-burst
#           beta1 dip to 0.05 so momentum never drags turbines back across
#           the boundary while a burst is pulling them in.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase
_KAPPA = 2.0          # phase warp w = fc^2: cycle lengths ~58% / 24% / 18%
_HI0 = 1.75           # first peak — hotter than tried, licensed by the chirp
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # burst amplitude at w=0, in alpha0 units
_A_B1 = 9.0           # burst amplitude at w=1: troughs land at 4, 6, 8 alpha0
_Q = 3.0              # sharpens bursts so alpha is low most of each cycle
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
_B1_BURST = 0.05      # reduced momentum inside each restoration burst
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 chirped (shortening) restarts -> linear tail ---
    # The warped clock w = fc^KAPPA moves slowly early (one long hot cycle)
    # and fast late (rapid final restarts). w(1) = 1 keeps cos(2*pi*N) = 1,
    # pinning the cool-down start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    w = fc ** _KAPPA
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * w))    # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + accelerating growing bursts -> plateau -> terminal climb ---
    # burst = 1 exactly at lr troughs, ~0 near lr peaks (and 0 for frac >= _F_COOL,
    # since w freezes at a peak), so restoration always coincides with low lr.
    # Amplitude grows on the warped clock so the chirped troughs hit 4, 6, 8.
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * w                   # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
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