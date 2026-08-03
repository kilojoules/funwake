import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the restart
# PERIOD GEOMETRY changes from equal-length cycles to GEOMETRICALLY HALVING
# periods (4:2:1) — a "ringdown" schedule no lineage member has tried. Every
# prior multi-cycle attempt sliced exploration into equal cycles, so the
# hottest peak was also the briefest heat the run ever felt. Here the FIRST
# half-cycle is both the HOTTEST (1.70*D) and by far the LONGEST (~4/7 of
# exploration): one long violent basin-search exactly when the layout is
# cheapest to restructure — precisely the "higher/longer lr peak early" the
# parent guidance asks for. The remaining cycles then shorten geometrically
# and cool toward the tail, so repair-and-hop events come FASTER and GENTLER
# as the polish approaches: the layout rings down into feasibility instead of
# being interrupted on a fixed clock.
#
#   lr    — 3% warmup (proven) -> three cosine restarts with halving periods
#           (phase u = N - log2(1 + (2^N-1)(1-fc)), smooth/traceable; cycle
#           starts at u = 0,1,2 and fc=1 lands exactly on a peak) and peaks
#           decaying 1.70*D -> 1.05*D, troughs pinned at the proven bounded
#           0.65*D. From 62% the proven straight linear tail lands lr exactly
#           on gamma_min at the last step, starting from the 1.05*D peak.
#   alpha — the PROVEN feasibility machinery, untouched in structure: a
#           0.4*alpha0 exploration floor with restoration bursts anti-phased
#           to the lr troughs (burst = (1-cyc)^3, amp growing 3 -> 8 alpha0
#           across the run). Under the ringdown geometry the bursts now
#           arrive at ~34%, ~74%, ~94% of exploration — later and stronger
#           for the long hot phase's larger debt, then increasingly frequent
#           just before cool-down, so the polish inherits an almost-feasible
#           layout. After 62% the proven logistic ramp lifts alpha to the
#           bounded 6*alpha0 ALM plateau, and the proven cubic-delayed
#           geometric climb from 78% lands it on the 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at cool-down; beta1
#           0.1 while exploring, dipped to 0.05 inside each restoration burst
#           (momentum never drags turbines back over the boundary mid-repair)
#           and gated to 0.02 during the terminal alpha spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts, periods halving 4:2:1 (ringdown)
_HI0 = 1.70           # first peak: hottest AND longest exposure in the lineage
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # restoration burst height at run start, in alpha0 units
_A_B1 = 8.0           # restoration burst height at cool-down, in alpha0 units
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

    # --- lr: warmup -> 3 halving-period cosine restarts -> linear tail ---
    # u runs 0 -> _N_CYC over exploration with cycle lengths 4:2:1; u is an
    # integer at every cycle start and at fc = 1, so cos(2*pi*u) = 1 pins the
    # cool-down start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    r = 2.0 ** _N_CYC - 1.0
    u = _N_CYC - jnp.log2(1.0 + r * (1.0 - fc))
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * u))             # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + anti-phased growing bursts -> plateau -> terminal climb ---
    # burst = 1 exactly at lr troughs, ~0 near lr peaks (and 0 for frac >= _F_COOL,
    # since fc freezes at a peak), so restoration always coincides with low lr.
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
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