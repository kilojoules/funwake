import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the restart
# GEOMETRY itself changes — HALVING-PERIOD SGDR (cycle lengths 4:2:1 inside
# the exploration phase) instead of the parent's three equal cycles. This
# directly implements the guidance "higher/longer lr peak early" as structure,
# not as a constant tweak (which the last 8 attempts prove is exhausted):
#
#   lr    — 3% warmup, then three cosine restarts whose periods HALVE
#           (boundaries at 4/7, 6/7, 1 of the exploration phase, realized
#           traceably via a log2 time warp). The FIRST cycle is both the
#           hottest and the LONGEST: ~35% of the run sweeping from a peak
#           above anything tried down to the proven 0.65*D trough — coarse
#           basin search gets the lion's share of hot steps. The first peak
#           is SPACING-AWARE (an untried input): half the min-spacing scale,
#           clipped to [1.85, 2.15]*D, so basin hops scale with the constraint
#           geometry on unseen farms while staying bounded. Later cycles are
#           short, cool refine/repair flicks. fc freezes at a peak, so the
#           proven straight linear tail starts from 1.05*D and lands exactly
#           on gamma_min at the last step.
#   alpha — the proven anti-phased restoration bursts ride the SAME warped
#           clock, so repair still coincides exactly with lr troughs — but
#           the halving periods move the LAST burst to ~58% of the run, right
#           against the cool-down (parent: 52%), so the layout enters the
#           polish phase with its violation debt just repaid. Burst strength
#           grows 3 -> 9 alpha0 (last burst stronger, paying for the hotter,
#           longer first cycle). Endgame untouched and proven: logistic ramp
#           to the bounded 6*alpha0 ALM plateau, cubic-delayed geometric climb
#           from 78% landing on the 5/5-seed-feasible 5*alpha0*D/gamma_min.
#   betas — all proven: beta2 0.2 -> 0.9 at cool-down; beta1 0.1 with per-burst
#           dips to 0.05 (momentum never drags turbines back over the boundary
#           mid-repair) and the gated drop to 0.02 in the terminal spike.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_HI0 = 1.85           # floor of the first-cycle peak, in D units (hotter than tried)
_HI0_CAP = 2.15       # ceiling of the spacing-aware first peak, in D units
_HI_SPC = 0.5         # first peak ~ half the spacing-constraint scale
_HI1 = 1.05           # final peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units
_A_B1 = 9.0           # last burst height — stronger, repaying the hotter start
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
    ms = jnp.asarray(min_spacing) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 halving-period cosine restarts -> linear tail ---
    # Log2 time warp: u hits integers at cycle boundaries fc = 0, 4/7, 6/7, 1,
    # giving cycle lengths in ratio 4:2:1. u = 3 exactly at fc = 1, so
    # cos(2*pi*u) = 1 pins the cool-down start at the final (coolest) peak.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    u = -jnp.log2(1.0 - 0.875 * fc)                           # warped cycle phase
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * u))             # 1 at peaks, 0 at troughs
    hi0 = jnp.clip(_HI_SPC * ms / Dj, _HI0, _HI0_CAP)         # spacing-aware, bounded
    hi = hi0 + (_HI1 - hi0) * fc                              # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + anti-phased growing bursts -> plateau -> terminal climb ---
    # burst = 1 exactly at lr troughs (fc = 0.335, 0.739, 0.941 under the warp),
    # ~0 near lr peaks, and 0 for frac >= _F_COOL since fc freezes at a peak —
    # the last repair sits at ~58% of the run, hard against the cool-down.
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