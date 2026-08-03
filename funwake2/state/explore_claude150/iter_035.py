import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the cycle
# GEOMETRY itself changes. Every restart schedule in the lineage used
# UNIFORM-length cycles; this is canonical SGDR with DOUBLING PERIODS
# (Loshchilov & Hutter T_mult=2), realized traceably by warping the cosine
# phase with phi = log2(1 + 7*fc), so the three exploration cycles have
# lengths 1:2:4. Mechanism for more AEP: several cheap, very hot basin hops
# packed into the first ~9% of the run (a 1.75*D first peak — hotter than any
# attempt, licensed by an immediate repair burst at ~4%), then one LONG slow
# anneal (over half the exploration phase) that actually converges inside the
# chosen basin before the polish tail begins — uniform cycles never give the
# final basin that settling time. The entire proven feasibility machinery is
# preserved untouched: anti-phased growing repair bursts at every trough,
# linear tail landing exactly on gamma_min, logistic ramp to the bounded
# 6*alpha0 ALM plateau, cubic-delayed terminal climb to 5*alpha0*D/gamma_min,
# and the proven beta transitions (beta2 0.2->0.9 at cool-down; beta1 dipped
# in bursts and gated to 0.02 in the terminal spike).
#
#   lr    — 2% warmup (shortened so the brief first cycle isn't fully damped)
#           -> three phase-warped cosine restarts, periods 1:2:4, peaks
#           decaying linearly PER CYCLE 1.75*D -> 1.05*D, bounded troughs at
#           0.65*D -> proven straight linear tail from the coolest peak to
#           exactly gamma_min at the last step.
#   alpha — 0.4*alpha0 exploration floor at lr peaks; sharpened restoration
#           bursts exactly at lr troughs (now at ~4%, ~16%, ~41% of the run:
#           debt is repaid early and often, and the deepest repair sprint
#           sits inside the long final anneal where the low lr is spent
#           purely pulling turbines feasible). Burst height grows per cycle
#           (3 -> 8 alpha0 in warped phase, matching the proven trough
#           magnitudes). After 62% the proven logistic ramp lifts alpha to
#           the bounded 6*alpha0 plateau and the proven cubic-delayed
#           geometric climb from 78% lands on 5*alpha0*D/gamma_min.
#   betas — exactly the proven 5/5-feasible transitions from the parent.
_F_WARM = 0.02        # short linear lr warmup (first cycle is only ~9% long)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_PHI_MAX = 3.0        # three cycles in warped phase
_T_GEO = 7.0          # 2**_PHI_MAX - 1: encodes doubling periods 1:2:4
_HI0 = 1.75           # first (shortest, hottest) restart peak, in units of D
_HI1 = 1.05           # final peak; the proven linear tail starts here
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units
_A_B1 = 8.0           # last restoration burst height, in alpha0 units
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

    # --- lr: warmup -> doubling-period cosine restarts -> linear tail ---
    # phi runs 0 -> 3 over the exploration phase; cycle k occupies phi in
    # [k, k+1], so cycle lengths in real time are 1:2:4. phi is an integer at
    # every peak, and fc freezes at 1 past _F_COOL, so cos(2*pi*3) = 1 pins
    # the cool-down start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    phi = jnp.log2(1.0 + _T_GEO * fc)
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * phi))   # 1 at peaks, 0 at troughs
    u = phi / _PHI_MAX                                # per-cycle progress, 0 -> 1
    hi = _HI0 + (_HI1 - _HI0) * u                     # equal peak decrement per cycle
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)       # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)           # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + anti-phased growing bursts -> plateau -> terminal climb ---
    # burst = 1 exactly at lr troughs, ~0 near lr peaks (and 0 for the whole
    # tail, since fc freezes at a peak), so restoration always coincides with
    # low lr. Growth in warped phase u keeps trough magnitudes on the proven
    # ~3.8 / 5.5 / 7.2 alpha0 ladder.
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * u
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst beta1 dip (unchanged) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst        # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2