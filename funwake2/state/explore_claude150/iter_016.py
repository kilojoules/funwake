import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR-restart best (+0.0450%): the one search-state
# direction not yet tried anywhere in the lineage — MID-RUN FEASIBILITY-
# RESTORATION BURSTS (filter/funnel restoration, prior-art §7.5) — realized as
# a CYCLIC ALPHA ANTI-PHASED with the lr restarts. Every parent so far repays
# the hot phase's violation debt only ONCE, at the end; here the debt is
# repaid every cycle, which in turn licenses a hotter first restart than any
# attempt has dared.
#
#   lr    — proven decaying-peak SGDR skeleton, pushed hotter/longer as the
#           parent guidance asks: 3% warmup -> three cosine restarts with
#           peaks decaying 1.65*D -> 1.05*D (first peak above the tried
#           1.55*D ceiling), bounded troughs at 0.65*D, exploration extended
#           to 62%, then the proven straight linear tail landing exactly on
#           gamma_min at the last step.
#   alpha — ANTI-PHASED BURSTS, then the proven bounded endgame. At lr PEAKS
#           alpha sits at the 0.4*alpha0 exploration floor (basin hops trade
#           violation for AEP freely); at each lr TROUGH a sharpened cosine
#           burst drives alpha up so the near-zero step size is spent purely
#           on constraint repair — restoration exactly when it cannot destroy
#           AEP structure. Burst strength GROWS across cycles (3 -> 8 alpha0):
#           early repairs are gentle, late ones near-strict, so the layout
#           enters the polish phase already close to feasible and the terminal
#           spike has almost no debt left to collect. After 62% the proven
#           logistic ramp lifts alpha to the bounded 6*alpha0 ALM plateau, and
#           the proven cubic-delayed geometric climb from 78% lands it on the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions (beta2 0.2 -> 0.9 at cool-down; beta1 gated
#           0.1 -> 0.02 in the terminal spike), plus the per-cycle version of
#           menu bet 4: beta1 dips to 0.05 inside each restoration burst so
#           momentum never carries turbines back across the boundary while
#           the burst is pulling them in.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase
_HI0 = 1.65           # first restart peak — hotter than anything tried
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
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

    # --- lr: warmup -> 3 decaying-peak cosine restarts -> linear tail ---
    # fc freezes at 1 past _F_COOL, so cos(2*pi*N) = 1 pins the cool-down
    # start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 at peaks, 0 at troughs
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