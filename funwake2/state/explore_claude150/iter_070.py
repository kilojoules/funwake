import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): a MID-RUN
# OPTIMIZER-REGIME SWITCH — prior-art menu bet 4 ("phase-transition the Adam
# moments with the alpha phase") taken to its full, untried extent. Every
# schedule in the lineage has kept beta1 <= 0.1 (TopFarm-style Adam) for all
# 8000 steps; the moment space beta1 ~ 0.9 has never been visited. Here the
# run passes through THREE optimizers:
#
#   1. EXPLORE (0-62%)  — the proven engine, byte-for-byte: 3% warmup, three
#      decaying-peak cosine restarts (1.65*D -> 1.05*D, troughs 0.65*D),
#      anti-phased growing alpha restoration bursts (3 -> 8 alpha0) over the
#      0.4*alpha0 floor, native low-momentum betas (0.1/0.2) with the per-burst
#      beta1 dip to 0.05. Nothing perturbed — this is what earned +0.0533%.
#   2. POLISH (~66-80%) — STANDARD ADAM (beta1 -> 0.9, beta2 -> 0.98) while lr
#      rides the proven straight linear tail and alpha sits on the bounded
#      6*alpha0 ALM plateau. Hypothesis: once basin-hopping is over, the AEP
#      landscape is a smooth ravine problem where high momentum + long
#      adaptive horizon extracts refinement the low-momentum native settings
#      leave on the table — this is where the extra AEP is, not in hotter
#      peaks (eight straight peak/ramp/constant tweaks all failed).
#   3. FEASIBILITY LOCK (from 78%) — the proven 5/5-seed endgame, entered
#      early enough that momentum is drained BEFORE alpha grows: beta1 falls
#      0.9 -> 0.02 centered at 82% (alpha's cubic-delayed climb is still
#      within ~6% of the plateau there), so no stored velocity can carry
#      turbines across the boundary while the terminal spike — the proven
#      cubic-delayed geometric climb to 5*alpha0*D/gamma_min — collects the
#      remaining violation debt. lr lands exactly on gamma_min at the last
#      step; beta2 stays high so the constraint-curvature conditioning of the
#      spike is absorbed by the adaptive scaling.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear lr decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase (proven)
_HI0 = 1.65           # first restart peak, in units of D (proven)
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_Q = 3.0              # sharpens bursts so alpha is low most of each cycle
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2          # native adaptive scaling while exploring
_B2_HI = 0.98         # near-standard-Adam horizon for polish + lock (was 0.9)
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.04
_B1_EXPLORE = 0.1     # native momentum during restarts (proven for hot phases)
_B1_BURST = 0.05      # reduced momentum inside each restoration burst (proven)
_B1_POLISH = 0.9      # standard-Adam momentum — the untried regime
_B1_LOCK = 0.02       # near-zero momentum during the terminal alpha spike
_B1_UP_CENTER = 0.66  # momentum ramps UP once exploration is over
_B1_UP_WIDTH = 0.025
_B1_DN_CENTER = 0.82  # and drains BEFORE the alpha spike gets large
_B1_DN_WIDTH = 0.025


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: proven warmup -> 3 decaying-peak restarts -> straight linear tail ---
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

    # --- alpha: proven floor + anti-phased bursts -> plateau -> terminal climb ---
    # burst = 1 exactly at lr troughs, ~0 near lr peaks (and 0 for frac >= _F_COOL,
    # since fc freezes at a peak), so restoration always coincides with low lr.
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * burst
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: NEW three-regime moment schedule ---
    # explore: native 0.1 (with per-burst dip to 0.05) -> polish: standard-Adam
    # 0.9 -> lock: 0.02. Two logistic gates compose smoothly; the down-gate is
    # centered where the terminal alpha climb is still ~flat, so momentum built
    # during polish (decay horizon ~10 steps once beta1 falls) is gone before
    # alpha gets large. beta2 ramps once, at cool-down, to a near-standard 0.98.
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_EXPLORE - (_B1_EXPLORE - _B1_BURST) * burst  # dip while repairing
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_exp + (_B1_POLISH - b1_exp) * up + (_B1_LOCK - _B1_POLISH) * dn

    return lr, alpha, beta1, beta2