import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the MOMENTUM
# dimension — untouched across the entire lineage (beta1 has never left
# [0.02, 0.1]) — is opened up as ONE-CYCLE MOMENTUM ANTI-CORRELATED WITH lr
# (prior-art §2 + menu bet 4 + the untried "standard Adam vs TopFarm" ablation),
# while the proven exploration/restoration/endgame skeleton is kept and pushed
# hotter exactly as the parent guidance licenses.
#
#   lr    — proven decaying-peak SGDR restarts, hotter and longer: 3% warmup,
#           three cosine cycles with peaks decaying 1.85*D -> 1.05*D (first
#           peak well above the tried 1.65*D ceiling — the per-cycle
#           restoration bursts repay its violation debt, so the extra heat is
#           licensed), bounded 0.65*D troughs, exploration extended to 64%,
#           then the proven straight linear tail landing exactly on gamma_min
#           at the last step.
#   alpha — the proven best's machinery unchanged in mechanism: 0.4*alpha0
#           floor at lr peaks; anti-phased sharpened-cosine restoration bursts
#           at each lr trough, strengthened 3.5 -> 8*alpha0 across cycles
#           (first burst slightly stronger to cover the hotter first peak);
#           logistic ramp onto the bounded 6*alpha0 ALM plateau after
#           cool-down; cubic-delayed geometric terminal climb from 78% landing
#           on the 5/5-seed-feasible 5*alpha0*D/gamma_min.
#   betas — THE NEW AXIS. beta1 runs a full one-cycle anti-correlated with lr:
#           0.1 while hot (keeping the proven 0.05 dips inside restoration
#           bursts so momentum never drags turbines back over the boundary),
#           then rises to 0.85 for the POLISH window (~64%-86%) — near-
#           standard-Adam momentum smooths the shallow AEP valleys exactly
#           when the layout is already near-feasible, alpha sits on its
#           bounded plateau, and lr is on its linear tail; this is where the
#           lineage's flat 0.1 momentum has been leaving AEP on the table —
#           and finally the proven gate down to ~0.02 (moved slightly earlier
#           and sharper, center 86%) so polish momentum never rides into the
#           diverging terminal alpha. beta2 makes the matching two-stage climb
#           0.2 -> 0.9 (proven, at cool-down) -> 0.97 (mid-polish, toward
#           standard Adam's long-memory adaptive scaling; the slightly stale
#           denominator even sharpens terminal restoration steps while lr is
#           already tiny).
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.64        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase
_HI0 = 1.85           # first restart peak — hotter than the tried 1.65*D
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.5           # first restoration burst height (raised for the hotter peak)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_Q = 3.0              # sharpens bursts so alpha is low most of each cycle
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.68      # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_MID = 0.9         # proven post-cool-down beta2
_B2_HI = 0.97         # NEW second stage toward standard Adam in the polish
_B2_CENTER = 0.64     # first beta2 transition, aligned with cool-down
_B2_WIDTH = 0.05
_B2_CENTER2 = 0.74    # second beta2 stage, mid-polish
_B2_WIDTH2 = 0.04
_B1_HI = 0.1          # native momentum while exploring
_B1_BURST = 0.05      # proven reduced momentum inside each restoration burst
_B1_POLISH = 0.85     # NEW near-standard-Adam momentum for the polish window
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_UP_CENTER = 0.70  # one-cycle momentum rise, just after cool-down
_B1_UP_WIDTH = 0.03
_B1_CENTER = 0.86     # terminal momentum gate — earlier/sharper than proven 0.88
_B1_WIDTH = 0.025     # so polish momentum is fully drained before alpha diverges


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

    # --- betas: one-cycle momentum, anti-correlated with lr ---
    b2r1 = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    b2r2 = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER2) / _B2_WIDTH2))
    beta2 = _B2_LO + (_B2_MID - _B2_LO) * b2r1 + (_B2_HI - _B2_MID) * b2r2

    b1_exp = _B1_HI - (_B1_HI - _B1_BURST) * burst            # dips while repairing
    b1u = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = b1_exp + (_B1_POLISH - b1_exp) * b1u             # one-cycle rise for polish
    b1d = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1d                  # terminal gate

    return lr, alpha, beta1, beta2