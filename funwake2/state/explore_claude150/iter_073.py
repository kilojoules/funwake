import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the two menu rows
# still untouched anywhere in the lineage.
#
# (1) EPSILON-CONSTRAINED SHRINKING BAND (prior-art §7.9) replaces the whole
#     {logistic ramp -> 6*alpha0 plateau -> cubic exp spike} endgame with ONE
#     continuous law: alpha = alpha0 * A(w), A contracting the enforced
#     violation band geometrically from ~D/(3) scale down to gamma_min/5, so
#     the tolerance reaches gamma_min ONLY at the last step. It starts laxer
#     than the plateau (AEP room while the tail is still hot) and overtakes
#     the parent's trajectory from ~80% on (MORE terminal restoration
#     authority), landing on the exact proven 5*alpha0*D/gamma_min.
# (2) FEASIBLE HOT POLISH — a NON-MONOTONE tail. Every parent decays lr
#     monotonically after cool-down; here a single Gaussian lr bump
#     (+0.28*D at 78%) re-heats the polish phase WHILE alpha is already
#     ~10*alpha0, i.e. constrained refinement: turbines micro-adjust for AEP
#     under a penalty strong enough to keep them inside. The bump's
#     (lr, alpha) point sits strictly inside the envelope the parent already
#     survives at 66% (0.94*D lr at alpha ~3-6), so 5/5 feasibility is not
#     gambled.
#
# The proven AEP engine is kept intact: 3% warmup, three decaying-peak cosine
# restarts with anti-phased GROWING alpha restoration bursts (debt repaid
# every cycle), pushed slightly hotter per the parent guidance (first peak
# 1.65 -> 1.72*D), then the straight linear tail landing exactly on gamma_min.
# Betas keep the proven transitions (beta2 0.2 -> 0.9 at cool-down; beta1
# gated to 0.02 in the terminal spike; per-burst dip to 0.05) plus a matching
# beta1 dip to 0.06 inside the polish bump so momentum cannot carry the
# re-heated steps across the boundary.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # exploration ends here; linear decay to gamma_min at 100%
_N_CYC = 3.0          # three restarts inside the exploration phase (proven)
_HI0 = 1.72           # first restart peak — slightly hotter than the tried 1.65
_HI1 = 1.05           # last peak; the linear tail starts from here (proven)
_LO = 0.65            # bounded trough lr, in units of D (proven)
_A_LO = 0.4           # exploration penalty floor at lr peaks, in alpha0 units
_A_B0 = 3.0           # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last restoration burst height, in alpha0 units (proven)
_Q = 3.0              # sharpens bursts so alpha is low most of each cycle
_A_START = 3.0        # shrinking band entry level, in alpha0 units
_PB = 2.2             # delay power of the geometric band contraction
_BLEND_C = 0.64       # logistic handoff exploration-floor -> band
_BLEND_W = 0.02
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_P_AMP = 0.28         # polish bump height, in units of D
_P_C = 0.78           # polish bump center
_P_W = 0.05           # polish bump width (Gaussian)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_BURST = 0.05      # reduced momentum inside each restoration burst
_B1_POLISH = 0.06     # reduced momentum inside the hot polish bump
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> 3 decaying-peak restarts -> linear tail + polish bump ---
    # fc freezes at 1 past _F_COOL, so cos(2*pi*N) = 1 pins the cool-down
    # start at the final (coolest) peak, _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    cyc = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * _N_CYC * fc))   # 1 at peaks, 0 at troughs
    hi = _HI0 + (_HI1 - _HI0) * fc                            # decaying peak envelope
    lr_cyc = (_LO + (hi - _LO) * cyc) * Dj
    w = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - w)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    polish = jnp.exp(-(((frac - _P_C) / _P_W) ** 2))          # ~0 outside the bump
    lr = lr_env * warm + _P_AMP * Dj * polish                 # bump is <1e-8*D at frac=1

    # --- alpha: floor + anti-phased growing bursts -> shrinking epsilon-band ---
    # burst = 1 exactly at lr troughs, ~0 near lr peaks (and 0 for frac >= _F_COOL),
    # so restoration always coincides with low lr.
    burst = (1.0 - cyc) ** _Q
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc                  # bursts strengthen per cycle
    expl_units = _A_LO + burst_amp * burst
    # geometric band contraction: A_START at cool-down start, 5*D/gmin at the end,
    # delayed by w^_PB so most of the tightening happens late.
    ratio = jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_START), 1.0)
    band_units = _A_START * jnp.exp((w ** _PB) * jnp.log(ratio))
    blend = 1.0 / (1.0 + jnp.exp(-(frac - _BLEND_C) / _BLEND_W))
    alpha_units = expl_units * (1.0 - blend) + band_units * blend
    alpha = alpha0 * alpha_units                              # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-burst and per-bump beta1 dips ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_mid = _B1_HI - (_B1_HI - _B1_BURST) * burst            # dip while repairing
    b1_mid = b1_mid - (_B1_HI - _B1_POLISH) * polish          # dip while re-heated
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_mid + (_B1_LO - b1_mid) * b1r

    return lr, alpha, beta1, beta2