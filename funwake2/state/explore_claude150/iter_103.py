import jax.numpy as jnp

# STRUCTURALLY NEW vs both frontier members — a PUNCTUATED HOT PLATEAU:
# the two proven-but-separate winning mechanisms fused into one lr shape
# that neither parent has. The cyclic best (+0.0533%) showed mid-run
# feasibility-restoration bursts (prior-art §7.5) license extra heat, but
# its SGDR cosines waste most of the exploration span at mid lr. The WSD
# trapezoid (+0.0423%) showed sustained heat beats cycled heat, but repays
# violation debt only once, capping how hot it dares run. Here the plateau
# is SUSTAINED *and* the debt is repaid: a tilted hot shelf with three
# NARROW smooth NOTCHES carved into it, each notch anti-phased with a
# growing alpha restoration burst.
#
#   lr    — 3% linear warmup, then a tilted shelf 1.70*D -> 1.05*D over the
#           first 62% (hotter than the 1.65*D fleeting peak and the 1.55*D
#           sustained shelf — affordable because each notch clears the debt)
#           with three sharpened-cosine notches (width ~20% of each third)
#           dipping to ~0.45-0.65*D for repair; time-average lr ~1.2*D —
#           the hottest, longest exploration in the lineage. At 62% the
#           notch function is exactly zero, so the proven straight linear
#           tail departs cleanly from 1.05*D and lands exactly on gamma_min
#           at the last step.
#   alpha — 0.4*alpha0 exploration floor + restoration bursts locked to the
#           lr notches (growing 3 -> 8 alpha0 across notches, proven range:
#           gentle early repairs, near-strict late ones), then the proven
#           bounded endgame unchanged: logistic ramp to the 6*alpha0 ALM
#           plateau after 62%, cubic back-loaded geometric climb from 78%
#           landing on the 5/5-seed-feasible 5*alpha0*D/gamma_min terminal
#           spike at the final step.
#   betas — proven transitions only: beta2 0.2 -> 0.9 logistic at the
#           cool-down boundary; beta1 0.1 native, dipped to 0.05 inside
#           each repair notch (momentum must not carry turbines back across
#           the boundary mid-repair), gated to 0.02 in the terminal spike.
_F_WARM = 0.03      # linear lr warmup fraction (proven)
_F_COOL = 0.62      # exploration ends here; linear decay to gamma_min at 100%
_N_NOTCH = 3.0      # three repair notches carved into the shelf
_Q = 8.0            # sharpens notches: shelf holds full heat ~80% of the time
_HI0 = 1.70         # shelf start, in units of D — hottest sustained level tried
_HI1 = 1.05         # shelf end at _F_COOL; the linear tail starts here (proven)
_DEPTH = 0.6        # notch depth: lr dips to 40% of the local shelf value
_A_LO = 0.4         # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 3.0         # first restoration burst height, in alpha0 units (proven)
_A_B1 = 8.0         # last restoration burst height, in alpha0 units (proven)
_A_PLAT = 6.0       # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66    # logistic alpha ramp centered just after cool-down start
_A_WIDTH = 0.04
_F_TERM = 0.78      # terminal geometric alpha climb starts here (proven)
_POW = 3.0          # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0    # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62   # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HI = 0.1        # native momentum on the shelf and in the polish phase
_B1_NOTCH = 0.05    # reduced momentum inside each repair notch
_B1_LO = 0.02       # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot shelf with 3 narrow repair notches -> tail ---
    # fc freezes at 1 past _F_COOL, where the notch function is exactly zero,
    # so the cool-down departs from the clean shelf value _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    notch = (0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * _N_NOTCH * fc))) ** _Q
    hi = (_HI0 + (_HI1 - _HI0) * fc) * Dj                 # tilted shelf envelope
    lr_shelf = hi * (1.0 - _DEPTH * notch)                # carve the repair notches
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_shelf - gmin) * (1.0 - p)         # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)               # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor + notch-locked growing bursts -> plateau -> terminal climb ---
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fc              # bursts strengthen per notch
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_LO + (_A_PLAT - _A_LO) * ramp + burst_amp * notch
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + per-notch beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch        # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2