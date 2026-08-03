import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst champion (+0.0533%): the lr
# backbone is no longer SGDR cosine cycling at all — it is the untried
# WARMUP-STABLE-DECAY (WSD) shape from the prior-art menu (§6: "hold near
# c*D, then (near-)linear cool-down beats cosine/product decay"), and the
# penalty is the untried EPSILON-CONSTRAINED CONTRACTING TOLERANCE BAND
# (§7.9): alpha follows alpha0 * D/gamma(t) with gamma(t) shrinking
# geometrically so the enforced violation band reaches gamma_min only at
# the very end. The champion's one proven *mechanism* is transplanted, not
# copied: its repair-at-low-lr events become NARROW NOTCHES carved into the
# hot hold, so the run spends far more time at peak lr than any cosine
# cycle can (cosine wastes half of every cycle in transition), while the
# proven repair windows and the proven terminal restoration are preserved.
#
#   lr    — 3% linear warmup -> HOT HOLD with a mild decaying envelope
#           1.55*D -> 1.30*D until 58% (time-at-hot ~15% above the champion,
#           and above its 1.05-1.65*D peak envelope for the entire hold),
#           interrupted only by three narrow sin^8 repair notches dipping to
#           0.55*D; then the proven straight linear tail from 1.30*D landing
#           exactly on gamma_min at the last step. The notch waveform is 0 at
#           fh=1, pinning the tail start at the full hold lr.
#   alpha — during the hold: the champion's proven anti-phased repair bursts
#           (growing 3 -> 8 alpha0), now confined to the notches so basin
#           descent between repairs is uninterrupted. Underneath and after:
#           the contracting band  alpha = 0.4*alpha0 * exp(u * ln(5*D/(0.4*
#           gamma_min))),  u = ((frac-0.30)/0.70)_+^3.5 — exploratory
#           (~0.5 alpha0) through the hold, gentler than the champion's 6*
#           plateau in the 62-80% polish window (more AEP freedom exactly
#           where lr is still warm), then rising smoothly through and beyond
#           it to land on the 5/5-seed-proven terminal 5*alpha0*D/gamma_min
#           — a stronger, smoother terminal restoration than the champion's
#           cubic spike, repaying the hotter hold's debt.
#   betas — proven transitions kept: beta2 logistic 0.2 -> 0.9 at the hold
#           end; beta1 0.1 with the proven dip to 0.05 inside each repair
#           notch and the proven terminal gate to 0.02 at 88%.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.58        # hot hold ends here; linear decay to gamma_min at 100%
_N_NOTCH = 3.0        # three repair notches inside the hold
_Q = 4.0              # sin^8 notch: narrow (~5% of run each), hold stays hot
_HOLD0 = 1.55         # hold envelope start, in units of D — hotter than any peak held before
_HOLD1 = 1.30         # hold envelope end; the linear tail starts from here
_NOTCH_LO = 0.55      # lr at the bottom of each repair notch, in units of D
_A_LO = 0.4           # exploration penalty floor, in alpha0 units (proven)
_A_B0 = 3.0           # first repair burst height, in alpha0 units (proven)
_A_B1 = 8.0           # last repair burst height, in alpha0 units (proven)
_A_DELAY = 0.30       # contraction band is flat-at-floor before this frac
_A_POW = 3.5          # back-loads the geometric band contraction
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the hold end
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_NOTCH = 0.05      # reduced momentum inside each repair notch (proven)
_B1_LO = 0.02         # near-zero momentum during the terminal restoration
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> notched hot hold -> linear tail to gamma_min ---
    # fh freezes at 1 past _F_HOLD; cos(2*pi*N) = 1 there makes notch = 0,
    # pinning the tail start at the full hold-end lr, _HOLD1 * D.
    fh = jnp.clip(frac, 0.0, _F_HOLD) / _F_HOLD
    notch = (0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * _N_NOTCH * fh))) ** _Q  # 1 at repair centers, ~0 elsewhere
    hold_env = (_HOLD0 + (_HOLD1 - _HOLD0) * fh) * Dj        # mildly decaying hot hold
    lr_hold = hold_env - (hold_env - _NOTCH_LO * Dj) * notch # carve the repair notches
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)             # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                  # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: contracting tolerance band + growing repair bursts in the notches ---
    # Band: geometric climb from the 0.4*alpha0 floor to the proven terminal
    # 5*alpha0*D/gamma_min, back-loaded so it stays exploratory through the
    # hold and gentler than the old plateau early in the polish phase.
    u = jnp.clip((frac - _A_DELAY) / (1.0 - _A_DELAY), 0.0, 1.0) ** _A_POW
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    band = _A_LO * jnp.exp(u * log_ratio)                    # ends at 5*D/gmin alpha0-units
    burst_amp = _A_B0 + (_A_B1 - _A_B0) * fh                 # bursts strengthen per notch
    alpha = alpha0 * (band + burst_amp * notch)              # notch = 0 past the hold

    # --- betas: proven transitions + per-notch beta1 dip ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_exp = _B1_HI - (_B1_HI - _B1_NOTCH) * notch           # dip while repairing
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2