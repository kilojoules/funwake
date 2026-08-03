import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): abandon restarts and
# bursts entirely and combine the THREE remaining untried menu rows into one
# coherent design:
#
#   lr    — WSD / ONE-CYCLE (prior-art §6/§2, untried): short linear warmup,
#           then a LONG HOT HOLD at 1.8*D (the lineage's restarts only touch
#           their 1.65*D peak momentarily; a 26%-of-run hold spends an order
#           of magnitude more time at maximum exploration — the literal
#           "higher/longer lr peak early" the parent guidance asks for), then
#           a single near-linear anneal from 30% onward landing exactly on
#           gamma_min at the last step. The anneal is far longer than the
#           parent's 62%-tail, so the extra heat is paid for with extra
#           convergence time, one-cycle style.
#   alpha — ε-CONSTRAINED CONTRACTING TOLERANCE BAND (prior-art §7.9, the one
#           alpha mechanism no ancestor has used): no bursts, no logistic, no
#           plateau. The enforced violation band gamma(t) starts at ~D (alpha
#           at the proven 0.4*alpha0 exploration floor) and contracts
#           GEOMETRICALLY to gamma_min/TERM_GAIN, i.e. alpha = 0.4*alpha0 *
#           exp(s * log(5*D/(0.4*gmin))) with s = u^3 back-loading the
#           contraction. This single smooth curve reproduces the proven
#           feasible envelope at its checkpoints — ~floor until midway,
#           ~5-6*alpha0 (the old plateau level) near 78%, and EXACTLY the
#           5/5-seed-proven terminal 5*alpha0*D/gamma_min at the last step —
#           so the terminal feasibility restoration is preserved by
#           construction while everything between is one monotone
#           ε-contraction instead of hand-stitched phases.
#   beta1 — ONE-CYCLE ANTI-CORRELATION with lr (§2, untried): low momentum
#           (0.06) while steps are huge so overshoot never carries turbines
#           deep across the boundary, rising toward 0.12 as lr anneals to
#           accelerate the polish, then the proven late gate to 0.02 during
#           the terminal alpha spike.
#   beta2 — proven fast-adapt 0.2 while hot -> 0.9 for the anneal, logistic
#           transition centered where lr has cooled to ~1.2*D.
_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_DECAY = 0.30       # hot hold ends here; single linear anneal to gamma_min
_HI = 1.8             # hold lr, in units of D — hotter AND far longer at peak
_A_FLOOR = 0.4        # exploration penalty floor, in alpha0 units (proven)
_POW = 3.0            # cubic back-loading of the band contraction (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 rises once lr has annealed to ~1.2*D
_B2_WIDTH = 0.06
_B1_HOT = 0.06        # momentum at full heat (one-cycle: low beta1 at high lr)
_B1_MID = 0.12        # momentum as lr anneals (one-cycle: rises as lr falls)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.90
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> long hot hold at _HI*D -> single linear anneal ---
    # p = 0 through the hold, then ramps linearly to 1; lr lands exactly on
    # gamma_min at the final step.
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - p)
    warm = jnp.minimum(frac / _F_WARM, 1.0)   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: epsilon-constrained geometrically contracting band ---
    # Enforced tolerance contracts from ~D to gamma_min/_TERM_GAIN; the cubic
    # back-loading keeps alpha near the exploration floor until midway, passes
    # the old plateau level (~5-6*alpha0) near 78%, and lands exactly on the
    # proven terminal 5*alpha0*D/gamma_min at the last step.
    s = p ** _POW
    logR = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_FLOOR), 1.0))
    alpha = alpha0 * _A_FLOOR * jnp.exp(s * logR)

    # --- betas: one-cycle beta1 anti-correlated with lr + proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    r = lr / (_HI * Dj)                       # 1 at full heat, ->0 as lr anneals
    b1e = _B1_HOT + (_B1_MID - _B1_HOT) * (1.0 - r)
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1e + (_B1_LO - b1e) * b1r

    return lr, alpha, beta1, beta2