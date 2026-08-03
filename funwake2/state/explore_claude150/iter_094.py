import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR-restart best (+0.0533%): this attempt takes the
# ONE search-state direction still untried anywhere in the lineage — an
# ADMM-STYLE CONSTANT MODERATE PENALTY (no floor-vs-plateau logistic, no
# bursts, no lr-coupling) — and pairs it with a ONE-CYCLE super-convergence lr
# (single hot peak, monotone anneal), which no parent has run either: every
# prior schedule was multi-peak (SGDR/multi-cosine) or a flat hold (WSD).
#
#   alpha — CONSTANT at 1.0*alpha0 for the whole run (ADMM fixed-rho: since
#           alpha0 = mean|grad J|/D, rho=1 makes the constraint force
#           commensurate with the objective force at every step — steady
#           pressure that never lets violation debt bank up, yet never chokes
#           basin hops the way the 6*alpha0 mid-run plateau could). Feasibility
#           is NOT entrusted to rho: the proven terminal restoration is kept
#           verbatim — a cubic-delayed geometric climb starting at 72% (3%
#           earlier than the best, hedging the absence of a mid-run plateau)
#           that lands alpha exactly on the 5/5-seed-feasible
#           5*alpha0*D/gamma_min at the last step.
#   lr    — ONE-CYCLE: 5% linear warmup to a 1.75*D peak (hotter than the
#           1.65*D ceiling any attempt has dared, licensed by the constant
#           penalty already leaning on the layout), then a single half-cosine
#           anneal down to 0.45*D at 70% — one long coherent descent instead
#           of restart churn, so late-mid steps polish one basin rather than
#           re-hopping — then the proven straight linear tail landing exactly
#           on gamma_min at the final step.
#   beta1 — menu bet §2 applied CONTINUOUSLY for the first time: momentum is
#           anti-correlated with lr through the whole cycle (0.04 at the
#           1.75*D peak so momentum cannot compound the hottest steps, easing
#           back to the native 0.1 as lr anneals), then the proven gate to
#           0.02 inside the terminal alpha spike.
#   beta2 — proven ramp kept: 0.2 while hot (fast adaptive reaction), logistic
#           rise to 0.9 centered at 60%, where the one-cycle anneal has already
#           cooled lr into the polish regime.
_F_WARM = 0.05         # linear lr warmup over the first 5%
_F_END = 0.70          # one-cycle cosine anneal ends here; linear tail after
_HI = 1.75             # single peak lr, in units of D — hottest yet tried
_LO = 0.45             # anneal floor at _F_END; tail runs 0.45*D -> gamma_min
_A_RHO = 1.0           # ADMM constant penalty, in alpha0 units, entire run
_F_TERM = 0.72         # terminal geometric alpha climb starts here
_POW = 3.0             # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0       # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.60      # beta2 rise centered where the anneal has cooled lr
_B2_WIDTH = 0.05
_B1_HI = 0.1           # native momentum once lr has annealed
_B1_MIN = 0.04         # momentum at the 1.75*D peak (anti-correlated with lr)
_B1_LO = 0.02          # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> single half-cosine anneal 1.75*D -> 0.45*D -> tail ---
    # q freezes at 1 past _F_END, pinning the tail's start at _LO * D.
    q = jnp.clip((frac - _F_WARM) / (_F_END - _F_WARM), 0.0, 1.0)
    lr_base = (_LO + (_HI - _LO) * 0.5 * (1.0 + jnp.cos(jnp.pi * q))) * Dj
    p = jnp.clip((frac - _F_END) / (1.0 - _F_END), 0.0, 1.0)
    lr_env = gmin + (lr_base - gmin) * (1.0 - p)     # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)          # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: ADMM constant rho -> proven cubic-delayed terminal climb ---
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_RHO), 1.0))
    alpha = _A_RHO * alpha0 * jnp.exp(s * log_term)  # ends at 5*alpha0*D/gmin

    # --- betas: beta1 anti-correlated with lr; proven beta2 ramp + endgame ---
    lr_norm = jnp.clip((lr_base / Dj - _LO) / (_HI - _LO), 0.0, 1.0)
    b1_expl = _B1_HI - (_B1_HI - _B1_MIN) * lr_norm  # 0.04 at peak, 0.1 annealed
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_expl + (_B1_LO - b1_expl) * b1r
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2