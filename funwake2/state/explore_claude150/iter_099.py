import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): the top untried lr row
# of the prior-art menu — WSD / ONE-CYCLE (warmup -> sustained hold near c*D
# -> near-linear cool-down to gamma_min, §2/§6) — paired with a fully
# DECOUPLED graduated dynamic penalty (§7.9) in place of cyclic bursts.
#
# Rationale: every restart-style parent spends most of exploration well below
# its peaks (mean lr ≈ 1.0*D across the cosine cycles; the 1.65*D peak lasts
# moments).  Here the ENTIRE exploration budget is held hot — a gently tilted
# plateau 1.35*D -> 1.05*D for 62% of the run — so the average exploration
# temperature is higher than any lineage member, while the momentary maximum
# (1.35*D) stays well inside the already-survived 1.65*D envelope.  From 62%
# onward the schedule is IDENTICAL to the proven 5/5-seed-feasible endgame,
# and pre-burst ancestors repaid their full violation debt with that endgame
# alone, so feasibility does not depend on mid-run repairs.
#
#   lr    — 3% linear warmup; tilted hot plateau 1.35*D -> 1.05*D until 62%
#           (WSD "stable" phase, ending exactly at the proven cool-down start
#           value 1.05*D); then the proven straight linear tail landing
#           exactly on gamma_min at the last step.
#   alpha — DECOUPLED from lr everywhere.  A graduated quadratic floor
#           0.4 -> 1.2 alpha0 across the stable phase (dynamic penalty
#           alpha0*(1+Ct)^p, §7.9) keeps late-exploration violations loosely
#           tethered without ever interrupting basin hops with bursts; then
#           the proven logistic ramp to the bounded 6*alpha0 ALM plateau and
#           the proven cubic-delayed geometric climb from 78% landing on the
#           5/5-seed-feasible terminal 5*alpha0*D/gamma_min.
#   betas — proven transitions only (beta2 0.2 -> 0.9 at cool-down; beta1
#           gated 0.1 -> 0.02 in the terminal spike); a burst-free body needs
#           no per-cycle momentum dips.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # stable phase ends here; linear decay to gamma_min at 100%
_HI0 = 1.35           # plateau start, in units of D — hot but inside tried envelope
_HI1 = 1.05           # plateau end; the proven linear tail starts from here
_A_LO = 0.4           # exploration penalty floor at t=0, in alpha0 units (proven)
_A_MID = 1.2          # graduated floor value reached at cool-down (dynamic penalty)
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
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold (WSD stable phase) -> linear tail ---
    # fs freezes at 1 past _F_COOL, pinning the cool-down start at _HI1 * D.
    fs = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    lr_stable = (_HI0 + (_HI1 - _HI0) * fs) * Dj              # sustained tilted plateau
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_stable - gmin) * (1.0 - p)            # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: graduated floor -> logistic plateau -> terminal climb ---
    # Quadratic back-loaded floor: gentle early freedom, mild tether late in
    # the hold; freezes at _A_MID past cool-down so the endgame matches the
    # proven trajectory (floor + plateau ramp -> exactly 6*alpha0).
    floor = _A_LO + (_A_MID - _A_LO) * fs ** 2
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = floor + (_A_PLAT - _A_MID) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2