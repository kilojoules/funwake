import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the TWO menu
# directions the lineage has never combined — a WSD/one-cycle TRAPEZOID lr
# (§6/§2: "hold near c*D, then (near-)linear cool-down beats cosine/product
# decay") and an ADMM-STYLE CONSTANT MODERATE PENALTY (§7.2, explicitly listed
# as untried). Every parent so far oscillates: lr restarts trade heat for
# trough dead-time (mean exploration lr only ~1.0*D despite 1.65*D peaks),
# and alpha whipsaws between a 0.4 floor and 3-8x bursts. Hypothesis: the
# SUSTAINED version of both wins — an unbroken hot hold delivers ~27% more
# exploration energy than the decaying-peak SGDR skeleton without ever
# dropping to a 0.65*D trough, and a steady moderate penalty is the
# better-conditioned way to keep violation debt bounded than intermittent
# repayment bursts (augmented-Lagrangian reasoning: a constant restoring
# force prevents debt from ever accumulating, so nothing needs bursting).
#
#   lr    — 3% linear warmup -> HOT TILTED HOLD, linear from 1.5*D down to
#           1.05*D at 62% (early steps hottest, where basin hops pay; the
#           tilt hands off at exactly the proven 1.05*D cool-down entry) ->
#           the proven straight linear tail landing exactly on gamma_min at
#           the last step. No restarts, no troughs: piecewise-linear
#           trapezoid, structurally disjoint from every cosine ancestor.
#   alpha — ADMM-style CONSTANT 1.3*alpha0 through the whole exploration
#           phase: no floor, no bursts, no coupling to lr. Steadily higher
#           than the parent's 0.4 floor, it holds the layout near-feasible
#           while the hot hold explores, so the endgame inherits little
#           debt. From 62% the PROVEN 5/5-seed endgame takes over verbatim:
#           logistic ramp to the bounded 6*alpha0 ALM plateau, then the
#           cubic-delayed geometric climb from 78% landing on the proven
#           terminal 5*alpha0*D/gamma_min feasibility spike.
#   betas — proven transitions only: beta2 0.2 -> 0.9 at the cool-down
#           start (adaptive scaling while hot, smoothing while polishing),
#           beta1 0.1 -> 0.02 in the terminal spike so momentum never
#           carries turbines back across the boundary. No per-burst dips —
#           there are no bursts to dip for.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.62        # hold ends here; linear decay to gamma_min at 100% (proven)
_HI0 = 1.5            # hold entry lr, in units of D — hottest sustained start tried
_HI1 = 1.05           # hold exit lr = proven cool-down entry point (proven)
_A_EXPL = 1.3         # ADMM-style constant exploration penalty, in alpha0 units
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp centered just after cool-down start (proven)
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start (proven)
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

    # --- lr: warmup -> tilted hot hold -> linear tail to gamma_min ---
    # fc freezes at 1 past _F_COOL, pinning the tail's start at _HI1 * D.
    fc = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    hold = (_HI0 + (_HI1 - _HI0) * fc) * Dj                   # unbroken hot hold, gently tilted
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hold - gmin) * (1.0 - p)                 # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> proven plateau -> terminal climb ---
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_EXPL + (_A_PLAT - _A_EXPL) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions, nothing else ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2