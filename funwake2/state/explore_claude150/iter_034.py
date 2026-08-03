import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the TWO menu
# directions no attempt has combined — a ONE-CYCLE / WSD lr (prior-art §2/§6:
# warmup -> hot HOLD -> near-linear cool-down; the frontier's explicit bet
# that a sustained hold beats cosine restarts) driving exploration, paired
# with an ADMM-STYLE CONSTANT MODERATE PENALTY (§7/§10: momentum acts as an
# implicit ALM multiplier, so a flat moderate alpha enforces constraints
# without bursts or 1/lr coupling). Every restart-family parent alternates
# hot/cold and repays violation debt in pulses; here there are NO pulses at
# all — one long, uninterrupted hot phase at a HIGHER lr than any peak yet
# (1.55*D held for 35% of the run, vs the best's brief 1.65*D spike averaging
# ~1.0*D), with debt serviced CONTINUOUSLY by the constant penalty.
#
#   lr    — 3% linear warmup (proven) -> HOLD at 1.55*D until 38% -> single
#           straight linear decay landing exactly on gamma_min at the last
#           step (the proven tail shape, extended to cover the whole descent;
#           §6's WSD hypothesis verbatim). Average exploration heat is well
#           above the restart best's, i.e. "higher AND longer peak" without
#           the AEP-structure-destroying whiplash of restarts.
#   alpha — DECOUPLED, FLAT, MODERATE: constant 2.0*alpha0 through the entire
#           hot phase — matched to the burst parent's TIME-AVERAGED penalty
#           pressure, so total constraint forcing is preserved while the
#           pulsing is removed. Then the endgame is kept VERBATIM from the
#           5/5-seed-feasible lineage: logistic ramp (center 0.66) to the
#           bounded 6*alpha0 ALM plateau, and the cubic-delayed geometric
#           climb from 78% landing on the terminal 5*alpha0*D/gamma_min
#           feasibility spike. Strict feasibility machinery untouched.
#   betas — proven transitions only: beta2 0.2 -> 0.9 (logistic, center 0.62,
#           where lr has cooled to ~1*D and alpha curvature is about to grow);
#           beta1 0.1 -> 0.02 (logistic, center 0.88) so momentum never fights
#           the terminal restoration. No per-burst dips — there are no bursts.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.38        # hot hold ends here; straight linear decay begins
_HI = 1.55            # held exploration lr, in units of D — hotter for longer
_A_CONST = 2.0        # ADMM-style flat penalty, in alpha0 units (= burst-era mean)
_A_PLAT = 6.0         # bounded ALM plateau, in alpha0 units (proven)
_A_CENTER = 0.66      # logistic alpha ramp center (proven)
_A_WIDTH = 0.04
_F_TERM = 0.78        # terminal geometric alpha climb starts here (proven)
_POW = 3.0            # cubic back-loading of the terminal climb (proven)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition where lr passes ~1*D (proven placement)
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum through exploration and polish
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot hold at 1.55*D -> linear decay to gamma_min ---
    # p = 0 through the hold (lr pinned at _HI*D), then rises linearly so the
    # envelope lands exactly on gamma_min at the final step (proven tail).
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - p)
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: flat moderate penalty -> plateau -> terminal geometric climb ---
    # Constant 2*alpha0 services violation debt continuously through the hot
    # hold; the proven logistic ramp then lifts to the bounded 6*alpha0
    # plateau, and the proven cubic-delayed climb collects what little debt
    # remains, ending at the 5/5-seed-feasible 5*alpha0*D/gamma_min.
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_units = _A_CONST + (_A_PLAT - _A_CONST) * ramp
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_PLAT), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions, nothing else ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2