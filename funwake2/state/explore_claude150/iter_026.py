import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the ANTI-THESIS of
# the entire oscillatory lineage. Every strong parent (SGDR peaks, cyclic alpha,
# restoration bursts) spends a large fraction of the exploration phase in cold
# troughs and repays violation debt in pulses. This schedule has ZERO cycles.
# It combines the two frontier bets the survey flags as strongest-untried:
#
#   lr    — WSD (warmup–stable–decay, §6): 4% linear warmup, then a FLAT hot
#           hold at 1.40*D through 58% of the run — the integrated exploration
#           lr exceeds every parent's cosine-cycle average (~1.0*D) without
#           ever exceeding the feasibility-recoverable 1.65*D peak the lineage
#           already survived — then the proven straight linear cool-down that
#           lands exactly on gamma_min at the last step. No troughs means no
#           exploration steps wasted at near-zero lr mid-run.
#   alpha — ε-CONSTRAINED CONTRACTING TOLERANCE (§7.9, explicitly untried):
#           instead of plateau+spike or bursts, the enforced violation band
#           contracts geometrically to gamma_min only at the end. A logistic
#           blend (centered 0.62, just after the cool-down starts — delayed
#           ramp, §7.3/§7.5) lifts alpha off the 0.5*alpha0 exploration floor
#           onto the contraction track, which climbs CONTINUOUSLY and
#           monotonically from ~5*alpha0 to the proven 5/5-seed-feasible
#           terminal 5*alpha0*D/gamma_min. The quintic clock back-loads the
#           climb so alpha sits in the proven ALM-plateau band (~5–8*alpha0)
#           through most of the polish, then rockets in the final ~15% — the
#           terminal feasibility restoration is PRESERVED and, being
#           continuous rather than kicked off at 78%, starts collecting debt
#           earlier, which hedges the hotter/longer exploration.
#   betas — proven phase transitions only (menu bet 4): beta2 0.2 -> 0.9 at
#           the hold->decay boundary (adaptive scaling absorbs the ~alpha
#           constraint curvature); beta1 0.1 -> 0.02 gated at 0.86 — slightly
#           earlier than the parent's 0.88 because the alpha climb is
#           continuous — so momentum never fights the contraction.
_F_WARM = 0.04        # linear lr warmup fraction
_F_HOLD = 0.58        # flat hot phase ends; linear decay to gamma_min at 100%
_LR_HOT = 1.40        # stable exploration lr, in units of D
_A_LO = 0.5           # exploration penalty floor, in alpha0 units
_A_START = 5.0        # contraction track entry level, in alpha0 units
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_F_ENF = 0.55         # ε-contraction clock starts just before the lr decay
_POW = 5.0            # quintic back-loading of the geometric contraction
_A_CENTER = 0.62      # logistic blend floor -> contraction track
_A_WIDTH = 0.035
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the hold->decay boundary
_B2_WIDTH = 0.05
_B1_HI = 0.1          # native momentum while exploring and polishing
_B1_LO = 0.02         # near-zero momentum during the terminal alpha climb
_B1_CENTER = 0.86
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hot hold -> linear tail landing on gamma_min ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr = (gmin + (_LR_HOT * Dj - gmin) * (1.0 - p)) * warm

    # --- alpha: floor -> ε-contraction geometric climb to the terminal spike ---
    # s is the back-loaded contraction clock; the track interpolates
    # geometrically between _A_START*alpha0 and the proven terminal value, so
    # the enforced violation band shrinks to gamma_min exactly at the end.
    s = jnp.clip((frac - _F_ENF) / (1.0 - _F_ENF), 0.0, 1.0) ** _POW
    a_end = _TERM_GAIN * Dj / gmin                          # in alpha0 units
    log_ratio = jnp.log(jnp.maximum(a_end / _A_START, 1.0))
    track = _A_START * jnp.exp(s * log_ratio)
    ramp = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha = alpha0 * (_A_LO + (track - _A_LO) * ramp)       # ends at 5*alpha0*D/gmin

    # --- betas: proven phase transitions, no mid-run oscillation ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2