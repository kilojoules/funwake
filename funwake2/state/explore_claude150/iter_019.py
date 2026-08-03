import jax.numpy as jnp

# STRUCTURAL BREAK from the restart/burst family that has plateaued at
# +0.0533%: the lineage's exploration phase has ALWAYS been oscillatory
# (SGDR restarts, decaying peaks, anti-phased bursts), and the last eight
# attempts show that reshaping those cycles is exhausted. This schedule
# abandons cycles entirely for the strongest UNTRIED row of the prior-art
# menu (§6/§2): a WSD / one-cycle backbone — warmup, a long HOT HOLD at a
# constant c*D, then ONE near-linear cool-down to gamma_min — composed with
# the two other untried menu rows (one-cycle momentum anti-correlated with
# lr, §2/§4; a smooth shrinking-tolerance penalty contraction, §7.9).
#
#   lr    — 4% linear warmup -> FLAT HOLD at 1.45*D until 36% (a larger
#           hot-lr time-integral than any restart scheme achieved, yet never
#           exceeding the 1.65*D ceiling already proven safe) -> a single
#           straight linear cool-down landing EXACTLY on gamma_min at the
#           last step. No re-heating: one slow anneal, with far more steps
#           spent at each intermediate lr scale than the parent's 38% tail.
#   alpha — delayed shrinking-band contraction instead of plateau+spike:
#           an exploration floor of 0.5*alpha0 through the entire hold and
#           early cool-down (basin selection is free), then from 50% a
#           quadratic rise to a moderate 4*alpha0 knee at 80% (ALM-scale,
#           bounded), then a single back-loaded geometric contraction of the
#           enforced violation band that lands on the 5/5-seed-proven
#           terminal 5*alpha0*D/gamma_min at the final step. The terminal
#           feasibility restoration (huge alpha + tiny lr + gated-down
#           momentum in the last ~15%) is fully preserved.
#   betas — the untried one-cycle bet: beta1 stays at the native 0.1 while
#           lr is hot, then RAMPS UP to 0.85 once lr falls below ~0.75*D —
#           high momentum at low lr accelerates polish along shallow AEP
#           valleys and acts as an implicit ALM multiplier, letting the
#           moderate 4*alpha0 knee enforce constraints — and is gated down
#           to 0.02 inside the terminal spike (proven) so momentum never
#           re-injects violations. beta2 leaves the native 0.2 at the start
#           of the cool-down (not at 62%) and rises to 0.95 for a long,
#           smooth adaptive polish phase.
_F_WARM = 0.04       # linear lr warmup over the first 4%
_F_DEC = 0.36        # hot hold ends here; single linear decay to gamma_min at 100%
_HI = 1.45           # hold lr, in units of D — hot, but under the tried 1.65 ceiling
_F_PEN = 0.50        # alpha stays at the exploration floor until here (delayed ramp)
_A_LO = 0.5          # exploration penalty floor, in alpha0 units
_A_KNEE = 4.0        # bounded moderate ALM knee reached at _F_TERM, in alpha0 units
_F_TERM = 0.80       # terminal geometric contraction starts here (proven start)
_POW = 2.5           # back-loads the terminal climb
_TERM_GAIN = 5.0     # terminal alpha = 5*alpha0*D/gamma_min (5/5-seed-proven scale)
_B2_LO = 0.2         # native beta2 while exploring
_B2_HI = 0.95        # long-memory adaptive scaling for the polish phase
_B2_CENTER = 0.40    # beta2 transition aligned with the start of the cool-down
_B2_WIDTH = 0.05
_B1_EXPLORE = 0.1    # native momentum while lr is hot
_B1_POLISH = 0.85    # one-cycle: high momentum once lr is low
_B1_UP_CENTER = 0.68 # momentum ramps up where lr has fallen to ~0.75*D
_B1_UP_WIDTH = 0.05
_B1_TERM = 0.02      # near-zero momentum during the terminal alpha spike (proven)
_B1_DN_CENTER = 0.88
_B1_DN_WIDTH = 0.025


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hot hold at 1.45*D -> single linear tail ---
    p = jnp.clip((frac - _F_DEC) / (1.0 - _F_DEC), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - p)             # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> quadratic rise to bounded knee -> geometric contraction ---
    knee = jnp.clip((frac - _F_PEN) / (_F_TERM - _F_PEN), 0.0, 1.0) ** 2
    alpha_units = _A_LO + (_A_KNEE - _A_LO) * knee
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_KNEE), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- betas: one-cycle momentum (up as lr cools, gated down at the spike) ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1_up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_base = _B1_EXPLORE + (_B1_POLISH - _B1_EXPLORE) * b1_up
    b1_dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_base + (_B1_TERM - b1_base) * b1_dn

    return lr, alpha, beta1, beta2