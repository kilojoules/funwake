import jax.numpy as jnp

# STRUCTURALLY NEW vs both the SGDR-burst best (+0.0533%) and the logistic-
# plateau parent (+0.0450%): NO cycles, NO bursts, NO plateau. This is the
# two remaining untried menu rows, composed:
#
#   lr    — WSD / ONE-CYCLE TRAPEZOID (prior-art §6/§2, the top untried lr
#           bet: "hold near c*D then (near-)linear cool-down beats cosine").
#           3% warmup -> a SUSTAINED tilted hold 1.45*D -> 1.10*D until 55%
#           -> the proven straight linear tail landing exactly on gamma_min
#           at the last step. Every prior schedule only TOUCHES ~1.3-1.65*D
#           at cosine peaks; here the layout spends the entire exploration
#           phase at that heat (~15% more integrated lr than the best run),
#           bouncing along the valley walls, with a single consolidation:
#           the tail. This is "a higher/LONGER lr peak early" taken to its
#           structural limit rather than another peak-constant tweak.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (the one search-state
#           direction never tried) fused with the eps-CONSTRAINED SHRINKING
#           TOLERANCE (§7.9): a flat 1.2*alpha0 through the first 30% (steady
#           bounded pressure — hotter lr is licensed by never dropping to the
#           0.4 floor the cyclic schemes used), then ONE smooth cubic-delayed
#           geometric contraction that monotonically tightens the enforced
#           violation band all the way onto the PROVEN terminal
#           5*alpha0*D/gamma_min at the final step. It passes ~1.6*alpha0 at
#           the hold end and ~6*alpha0 (the old plateau level) near 73%, so
#           the feasibility envelope brackets the 5/5-seed-feasible lineage —
#           the terminal restoration is preserved, just reached smoothly
#           instead of via a piecewise floor->plateau->spike.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition at the cool-down
#           start. beta1 finally runs the untried menu bet 4 both ways:
#           anti-correlated with lr (0.1 during the hot hold), then a
#           Sutskever-style INCREASING ramp to 0.4 through the polish —
#           momentum as an implicit ALM multiplier, letting the still-
#           moderate alpha enforce constraints while refining AEP — and
#           finally the proven gate down to 0.02 under the terminal alpha
#           climb so the diverging penalty never rides momentum.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_COOL = 0.55        # hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.45           # hold entry level, in units of D (hot, sustained)
_HI1 = 1.10           # hold exit level — the linear tail starts from here
_A_BASE = 1.2         # ADMM-style constant moderate penalty, in alpha0 units
_F_A0 = 0.30          # contraction of the violation band starts here
_POW = 3.0            # cubic back-loading of the contraction (proven exponent)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_HOT = 0.1         # low momentum while lr is hot (anti-correlated)
_B1_MID = 0.4         # increasing momentum ramp through the polish
_B1_MID_CENTER = 0.68
_B1_MID_WIDTH = 0.05
_B1_END = 0.02        # near-zero momentum under the terminal alpha climb
_B1_GATE_CENTER = 0.90
_B1_GATE_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted trapezoid hold -> linear tail onto gamma_min ---
    fh = jnp.clip(frac, 0.0, _F_COOL) / _F_COOL
    hold = (_HI0 + (_HI1 - _HI0) * fh) * Dj                   # sustained tilted hold
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (hold - gmin) * (1.0 - p)                 # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate base -> monotone geometric band contraction ---
    s = jnp.clip((frac - _F_A0) / (1.0 - _F_A0), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_BASE), 1.0))
    alpha = alpha0 * _A_BASE * jnp.exp(s * log_term)          # ends at 5*alpha0*D/gmin

    # --- betas: proven beta2 transition; beta1 hot-low -> polish ramp -> gate ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1m = 1.0 / (1.0 + jnp.exp(-(frac - _B1_MID_CENTER) / _B1_MID_WIDTH))
    b1_mid = _B1_HOT + (_B1_MID - _B1_HOT) * b1m              # increasing momentum ramp
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_GATE_CENTER) / _B1_GATE_WIDTH))
    beta1 = b1_mid + (_B1_END - b1_mid) * b1g                 # gated under the spike

    return lr, alpha, beta1, beta2