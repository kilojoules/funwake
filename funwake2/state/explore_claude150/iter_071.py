import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): abandon the
# oscillatory lr/alpha machinery entirely. This is a WSD HOLD + EPSILON-
# CONTRACTION schedule — the two menu rows not yet embodied anywhere in the
# lineage (lr row: "hold near c*D, then (near-)linear cool-down beats
# cosine/product decay", §6/§2; alpha row: "epsilon-constrained shrinking
# tolerance", §7.9).
#
# Mechanism argument: cosine restarts touch their hot peaks only briefly —
# most of the exploration phase is spent in transit between peak and trough.
# A tilted HOLD keeps lr >= 1.30*D for the entire first 58% of the run, so
# the integrated time-at-high-lr (what actually drives basin hopping) far
# exceeds any restart scheme tried, without needing the untested >1.65*D
# peaks. The violation debt this accumulates is repaid not by bursts or a
# plateau but by ONE smooth geometric alpha climb, framed as a contracting
# feasibility band: alpha ~ alpha0 * D/gamma(t), where the enforced band
# gamma(t) shrinks from ~D to gamma_min only at the final step. This is the
# native "alpha up as lr down" coupling — the mechanism every feasible
# ancestor relies on — but decoupled, delayed, and landing exactly on the
# 5/5-seed-proven terminal magnitude 5*alpha0*D/gamma_min.
#
#   lr    — 4% linear warmup -> tilted hold 1.55*D -> 1.30*D over [0, 58%]
#           (hot, but every value inside the proven-feasible range) -> the
#           proven straight linear tail landing exactly on gamma_min at the
#           last step. No cycles, no restarts.
#   alpha — exploration floor rising gently 0.5 -> 1.0 alpha0 across the
#           hold (bounds the debt a burst-free hot phase can accumulate),
#           then from 55% a power-delayed GEOMETRIC climb — the contracting
#           band — that passes ~4.5*alpha0 at 80%, ~130*alpha0 at 90%, and
#           ends at 5*alpha0*D/gamma_min. Stricter through the endgame than
#           the burst parent precisely because no mid-run restoration ever
#           ran; the whole debt is amortized continuously against the
#           falling lr instead of in spikes.
#   betas — the lineage-proven transitions only: beta2 0.2 -> 0.9 logistic
#           at the hold/decay boundary (adaptive scaling absorbs the rising
#           constraint curvature, menu bet 4); beta1 0.1 -> 0.02 gated late
#           so momentum cannot carry turbines back across the boundary
#           during the terminal contraction.
_F_WARM = 0.04       # linear lr warmup fraction
_F_HOLD = 0.58       # hold ends here; linear decay to gamma_min at 100%
_HI0 = 1.55          # hold start lr, in units of D
_HI1 = 1.30          # hold end lr, in units of D (tail starts from here)
_A_FLOOR0 = 0.5      # alpha floor at t=0, in alpha0 units
_A_FLOOR1 = 1.0      # alpha floor at hold end, in alpha0 units
_F_RAMP = 0.55       # geometric alpha climb (band contraction) starts here
_POW = 3.5           # back-loads the contraction toward the final steps
_TERM_GAIN = 5.0     # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58    # beta2 transition aligned with the decay start
_B2_WIDTH = 0.05
_B1_HI = 0.1         # native momentum through hold and early decay
_B1_LO = 0.02        # near-zero momentum during the terminal contraction
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted WSD hold -> linear tail to gamma_min ---
    # h freezes at 1 past _F_HOLD, so the tail decays from exactly _HI1 * D.
    h = jnp.clip(frac / _F_HOLD, 0.0, 1.0)
    lr_hold = (_HI0 + (_HI1 - _HI0) * h) * Dj
    p = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: rising floor -> delayed geometric band contraction ---
    # floor reaches _A_FLOOR1 exactly when the climb takes over, so the
    # contraction alpha(t) = alpha0 * (TERM/1)^s starts from a well-defined
    # base and ends at 5*alpha0*D/gamma_min at the last step.
    floor_units = _A_FLOOR0 + (_A_FLOOR1 - _A_FLOOR0) * h
    s = jnp.clip((frac - _F_RAMP) / (1.0 - _F_RAMP), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_FLOOR1), 1.0))
    alpha = alpha0 * floor_units * jnp.exp(s * log_term)

    # --- betas: proven transitions only ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = _B1_HI + (_B1_LO - _B1_HI) * b1r

    return lr, alpha, beta1, beta2