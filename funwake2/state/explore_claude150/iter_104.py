import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): no restarts, no
# bursts, no plateau-then-spike. Three untried menu directions combined into
# one coherent homotopy:
#
#   lr    — ONE-CYCLE / WSD (prior-art §2/§6, the top untried lr row): 3%
#           linear warmup, then a SUSTAINED hot hold — tilted 1.5*D -> 1.15*D
#           over the first 45% — instead of momentary cosine peaks. This is
#           the "higher/LONGER peak early" the guidance asks for: far more
#           total hot exploration than any restart schedule in the lineage.
#           Then the WSD bet: a single long LINEAR decay (55% of the run)
#           landing exactly on gamma_min at the last step — the hypothesis
#           that near-linear cool-down beats cosine/product fine-tuning.
#   alpha — ADMM-STYLE CONSTANT moderate penalty (0.5*alpha0) for the whole
#           hold (untried direction: penalty fully decoupled and *flat*, no
#           coupling, no ramp), then the ε-CONSTRAINED SHRINKING BAND (§7.9,
#           the one menu row nothing in the lineage embodies): alpha grows
#           GEOMETRICALLY through the decay phase, equivalent to contracting
#           the enforced violation tolerance from ~10*D down to gamma_min
#           exactly at the end. Back-loaded (q^2.5) so mid-decay alpha stays
#           gentle (~1-3*alpha0, AEP keeps improving inside a loose band)
#           while the final ~15% compounds steeply and lands on the proven
#           5/5-seed-feasible terminal anchor 5*alpha0*D/gamma_min. The band
#           contracts continuously, so feasibility debt is amortized across
#           the whole cool-down instead of repaid in one terminal spike —
#           and the 80-95% region is STRICTER than the parent's, protecting
#           feasibility despite the hotter start.
#   betas — beta2 0.2 -> 0.9 at the hold/decay boundary (proven transition,
#           re-centered), plus the untried ONE-CYCLE momentum bet (§2/§4):
#           beta1 ANTI-CORRELATED with lr — 0.1 while hot, rising to 0.4 as
#           lr decays (momentum as implicit ALM multiplier, smoothing the
#           growing constraint gradient), then the proven terminal gate to
#           0.02 so no momentum carries turbines across the boundary at the
#           end.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_HOLD = 0.45        # hot hold ends here; single linear decay to gamma_min after
_HI0 = 1.50           # hold start, in units of D — sustained, not a momentary peak
_HI1 = 1.15           # hold end / decay start, in units of D
_A_EXP = 0.5          # ADMM-style constant penalty during the hold, in alpha0 units
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven feasible anchor)
_POW = 2.5            # back-loads the geometric band contraction into the late decay
_B2_LO = 0.2          # native fast adaptation while exploring
_B2_HI = 0.9          # smoother adaptive scaling through the long decay
_B2_CENTER = 0.45     # beta2 transition at the hold/decay boundary
_B2_WIDTH = 0.05
_B1_HOT = 0.1         # low momentum at high lr (one-cycle)
_B1_MID = 0.4         # momentum rises as lr falls — implicit ALM averaging
_B1_UP_CENTER = 0.60
_B1_UP_WIDTH = 0.06
_B1_END = 0.02        # near-zero momentum in the terminal contraction (proven)
_B1_DN_CENTER = 0.88
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> tilted hot hold -> single long linear decay ---
    # h freezes at 1 past _F_HOLD, so the decay starts from _HI1 * D and the
    # linear ramp lands exactly on gamma_min at the last step.
    h = jnp.minimum(frac / _F_HOLD, 1.0)
    hi = (_HI0 + (_HI1 - _HI0) * h) * Dj                      # sustained hot envelope
    q = jnp.clip((frac - _F_HOLD) / (1.0 - _F_HOLD), 0.0, 1.0)
    lr_env = gmin + (hi - gmin) * (1.0 - q)                   # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: constant moderate penalty -> geometric tolerance contraction ---
    # u = 0 through the hold (flat 0.5*alpha0), then the enforced violation
    # band shrinks geometrically to gamma_min, ending at 5*alpha0*D/gamma_min.
    u = q ** _POW
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_EXP), 1.0))
    alpha = _A_EXP * alpha0 * jnp.exp(u * log_ratio)

    # --- betas: beta2 up at the phase boundary; beta1 anti-correlated with lr ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    b1_base = _B1_HOT + (_B1_MID - _B1_HOT) * up              # rises as lr decays
    beta1 = b1_base + (_B1_END - b1_base) * dn                # terminal gate (proven)

    return lr, alpha, beta1, beta2