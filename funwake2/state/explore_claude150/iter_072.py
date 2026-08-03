import jax.numpy as jnp

# STRUCTURALLY NEW vs both the native parent and the burst/SGDR best (+0.0533%):
# the two prior-art menu bets no lineage member has tried TOGETHER or ALONE —
# (1) a ONE-CYCLE lr (§2): a single smooth hot arch instead of restarts, and
# (2) an EPSILON-CONSTRAINT alpha (§7.9): one continuous geometric contraction
# of the enforced violation band, replacing the floor/burst/logistic/plateau/
# spike patchwork with a single law that lands on the proven terminal scale.
#
#   lr    — one-cycle arch: half-cosine warmup 0.35*D -> 1.8*D over the first
#           30% (an apex both HIGHER than the 1.65*D ceiling tried and far
#           WIDER than any restart peak — exactly the "higher/longer peak"
#           the search state asks for), then one long half-cosine anneal that
#           lands exactly on gamma_min at the last step. No troughs: the
#           exploration budget is spent hot instead of oscillating.
#   alpha — DECOUPLED from lr everywhere. The enforced tolerance band
#           contracts geometrically from native-loose (0.5*alpha0) to strict:
#           alpha = 0.5*alpha0 * exp(frac^3 * log(5*D/(gamma_min*0.5))),
#           i.e. log-alpha grows cubically in time. During the hot arch alpha
#           stays ~0.6–1.5*alpha0 (native-like freedom); by 62% it passes the
#           proven ~6*alpha0 ALM plateau; the same law then continues into an
#           ever-steepening terminal restoration that finishes at the proven
#           5/5-seed-feasible 5*alpha0*D/gamma_min — strictly MORE repair in
#           the 62–90% window than the best schedule, so the sustained-hot
#           arch's violation debt is collected earlier and harder.
#   betas — beta1 anti-correlated with lr (the one-cycle signature, §2):
#           0.10 at low lr dipping to 0.04 at the apex so momentum never
#           compounds the biggest steps, then the proven logistic gate to
#           0.02 during the terminal spike; beta2 keeps the proven 0.2 -> 0.9
#           transition at the 62% cool-down so the adaptive scaling absorbs
#           the growing constraint curvature.
_F_PEAK = 0.30       # one-cycle apex location (fraction of run)
_LR_START = 0.35     # warmup start lr, in units of D
_LR_PEAK = 1.80      # apex lr, in units of D — hotter than anything tried
_A_LO = 0.5          # initial penalty, in alpha0 units (native-loose)
_A_GAIN = 5.0        # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_P_W = 3.0           # cubic back-loading of the tolerance contraction
_B2_LO = 0.2         # native fast-adapting second moment while exploring
_B2_HI = 0.9         # proven cool-down beta2
_B2_CENTER = 0.62
_B2_WIDTH = 0.05
_B1_HI = 0.10        # native momentum at low lr
_B1_PEAK = 0.04      # momentum dip at apex lr (anti-correlation)
_B1_LO = 0.02        # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.86
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total          # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle — half-cosine rise to the apex, half-cosine anneal ---
    # Both pieces equal _LR_PEAK*D at frac = _F_PEAK (continuous), and the
    # anneal lands exactly on gamma_min at the final step.
    r = jnp.clip(frac / _F_PEAK, 0.0, 1.0)
    rise = (_LR_START + (_LR_PEAK - _LR_START) * 0.5 * (1.0 - jnp.cos(jnp.pi * r))) * Dj
    p = jnp.clip((frac - _F_PEAK) / (1.0 - _F_PEAK), 0.0, 1.0)
    fall = gmin + (_LR_PEAK * Dj - gmin) * 0.5 * (1.0 + jnp.cos(jnp.pi * p))
    lr = jnp.where(frac < _F_PEAK, rise, fall)

    # --- alpha: epsilon-constraint contraction, decoupled from lr ---
    # log-alpha grows cubically from log(0.5*alpha0) to the proven terminal
    # log(5*alpha0*D/gamma_min): loose band while hot, ALM-plateau-scale by
    # the 62% cool-down, ever-steepening restoration through the tail.
    L = jnp.log(jnp.maximum(_A_GAIN * Dj / (gmin * _A_LO), 1.0))
    w = jnp.clip(frac, 0.0, 1.0) ** _P_W
    alpha = alpha0 * _A_LO * jnp.exp(w * L)

    # --- betas: lr-anti-correlated beta1 + proven terminal/cool-down gates ---
    lr_norm = jnp.clip((lr / Dj - _LR_START) / (_LR_PEAK - _LR_START), 0.0, 1.0)
    b1_exp = _B1_HI - (_B1_HI - _B1_PEAK) * lr_norm
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2