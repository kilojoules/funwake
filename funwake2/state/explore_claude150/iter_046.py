import jax.numpy as jnp

# STRUCTURAL PIVOT vs the anti-phased-burst best (+0.0533%): eight straight
# tweak attempts to the restart/burst family have failed, so this abandons the
# multi-cycle topology entirely and combines the three strongest prior-art
# menu items NOT yet embodied anywhere in the lineage:
#
#   lr    — ONE-CYCLE SUPERCONVERGENCE (§2): a single linear warmup to a
#           record-hot 1.8*D peak (above the parent's 1.65*D — hotter, as the
#           guidance asks), then ONE skewed cosine anneal that lands exactly
#           on gamma_min at the last step. Instead of paying for three
#           re-coolings, the run holds >1.35*D for roughly its first third
#           (the skew stretches the hot shoulder), then sweeps CONTINUOUSLY
#           through every displacement scale from basin-hopping to polish —
#           consolidation is built into the anneal rather than into troughs.
#   alpha — ε-CONSTRAINED CONTRACTING TOLERANCE BAND (§7.9, untried): the
#           penalty is a single smooth geometric law alpha =
#           (K/C0)*alpha0 * (C0*D/gamma_min)^w(t), i.e. the enforced
#           violation band starts at ~C0*D (floor 0.5*alpha0, matching the
#           proven exploration floor) and contracts to gamma_min only at the
#           very end. The contraction exponent w is delayed to 40% and
#           back-loaded (power 2.5), which makes alpha pass ~6*alpha0 near
#           78% — the parent's proven plateau strictness at the same point —
#           and then climb monotonically to land on the exact
#           5*alpha0*D/gamma_min terminal value that was 5/5-seed feasible.
#           Terminal restoration is therefore preserved by construction:
#           monotone alpha growth into a vanishing lr, no debt left unpaid.
#   beta1 — ANTI-CORRELATED WITH lr (one-cycle momentum, §2, untried):
#           near-zero momentum (0.02) at the record-hot peak so no overshoot
#           compounds — this is what licenses 1.8*D — rising to 0.12 as lr
#           anneals so accumulated momentum acts as an implicit ALM
#           multiplier on the constraint gradient (menu bet 4), then gated
#           back down to 0.02 inside the terminal alpha spike (proven).
#   beta2 — proven 0.2 -> 0.9 logistic ramp, centered at the point where the
#           anneal crosses into consolidation-scale steps.
_F_WARM = 0.08        # linear lr warmup; longer than parent's 3% to tame the hotter peak
_PEAK = 1.8           # one-cycle peak in units of D — beyond the tried 1.65*D ceiling
_SKEW = 1.2           # >1 stretches the hot shoulder of the cosine anneal
_A_K = 5.0            # terminal alpha = _A_K*alpha0*D/gamma_min (proven feasible scale)
_C0 = 10.0            # initial tolerance band = _C0*D -> exploration floor 0.5*alpha0
_F_AON = 0.40         # band contraction begins here (delayed ramp, §7.3/7.5)
_APOW = 2.5           # back-loads contraction; hits ~6*alpha0 (proven plateau) near 78%
_B1_MIN = 0.02        # momentum at the hot peak and inside the terminal spike
_B1_MAX = 0.12        # momentum as lr anneals (implicit ALM multiplier)
_B1_CENTER = 0.85     # terminal beta1 gate, aligned with the steep alpha climb
_B1_WIDTH = 0.04
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58
_B2_WIDTH = 0.06


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle — warmup to 1.8*D, single skewed cosine anneal ---
    # At u = 1 the cosine hits -1 exactly, so lr = gamma_min at the last step.
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    u = jnp.clip((frac - _F_WARM) / (1.0 - _F_WARM), 0.0, 1.0)
    anneal = 0.5 * (1.0 + jnp.cos(jnp.pi * u ** _SKEW))
    lr = warm * (gmin + (_PEAK * Dj - gmin) * anneal)

    # --- alpha: contracting tolerance band, gamma_eff: C0*D -> gamma_min ---
    # alpha = _A_K*alpha0*D/gamma_eff with gamma_eff shrinking geometrically;
    # w = 0 gives the 0.5*alpha0 exploration floor, w = 1 lands exactly on
    # the proven 5*alpha0*D/gamma_min terminal restoration value.
    w = jnp.clip((frac - _F_AON) / (1.0 - _F_AON), 0.0, 1.0) ** _APOW
    log_r = jnp.log(jnp.maximum(_C0 * Dj / gmin, 1.0))
    alpha = (_A_K / _C0) * alpha0 * jnp.exp(w * log_r)

    # --- betas: beta1 anti-correlated with lr + terminal gate; beta2 ramp ---
    lr_norm = jnp.clip(lr / (_PEAK * Dj), 0.0, 1.0)
    b1_cycle = _B1_MAX - (_B1_MAX - _B1_MIN) * lr_norm    # low when hot, high when cool
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_cycle + (_B1_MIN - b1_cycle) * gate        # near-zero in the spike

    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2