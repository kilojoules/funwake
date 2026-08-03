import jax.numpy as jnp

# STRUCTURALLY NEW vs the SGDR/anti-phased-burst best (+0.0533%): the two
# prior-art menu rows no lineage member has embodied yet, composed into one
# schedule.
#
#   lr    — WSD / ONE-CYCLE TRAPEZOID (§2/§6: "hold near c*D, then (near-)
#           linear cool-down beats cosine/product decay"). No restarts, no
#           troughs: a short linear warmup, then a SUSTAINED hot plateau at
#           1.4*D — hotter in time-integral than any cyclic parent, whose
#           cosine dips spend half the exploration budget below 1*D — held to
#           55%, then one long straight linear tail landing exactly on
#           gamma_min at the last step (the proven landing).
#   alpha — EPSILON-CONSTRAINED SHRINKING TOLERANCE (§7.9), fully decoupled
#           from lr. alpha is parameterized as alpha0 * D / eps(t), where
#           eps(t) is the violation band the penalty enforces. Through the
#           hot plateau eps = D/2 — i.e. an ADMM-style CONSTANT moderate
#           penalty of 2*alpha0 (5x the burst-best's exploration floor, so
#           the sustained heat never builds unbounded violation debt).
#           From 55% the band contracts GEOMETRICALLY, back-loaded by a
#           quartic in progress, reaching eps = gamma_min/5 exactly at the
#           end — which is precisely the proven 5/5-seed-feasible terminal
#           alpha = 5*alpha0*D/gamma_min. The terminal feasibility spike is
#           therefore not bolted on: it is the endpoint of the contracting
#           band, and by 90% the band is already ~D/50 so the tail's tiny
#           steps are pure in-band polish.
#   betas — proven transitions (beta2 0.2 -> 0.9 at cool-down start; beta1
#           gated to 0.02 in the terminal spike) plus menu bet 4 /
#           momentum-as-ALM (§4): a mid-cool-down beta1 hump to 0.3 lets
#           accumulated constraint gradients act as implicit multipliers
#           while alpha is still moderate, then the gate kills momentum
#           before the spike so nothing is carried back across the boundary.

_F_WARM = 0.04        # linear lr warmup over the first 4%
_F_DECAY = 0.55       # plateau ends; straight linear tail to gamma_min at 100%
_HI = 1.4             # plateau lr in units of D — sustained, not just a peak
_A_EXPL = 2.0         # plateau penalty (alpha0 units): eps band = D/2
_TERM_GAIN = 5.0      # terminal eps = gamma_min/5 -> alpha = 5*alpha0*D/gamma_min
_POW = 4.0            # quartic back-loading of the band contraction
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 transition aligned with cool-down start
_B2_WIDTH = 0.05
_B1_BASE = 0.1        # native momentum on the plateau
_B1_HUMP = 0.3        # momentum-as-ALM hump peak
_B1_HUMP_CENTER = 0.70
_B1_HUMP_WIDTH = 0.07
_B1_LO = 0.02         # near-zero momentum through the terminal spike
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> hot WSD plateau at 1.4*D -> straight linear tail ---
    warm = jnp.minimum(frac / _F_WARM, 1.0)
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - p)   # exact landing on gamma_min
    lr = lr_env * warm

    # --- alpha: constant moderate penalty, then geometric band contraction ---
    # alpha = alpha0 * D / eps(t); eps holds at D/_A_EXPL through the plateau,
    # then contracts geometrically (quartic-back-loaded) to gamma_min/_TERM_GAIN.
    s = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0) ** _POW
    log_ratio = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_EXPL), 1.0))
    alpha = alpha0 * _A_EXPL * jnp.exp(s * log_ratio)  # ends at 5*alpha0*D/gmin

    # --- betas: proven transitions + momentum-as-ALM hump in the cool-down ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    hump = jnp.exp(-(((frac - _B1_HUMP_CENTER) / _B1_HUMP_WIDTH) ** 2))
    b1_exp = _B1_BASE + (_B1_HUMP - _B1_BASE) * hump
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_exp + (_B1_LO - b1_exp) * b1r

    return lr, alpha, beta1, beta2