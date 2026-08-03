import jax.numpy as jnp

# STRUCTURALLY NEW vs the flat-top-notch best (+0.0577%), the SGDR line, and
# the one-cycle parent: every schedule tried so far is a SMOOTH waveform
# (cosine, product, linear ramps, sin^2Q notches). This is the one classic lr
# family never attempted — a GEOMETRIC STEP-DECAY STAIRCASE (ImageNet-style
# step decay = simulated annealing's stepped temperature plateaus), fused with
# the §7.9 ε-CONSTRAINED contracting-tolerance law in stair form.
#
# Mechanism: each stair holds lr CONSTANT so the layout fully equilibrates at
# one spatial resolution before the next drop — no steps wasted in transit
# between hot and cold regimes (the duty-cycle argument that made the flat-top
# best win, taken to its limit: 100% duty at every scale). Each discontinuous
# stair drop IS the consolidation event: lr falls x0.62 exactly as alpha jumps
# x2.5, so the enforced violation band ~lr/alpha contracts by x0.25 per stair
# — a discrete ε-constrained tolerance sequence ending at gamma_min.
#
#   lr    — 2% warmup, then FOUR CONSTANT STAIRS with dwell FRONT-LOADED HOT
#           (26/20/14/10% of the run): 1.9*D sustained (the parent's proven
#           momentary peak, now held for 2000+ steps — the "longer peak"
#           bet), then 1.18*D, 0.73*D, 0.45*D. Cumulative heat ~matches the
#           best but is redistributed to the hottest scale. From 70% the
#           proven straight linear tail lands EXACTLY on gamma_min.
#   alpha — anti-staircase: 0.3*alpha0 on the hot stair (freest proven
#           start), x2.5 per stair to a bounded ~4.7*alpha0 plateau (ALM
#           §7.2/7.3 — never diverging with 1/lr), then the 5/5-seed-proven
#           cubic-delayed geometric climb from 78% to the terminal
#           5*alpha0*D/gamma_min feasibility spike. Feasibility endgame
#           byte-identical in structure to the proven one.
#   beta1 — anti-correlated with the lr envelope (one-cycle momentum, §2/§4):
#           0.05 on the 1.9*D stair (huge steps must not compound), rising
#           stair-by-stair to ~0.35 as lr cools (momentum as implicit ALM
#           multiplier), gated to the proven 0.02 during the terminal spike.
#   beta2 — proven 0.2 -> 0.9 logistic, centered at the tail start (70%),
#           where low-beta2 fast adaptation (needed to absorb stair shocks)
#           gives way to smooth polishing.
_F_WARM = 0.02                     # linear lr warmup over the first 2%
_BOUNDS = (0.26, 0.46, 0.60, 0.70) # stair-drop fractions; tail starts at 0.70
_F_EXPL = 0.70                     # staircase ends; linear tail to gamma_min
_L0 = 1.9                          # hottest stair, in units of D (sustained)
_R = 0.62                          # lr ratio per stair: 1.9 -> 1.18 -> .73 -> .45
_A0U = 0.3                         # alpha on the hot stair, in alpha0 units
_G = 2.5                           # alpha growth per stair (band contracts x0.25)
_A_LAST = _A0U * _G ** 3           # bounded plateau ~4.69*alpha0 before spike
_F_TERM = 0.78                     # terminal geometric alpha climb starts (proven)
_POW = 3.0                         # cubic back-loading of the climb (proven)
_TERM_GAIN = 5.0                   # terminal alpha = 5*alpha0*D/gamma_min (proven)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.72                  # beta2 pivot as the staircase hands off to tail
_B2_WIDTH = 0.05
_B1_MIN = 0.05                     # momentum on the hottest stair
_B1_MAX = 0.35                     # momentum as lr -> 0 (implicit multiplier)
_B1_END = 0.02                     # near-zero momentum during the terminal spike
_B1_CENTER = 0.86                  # gate closes before the spike gets steep
_B1_WIDTH = 0.04


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- stair index: 0..3, via boundary crossings (traceable, no branches) ---
    bounds = jnp.asarray(_BOUNDS)
    level = jnp.clip(jnp.sum((frac >= bounds) * 1.0), 0.0, 3.0)

    # --- lr: warmup -> four constant geometric stairs -> linear tail ---
    lr_hold = _L0 * Dj * jnp.power(_R, level)                 # constant within a stair
    p = jnp.clip((frac - _F_EXPL) / (1.0 - _F_EXPL), 0.0, 1.0)
    lr_env = gmin + (lr_hold - gmin) * (1.0 - p)              # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: anti-staircase to a bounded plateau -> proven terminal spike ---
    alpha_units = _A0U * jnp.power(_G, level)                 # 0.3 -> 4.69 alpha0, in stairs
    s = jnp.clip((frac - _F_TERM) / (1.0 - _F_TERM), 0.0, 1.0) ** _POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LAST), 1.0))
    alpha = alpha0 * alpha_units * jnp.exp(s * log_term)      # ends at 5*alpha0*D/gmin

    # --- beta1: anti-correlated with the lr envelope, gated for the spike ---
    b1_anti = _B1_MAX - (_B1_MAX - _B1_MIN) * (lr_env / (_L0 * Dj))
    b1g = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_anti * (1.0 - b1g) + _B1_END * b1g

    # --- beta2: proven low -> high logistic at the staircase/tail handoff ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2