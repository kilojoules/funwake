import jax.numpy as jnp

# STRUCTURALLY NEW vs the burst/SGDR best (+0.0533%): the lineage has only
# ever explored with OSCILLATING lr (cosine restarts) and repaid violation
# debt in discrete events (bursts, plateau, cubic spike). This schedule is
# the opposite regime, straight off the untried rows of the prior-art menu:
#
#   lr    — WSD / one-cycle "hold-then-linear" (§6, §2): 3% warmup, then a
#           SINGLE FLAT HOLD at 1.25*D for the whole exploration phase
#           (hotter time-integral than any SGDR parent, whose cycle-average
#           was ~1.0*D — the "longer/hotter peak" the guidance asks for,
#           taken to its limit), then the proven straight linear tail from
#           58% landing exactly on gamma_min at the last step. No restarts:
#           every exploration step moves at full temperature.
#   alpha — ε-CONSTRAINED CONTRACTING TOLERANCE (§7.9), decoupled from lr:
#           one smooth GEOMETRIC law instead of floor+bursts+logistic+
#           plateau+cubic-spike. Alpha sits at a 0.5*alpha0 exploration
#           floor (proven-feasible territory: the pre-burst SGDR best was
#           5/5-feasible from a similar floor), then from 50% climbs
#           geometrically — the implied enforced violation band
#           gamma(t) ~ D/alpha(t) contracts smoothly to gamma_min only at
#           the end — with quadratic back-loading so mid-run alpha stays
#           moderate, landing exactly on the 5/5-seed-proven terminal
#           5*alpha0*D/gamma_min. The terminal feasibility restoration is
#           therefore preserved in full strength, just approached along a
#           log-linear path instead of plateau+spike.
#   betas — beta2 keeps the proven 0.2 -> 0.9 transition at the cool-down.
#           beta1 does the untried ONE-CYCLE ANTI-CORRELATION (§2/§4):
#           0.1 while lr is hot, RISING to 0.35 as lr shrinks (momentum as
#           implicit ALM multiplier — lets the still-moderate alpha enforce
#           constraints while polishing AEP), then gated down to the proven
#           0.02 for the terminal contraction so momentum never carries
#           turbines across the boundary at the end. The lineage has only
#           ever lowered beta1; this rise-then-fall is new.
_F_WARM = 0.03        # linear lr warmup over the first 3% (proven)
_F_DECAY = 0.58       # hold ends; linear decay to gamma_min at 100% (proven tail)
_HI = 1.25            # flat hold lr, in units of D (hotter integral than SGDR mean)
_A_LO = 0.5           # exploration penalty floor, in alpha0 units
_F_ACLIMB = 0.50      # delayed ramp (menu bet 2): geometric climb starts mid-run
_A_POW = 2.0          # quadratic back-loading of the contraction
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven 5/5 scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.58     # beta2 transition aligned with the decay start
_B2_WIDTH = 0.05
_B1_LO = 0.1          # low momentum while lr is hot (native exploration value)
_B1_MID = 0.35        # one-cycle momentum peak while lr shrinks
_B1_UP_CENTER = 0.62  # momentum rises once the cool-down is underway
_B1_UP_WIDTH = 0.04
_B1_END = 0.02        # near-zero momentum during the terminal contraction (proven)
_B1_DN_CENTER = 0.88
_B1_DN_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> flat hold at _HI*D -> linear tail to gamma_min ---
    p = jnp.clip((frac - _F_DECAY) / (1.0 - _F_DECAY), 0.0, 1.0)
    lr_env = gmin + (_HI * Dj - gmin) * (1.0 - p)             # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: floor -> single geometric contracting-tolerance climb ---
    # alpha = _A_LO*alpha0 * exp(s^POW * log(ratio)); at s=1 the implied
    # tolerance band has contracted to gamma_min and alpha equals the proven
    # terminal 5*alpha0*D/gamma_min.
    ratio = _TERM_GAIN * Dj / (gmin * _A_LO)                  # terminal / floor, alpha0-free
    log_climb = jnp.log(jnp.maximum(ratio, 1.0))
    s = jnp.clip((frac - _F_ACLIMB) / (1.0 - _F_ACLIMB), 0.0, 1.0) ** _A_POW
    alpha = _A_LO * alpha0 * jnp.exp(s * log_climb)

    # --- betas: proven beta2 transition + one-cycle rise-then-fall beta1 ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r
    b1up = 1.0 / (1.0 + jnp.exp(-(frac - _B1_UP_CENTER) / _B1_UP_WIDTH))
    b1_mid = _B1_LO + (_B1_MID - _B1_LO) * b1up               # momentum up as lr cools
    b1dn = 1.0 / (1.0 + jnp.exp(-(frac - _B1_DN_CENTER) / _B1_DN_WIDTH))
    beta1 = b1_mid + (_B1_END - b1_mid) * b1dn                # gated off for the endgame

    return lr, alpha, beta1, beta2