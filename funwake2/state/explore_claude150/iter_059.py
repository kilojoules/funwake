import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): every prior
# attempt in the lineage is built on CYCLES (SGDR restarts, cyclic alpha,
# restoration bursts). This one abandons cycles entirely and composes the
# four remaining UNTRIED menu bets into a single coherent design:
#
#   lr    — ONE-CYCLE SUPER-CONVERGENCE (prior-art §2, untried): a long
#           half-cosine warmup from 0.15*D to a 1.70*D peak at 30% (hotter
#           than the 1.65*D burst best, and — the structural point —
#           SUSTAINED: the mean lr over the run is higher than any cyclic
#           parent because there are no 0.65*D troughs), then one long
#           half-cosine annihilation that lands EXACTLY on gamma_min at the
#           last step. Exploration is spent as one continuous hot traverse
#           instead of brief peaks, so basin structure found early is never
#           destroyed by a restart kick.
#   alpha — EPSILON-CONSTRAINED SHRINKING TOLERANCE (§7.9, untried): fully
#           decoupled from lr. A 0.5*alpha0 floor through warmup and peak
#           frees the hot traverse to trade violation for AEP; from 42% a
#           single delayed geometric contraction (power-2.5 back-loading)
#           continuously shrinks the enforced violation band — passing the
#           proven ~6*alpha0 plateau level near 75% and diverging smoothly
#           onto the PROVEN terminal 5*alpha0*D/gamma_min at the final step.
#           No plateau kink, no bursts: one monotone contraction whose band
#           reaches gamma_min only at the end, i.e. a built-in terminal
#           feasibility spike, so the 5/5-seed feasible endgame is preserved.
#   beta1 — ANTI-CORRELATED WITH lr (one-cycle momentum, §2/§4, untried):
#           beta1 = 0.35 when lr is small, 0.05 at the 1.70*D peak. Early:
#           momentum accelerates the cold warmup. At peak: near-zero momentum
#           stabilizes the hottest steps ever run in this lineage. Late
#           anneal: momentum rises back toward 0.35, acting as an implicit
#           ALM multiplier that accumulates constraint gradients so the
#           contracting alpha enforces feasibility at moderate magnitudes —
#           then the PROVEN terminal gate drops beta1 to 0.02 at 88% so the
#           diverging alpha never rides momentum.
#   beta2 — proven phase transition 0.2 -> 0.9, centered mid-anneal (55%)
#           where the alpha contraction begins, absorbing the growing
#           constraint curvature (~alpha) with adaptive scaling.
_F_PEAK = 0.30        # one-cycle lr peak position (end of warmup traverse)
_LR_START = 0.15      # warmup start lr, in units of D
_LR_PEAK = 1.70       # sustained one-cycle peak — hotter than any cyclic peak
_A_LO = 0.5           # exploration penalty floor, in alpha0 units
_F_A = 0.42           # epsilon-contraction of the violation band starts here
_A_POW = 2.5          # back-loads the contraction (smooth, no plateau kink)
_TERM_GAIN = 5.0      # terminal alpha = 5*alpha0*D/gamma_min (proven scale)
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.55     # beta2 ramp centered where the alpha contraction bites
_B2_WIDTH = 0.06
_B1_MAX = 0.35        # momentum when lr is low (warmup start, late anneal)
_B1_MIN = 0.05        # momentum at the 1.70*D peak (stability when hottest)
_B1_LO = 0.02         # near-zero momentum during the terminal alpha spike
_B1_CENTER = 0.88     # proven terminal momentum gate
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: one-cycle — half-cosine warmup, half-cosine annihilation ---
    # lr_up == peak for frac >= _F_PEAK and lr_dn == peak for frac <= _F_PEAK,
    # so jnp.minimum splices the two half-cycles exactly at the peak with no
    # branch. The down half-cycle lands lr exactly on gamma_min at frac = 1.
    s_up = jnp.clip(frac / _F_PEAK, 0.0, 1.0)
    lr_up = (_LR_START + (_LR_PEAK - _LR_START) * 0.5 * (1.0 - jnp.cos(jnp.pi * s_up))) * Dj
    s_dn = jnp.clip((frac - _F_PEAK) / (1.0 - _F_PEAK), 0.0, 1.0)
    lr_dn = gmin + (_LR_PEAK * Dj - gmin) * 0.5 * (1.0 + jnp.cos(jnp.pi * s_dn))
    lr = jnp.minimum(lr_up, lr_dn)

    # --- alpha: floor -> single epsilon-constrained geometric contraction ---
    # One monotone climb from the exploration floor to the proven terminal
    # 5*alpha0*D/gamma_min; the enforced violation band shrinks continuously
    # and reaches the gamma_min tolerance only at the final step.
    r = jnp.clip((frac - _F_A) / (1.0 - _F_A), 0.0, 1.0) ** _A_POW
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_LO), 1.0))
    alpha = alpha0 * _A_LO * jnp.exp(r * log_term)

    # --- beta2: proven low -> high transition, mid-anneal ---
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    # --- beta1: anti-correlated with lr, gated down for the terminal spike ---
    lr_norm = jnp.clip(lr / (_LR_PEAK * Dj), 0.0, 1.0)
    b1_oc = _B1_MAX + (_B1_MIN - _B1_MAX) * lr_norm
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_oc + (_B1_LO - b1_oc) * gate

    return lr, alpha, beta1, beta2