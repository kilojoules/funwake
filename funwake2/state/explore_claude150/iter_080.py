import jax.numpy as jnp

# STRUCTURALLY NEW vs the anti-phased-burst best (+0.0533%): the lineage is
# saturated on the RESTART family (SGDR peaks, cyclic/burst alpha, plateau +
# cubic spike) — eight straight variations scored at or below it. This attempt
# abandons restarts entirely and takes the three prior-art menu bets the
# lineage has never combined:
#
#   lr    — ONE-CYCLE (§2): a single super-cycle. Linear warmup over the first
#           8% to a 1.8*D peak (hotter than the 1.65*D best, and — unlike a
#           restart peak — annealed smoothly rather than revisited), then ONE
#           long cosine anneal down to 1.0*D at 62%, then the proven straight
#           linear tail landing exactly on gamma_min at the last step. Time
#           spent at high lr far exceeds the restart schedules (no trough
#           visits), which is where the extra AEP must come from.
#   alpha — ADMM-STYLE CONSTANT MODERATE PENALTY (§7.9, explicitly untried):
#           dead flat at 1.5*alpha0 for the whole one-cycle — roughly the
#           TIME-AVERAGE of the best schedule's bursty alpha, but delivered
#           uniformly, so violation debt is leaned on continuously instead of
#           episodically while the hot cycle explores. From 62% the endgame is
#           an EPSILON-CONSTRAINED SHRINKING TOLERANCE (§7.9): alpha =
#           5*alpha0*D/gamma_t with the enforced band gamma_t contracting
#           geometrically (back-loaded, power 2.5) from ~(10/3)*D down to
#           gamma_min exactly at the final step — one continuous contraction
#           replacing the plateau+spike, ending at the SAME proven 5/5-seed
#           terminal value 5*alpha0*D/gamma_min, so strict feasibility
#           restoration is preserved.
#   beta1 — ONE-CYCLE ANTI-CORRELATION as an IMPLICIT ALM MULTIPLIER (§2/§4):
#           momentum 0.05 at peak lr (steps too hot to trust accumulated
#           direction) rising to 0.25 as the cycle cools — the accumulating
#           constraint gradients act like an augmented-Lagrangian multiplier,
#           letting the moderate constant alpha enforce more than its size
#           suggests — then the proven logistic gate drops it to 0.02 for the
#           terminal contraction so momentum never re-ejects turbines.
#   beta2 — the proven 0.2 -> 0.9 logistic transition at the cool-down start.
_F_WARM = 0.08        # linear warmup to the one-cycle peak
_F_COOL = 0.62        # anneal ends; proven linear tail to gamma_min begins
_PEAK = 1.8           # one-cycle peak lr, in D units — hotter than any tried
_END = 1.0            # lr at anneal end, where the proven tail starts
_A_CONST = 1.5        # ADMM constant penalty during the cycle, in alpha0 units
_P_RAMP = 2.5         # back-loading of the geometric tolerance contraction
_TERM_GAIN = 5.0      # proven terminal alpha = 5*alpha0*D/gamma_min
_B2_LO = 0.2
_B2_HI = 0.9
_B2_CENTER = 0.62     # beta2 transition aligned with the cool-down start
_B2_WIDTH = 0.05
_B1_PEAK = 0.05       # momentum at peak lr (one-cycle anti-correlation)
_B1_MAX = 0.25        # momentum once the cycle has cooled (implicit ALM)
_B1_LO = 0.02         # proven near-zero momentum in the terminal contraction
_B1_CENTER = 0.88
_B1_WIDTH = 0.03


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dj = jnp.asarray(D) * 1.0
    gmin = jnp.maximum(jnp.asarray(gamma_min) * 1.0, 1e-30)

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: warmup -> single long cosine anneal 1.8D -> 1.0D -> linear tail ---
    a = jnp.clip((frac - _F_WARM) / (_F_COOL - _F_WARM), 0.0, 1.0)
    lr_cyc = (_END + (_PEAK - _END) * 0.5 * (1.0 + jnp.cos(jnp.pi * a))) * Dj
    p = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0)
    lr_env = gmin + (lr_cyc - gmin) * (1.0 - p)               # exact landing on gamma_min
    warm = jnp.minimum(frac / _F_WARM, 1.0)                   # damps the hot start; lr only
    lr = lr_env * warm

    # --- alpha: flat ADMM penalty -> geometric epsilon-contraction endgame ---
    # alpha = A_CONST*alpha0 * (TERM_GAIN*D / (gmin*A_CONST))^s : the enforced
    # violation band contracts geometrically to gamma_min only at the end,
    # arriving at the proven terminal 5*alpha0*D/gamma_min.
    s = jnp.clip((frac - _F_COOL) / (1.0 - _F_COOL), 0.0, 1.0) ** _P_RAMP
    log_term = jnp.log(jnp.maximum(_TERM_GAIN * Dj / (gmin * _A_CONST), 1.0))
    alpha = alpha0 * _A_CONST * jnp.exp(s * log_term)

    # --- betas: one-cycle beta1 anti-correlation + proven transitions ---
    lr_norm = (lr_cyc / Dj - _END) / (_PEAK - _END)           # 1 at peak, 0 when cooled
    b1_cycle = _B1_MAX - (_B1_MAX - _B1_PEAK) * lr_norm       # implicit ALM ramp
    b1r = 1.0 / (1.0 + jnp.exp(-(frac - _B1_CENTER) / _B1_WIDTH))
    beta1 = b1_cycle + (_B1_LO - b1_cycle) * b1r
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _B2_LO + (_B2_HI - _B2_LO) * b2r

    return lr, alpha, beta1, beta2