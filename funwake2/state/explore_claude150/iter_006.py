import jax.numpy as jnp

# STRUCTURALLY NEW vs the one-cycle parent: a THREE-CYCLE SGDR lr schedule with
# MID-RUN FEASIBILITY-RESTORATION BURSTS — two untried directions from the menu,
# combined:
#   lr    — SGDR warm restarts: three half-cosine cycles with growing period and
#           decaying peaks (1.15D -> 0.75D -> 0.45D). Cycles 1-2 cool to a small
#           metre-valued trough (0.04*D); the LAST cycle's cool-down lands
#           exactly on gamma_min. First-cycle peak is HOTTER than the parent's
#           (1.15*D vs 1.0*D) to push AEP, and each restart re-injects
#           exploration energy the monotone parent never recovers.
#   alpha — fully DECOUPLED from 1/lr during the run: an ADMM-style moderate
#           base (logistic drift 1*alpha0 -> 4*alpha0 so the final cycle runs
#           stiffer), plus Gaussian RESTORATION BURSTS at the two lr troughs
#           (repair violations cheaply while steps are tiny, then relax for the
#           next hot restart), plus the parent-PROVEN gated TERMINAL divergence
#           ~5*alpha0*D/lr_env over the final ~10%, guaranteeing strict
#           feasibility at finish (5/5-seed-validated endgame magnitude).
#   betas — beta1 held at native 0.1; beta2 phase-transitions 0.2 -> 0.9 as the
#           last cycle enters its polish/feasibility phase.
_BOUND1 = 0.25         # end of cycle 1 (restart)
_BOUND2 = 0.55         # end of cycle 2 (restart); cycle 3 runs 0.55 -> 1.0
_PEAK1 = 1.15          # cycle peaks, in units of D (decaying restarts)
_PEAK2 = 0.75
_PEAK3 = 0.45
_TROUGH = 0.04         # cycles 1-2 cool to 0.04*D; cycle 3 cools to gamma_min
_F_WARM = 0.02         # linear lr warmup over first 2% (protects grid init)
_ABASE_LO = 1.0        # ADMM-style base penalty, units of alpha0
_ABASE_HI = 4.0        # stiffer base for the final cycle
_ABASE_CENTER = 0.60
_ABASE_WIDTH = 0.08
_BURST_GAIN = 8.0      # restoration-burst amplitude, units of alpha0
_BURST_WIDTH = 0.015   # ~120 steps of 8000 per burst
_GATE_CENTER = 0.90    # terminal restoration engages over the final ~10%
_GATE_WIDTH = 0.02
_TERM_GAIN = 5.0       # terminal alpha ~ 5*alpha0*D/lr_env (parent-proven)
_BETA2_LO = 0.2
_BETA2_HI = 0.9
_B2_CENTER = 0.75      # beta2 ramps up as cycle 3 passes its peak
_B2_WIDTH = 0.05


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = jnp.asarray(total_steps) * 1.0
    Dv = jnp.asarray(D) * 1.0
    gmin = jnp.asarray(gamma_min) * 1.0

    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- lr: SGDR warm restarts, three half-cosine cycles ---
    p1 = jnp.clip(frac / _BOUND1, 0.0, 1.0)
    p2 = jnp.clip((frac - _BOUND1) / (_BOUND2 - _BOUND1), 0.0, 1.0)
    p3 = jnp.clip((frac - _BOUND2) / (1.0 - _BOUND2), 0.0, 1.0)

    floor12 = _TROUGH * Dv
    lr_c1 = floor12 + 0.5 * (_PEAK1 * Dv - floor12) * (1.0 + jnp.cos(jnp.pi * p1))
    lr_c2 = floor12 + 0.5 * (_PEAK2 * Dv - floor12) * (1.0 + jnp.cos(jnp.pi * p2))
    lr_c3 = gmin + 0.5 * (_PEAK3 * Dv - gmin) * (1.0 + jnp.cos(jnp.pi * p3))
    lr_env = jnp.where(frac < _BOUND1, lr_c1,
                       jnp.where(frac < _BOUND2, lr_c2, lr_c3))

    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: moderate drifting base (decoupled from lr) ---
    base_r = 1.0 / (1.0 + jnp.exp(-(frac - _ABASE_CENTER) / _ABASE_WIDTH))
    alpha_base = alpha0 * (_ABASE_LO + (_ABASE_HI - _ABASE_LO) * base_r)

    # --- mid-run feasibility-restoration bursts at the two lr troughs:
    # violations accumulated during each hot cycle are repaired while steps are
    # tiny, then the penalty relaxes so the next restart explores freely ---
    burst1 = jnp.exp(-jnp.square((frac - _BOUND1) / _BURST_WIDTH))
    burst2 = jnp.exp(-jnp.square((frac - _BOUND2) / _BURST_WIDTH))
    alpha_burst = _BURST_GAIN * alpha0 * (burst1 + burst2)

    # --- terminal feasibility restoration: gated native-style divergence,
    # -> _TERM_GAIN*alpha0*D/gamma_min at finish (feasible-parent magnitude) ---
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _GATE_CENTER) / _GATE_WIDTH))
    alpha_term = _TERM_GAIN * alpha0 * Dv / jnp.maximum(lr_env, 1e-30)
    alpha = alpha_base + alpha_burst + gate * alpha_term

    # --- betas: native low momentum; beta2 up for the polish phase ---
    beta1 = 0.1
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _BETA2_LO + (_BETA2_HI - _BETA2_LO) * b2r

    return lr, alpha, beta1, beta2