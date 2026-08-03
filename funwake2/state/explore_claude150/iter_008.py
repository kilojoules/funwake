import jax.numpy as jnp

# STRUCTURALLY NEW vs the one-cycle parent: SGDR-style MULTI-CYCLE cosine lr
# with decaying warm-restart peaks, paired with ANTI-PHASED cyclic alpha
# "feasibility-restoration bursts" (alpha spikes as each cycle's lr bottoms
# out, then releases at the next restart — graduated non-convexity). The run
# ends with a final gentle warm restart that half-cosines exactly onto
# gamma_min, under the parent's PROVEN gated terminal alpha divergence
# (~5*alpha0*D/lr -> 5*alpha0*D/gamma_min), so strict feasibility is kept.
#   lr    — 3 cosine cycles over the first 78% (peaks 1.25D -> 0.90D -> 0.65D,
#           hotter first peak than the parent's 1.0D hold), then a blended
#           terminal cycle from 0.40D landing exactly on gamma_min.
#   alpha — DECOUPLED from 1/lr: slow bounded logistic base ramp
#           (0.5 -> 4 alpha0) + per-cycle end-of-cycle restoration bursts
#           (+6 alpha0) + the parent's gated terminal divergence.
#   betas — beta1 = 0.1 throughout (native); beta2 0.2 -> 0.9 at the terminal
#           phase (parent-proven polish conditioning).
_N_CYC = 3.0           # exploration warm-restart cycles
_F_EXPL_END = 0.78     # cycles occupy [0, 0.78]; terminal polish afterwards
_PEAK0 = 1.25          # first-cycle lr peak, in units of D (hotter than parent)
_PEAK_DECAY = 0.72     # each restart peak = 0.72 * previous peak
_LR_FLOOR = 0.08       # per-cycle cosine floor, in units of D
_F_WARM = 0.02         # linear warmup so the feasible grid init survives
_LR_TERM0 = 0.40       # terminal (final) cycle starts at 0.40 * D
_BLEND_LO = 0.78       # smoothstep blend cycles -> terminal over [0.78, 0.86]
_BLEND_W = 0.08
_ALPHA_LO = 0.5        # early base penalty, in units of alpha0
_ALPHA_HI = 4.0        # bounded base plateau, in units of alpha0
_A_CENTER = 0.40       # slow base-alpha logistic ramp center
_A_WIDTH = 0.10
_BURST_ADD = 6.0       # end-of-cycle restoration burst height, units of alpha0
_BURST_CENTER = 0.82   # burst engages late within each cycle (cycle-local frac)
_BURST_WIDTH = 0.05
_GATE_CENTER = 0.90    # parent-proven terminal restoration gate
_GATE_WIDTH = 0.02
_TERM_GAIN = 5.0       # terminal alpha ~ 5*alpha0*D/lr_term (parent scale)
_BETA2_LO = 0.2
_BETA2_HI = 0.9
_B2_CENTER = 0.80      # beta2 ramps up as the terminal phase begins
_B2_WIDTH = 0.04


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    n_total = total_steps * 1.0
    frac = (step + 1.0) / n_total            # traced; (0, 1], hits 1 at last step

    # --- exploration lr: SGDR cosine cycles with decaying peaks ---
    u = jnp.clip(frac / _F_EXPL_END, 0.0, 1.0)
    cyc_pos = u * _N_CYC
    k = jnp.minimum(jnp.floor(cyc_pos), _N_CYC - 1.0)   # cycle index 0..2
    frac_c = cyc_pos - k                                # cycle-local position [0,1]
    peak_k = _PEAK0 * D * jnp.power(_PEAK_DECAY, k)
    floor_lr = _LR_FLOOR * D
    lr_cyc = floor_lr + 0.5 * (peak_k - floor_lr) * (1.0 + jnp.cos(jnp.pi * frac_c))

    # --- terminal lr: final gentle warm restart, half-cosine onto gamma_min ---
    p = jnp.clip((frac - _BLEND_LO) / (1.0 - _BLEND_LO), 0.0, 1.0)
    lr_term = gamma_min + 0.5 * (_LR_TERM0 * D - gamma_min) * (1.0 + jnp.cos(jnp.pi * p))

    # smoothstep blend: exactly cyclic before 0.78, exactly terminal after 0.86,
    # so the last step lands exactly on gamma_min (no residual mixing)
    t = jnp.clip((frac - _BLEND_LO) / _BLEND_W, 0.0, 1.0)
    w_term = t * t * (3.0 - 2.0 * t)
    lr_env = (1.0 - w_term) * lr_cyc + w_term * lr_term

    warm = jnp.minimum(frac / _F_WARM, 1.0)
    lr = lr_env * warm

    # --- alpha: bounded slow base ramp (decoupled from lr) ---
    base = 1.0 / (1.0 + jnp.exp(-(frac - _A_CENTER) / _A_WIDTH))
    alpha_base = alpha0 * (_ALPHA_LO + (_ALPHA_HI - _ALPHA_LO) * base)

    # anti-phased restoration bursts: alpha spikes as each cycle's lr bottoms
    # out (cleaning violations before the next restart), releases at restart;
    # frac_c pins at 1 after the cycles end, keeping bounded pressure through
    # the terminal cool-down
    burst = 1.0 / (1.0 + jnp.exp(-(frac_c - _BURST_CENTER) / _BURST_WIDTH))
    alpha_burst = _BURST_ADD * alpha0 * burst

    # parent-proven terminal feasibility restoration: gated divergence as
    # lr_term -> gamma_min, recovering ~5*alpha0*D/gamma_min at the finish
    gate = 1.0 / (1.0 + jnp.exp(-(frac - _GATE_CENTER) / _GATE_WIDTH))
    alpha_term = _TERM_GAIN * alpha0 * D / jnp.maximum(lr_term, 1e-30)
    alpha = alpha_base + alpha_burst + gate * alpha_term

    # --- betas: native low momentum; beta2 up for the terminal polish ---
    beta1 = 0.1
    b2r = 1.0 / (1.0 + jnp.exp(-(frac - _B2_CENTER) / _B2_WIDTH))
    beta2 = _BETA2_LO + (_BETA2_HI - _BETA2_LO) * b2r

    return lr, alpha, beta1, beta2