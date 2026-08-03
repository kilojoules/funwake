import math

import jax.numpy as jnp

# Structural redesign (gen-4): abandon the single-peak product decay of the
# parent for SGDR warm restarts + a DECOUPLED penalty. The native alpha ~ 1/lr
# coupling is kept ONLY as a terminal feasibility spike:
#   lr    — 3 cosine warm-restart cycles (peaks _C*D, geometrically decaying),
#           then a geometric cool-down that lands exactly on gamma_min.
#   alpha — decoupled: ~alpha0 while exploring, Gaussian feasibility-
#           restoration BURSTS at each cycle end (while lr sits at its floor),
#           a bounded logistic PLATEAU (~40*alpha0, ADMM-style) through the
#           cool-down, and a terminal spike restoring 3x the native
#           alpha0*D/lr divergence over the final ~8% to guarantee
#           strict feasibility.
#   betas — native (0.1, 0.2) while exploring; beta2 ramps to 0.9 in the
#           feasibility phase to absorb the ~alpha constraint curvature.
_C = 1.0                 # peak exploration lr = _C * D (hotter than parent's 0.958)
_F_EXPLORE = 0.68        # fraction of the run spent in the warm-restart phase
_N_CYCLES = 3
_PEAK_DECAY = 0.55       # per-cycle peak decay: 1.0*D, 0.55*D, 0.30*D
_FLOOR_FRAC = 0.12       # each cosine cycle decays to 12% of its own peak
_WARM_DEN = 50           # linear lr warmup over the first ~2% of steps
_A_EXPL = 1.0            # exploration alpha = _A_EXPL * alpha0 (native early scale)
_A_PLAT = 40.0           # bounded cool-down plateau (moderate constant penalty)
_RAMP_CENTER = 0.25      # plateau logistic ramp midpoint, in cool-phase progress s
_RAMP_WIDTH = 0.08
_BURST_GAIN = 8.0        # mid-run restoration bursts: alpha up to ~9*alpha0
_BURST_W = 0.055         # burst width, in within-cycle position
_SPIKE_CENTER = 0.92     # terminal spike engages over the final ~8% of steps
_SPIKE_WIDTH = 0.018
_SPIKE_GAIN = 3.0        # terminal alpha ~ 3x native alpha0*D/gamma_min


def schedule_fn(step, total_steps, D, min_spacing, n_turbines, gamma_min, alpha0):
    lr0 = _C * float(D)
    n_total = int(total_steps)
    n_warm = max(n_total // _WARM_DEN, 1)
    g_min = float(gamma_min)
    # lr handed from the last cycle's floor to the geometric cool-down; both
    # operands are Python floats at trace time, so the log ratio is static.
    lr_hand = _FLOOR_FRAC * lr0 * _PEAK_DECAY ** (_N_CYCLES - 1)
    log_ratio = math.log(g_min / lr_hand)

    f = (step + 1.0) / n_total               # traced run fraction in (0, 1]

    # --- exploration: cosine warm restarts with decaying peaks ------------
    u = jnp.clip(f / _F_EXPLORE, 0.0, 1.0) * _N_CYCLES
    cyc = jnp.clip(jnp.floor(u), 0.0, _N_CYCLES - 1.0)
    pos = u - cyc                             # 0..1 within the current cycle
    peak = lr0 * jnp.power(_PEAK_DECAY, cyc)
    lr_cyc = peak * (_FLOOR_FRAC + (1.0 - _FLOOR_FRAC)
                     * 0.5 * (1.0 + jnp.cos(jnp.pi * pos)))

    # --- cool-down: geometric decay landing exactly on gamma_min ----------
    s = jnp.clip((f - _F_EXPLORE) / (1.0 - _F_EXPLORE), 0.0, 1.0)
    lr_cool = lr_hand * jnp.exp(s * log_ratio)

    explore = f < _F_EXPLORE
    lr_env = jnp.where(explore, lr_cyc, lr_cool)

    # Short warmup so the hotter first peak does not blow apart the feasible
    # grid init; it scales lr only, never alpha.
    warm = jnp.minimum((step + 1.0) / n_warm, 1.0)
    lr = lr_env * warm

    # --- alpha: decoupled bursts + bounded plateau + terminal spike -------
    # Restoration burst at each cycle end, exactly when lr is at its floor:
    # violations accumulated during the hot half-cycle are repaired cheaply.
    burst = _BURST_GAIN * jnp.exp(-0.5 * ((pos - 1.0) / _BURST_W) ** 2)
    alpha_expl = alpha0 * (_A_EXPL + jnp.where(explore, burst, 0.0))

    # Delayed logistic ramp to a bounded plateau through the cool-down —
    # well-conditioned refinement instead of the native diverging coupling.
    ramp = 1.0 / (1.0 + jnp.exp(-(s - _RAMP_CENTER) / _RAMP_WIDTH))
    alpha_plat = alpha0 * (_A_EXPL + (_A_PLAT - _A_EXPL) * ramp)

    # Terminal feasibility restoration: re-introduce the native alpha0*D/lr
    # divergence (x_SPIKE_GAIN) over the final ~8%, so alpha reaches
    # ~3 * alpha0 * D / gamma_min at the last step.
    gate = 1.0 / (1.0 + jnp.exp(-(f - _SPIKE_CENTER) / _SPIKE_WIDTH))
    alpha_term = _SPIKE_GAIN * gate * alpha0 * float(D) / jnp.maximum(lr_env, 1e-30)

    alpha = jnp.where(explore, alpha_expl, alpha_plat + alpha_term)

    # Phase-transition the moments with the alpha phase: native (0.1, 0.2)
    # while exploring, beta2 -> 0.9 during feasibility refinement.
    beta1 = 0.1
    beta2 = jnp.where(explore, 0.2, 0.2 + (0.9 - 0.2) * ramp)
    return lr, alpha, beta1, beta2